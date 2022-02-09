import numpy as np
import cv2

class PaintingDetector:

    # Converts img to grayscale, applies bilateral filter and canny edge detection
    def __apply_filter_and_canny(self, img):
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

        # Apply blur to remove less possible gradient changes and preserve the more intense ones
        # using the bilateral filter: it preserves edges/corners of high gradient change but
        # blurs regions that have minimal gradient changes.
        gray = cv2.bilateralFilter(gray, 15, 10, 10)

        ret2, th2 = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find Canny edges
        edged = cv2.Canny(th2, 20, 120)

        return edged

    # Dilates the canny lines so finding contours is easier
    def __dilate_edges(self, img, iterations):
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
        dilated_img = cv2.dilate(img, dilation_kernel, iterations)
        return dilated_img

    # Fills up all contour polygons, inverts and dilates img
    def __invert_fill_dilate2(self, img, contours):
        cv2.fillPoly(img, pts=contours, color=(255, 255, 255))
        inv_dilate = cv2.bitwise_not(img)
        inv_dilate = self.__dilate_edges(inv_dilate, 10)
        inv_dilate = cv2.bitwise_not(inv_dilate)
        return inv_dilate


    # Fills up all contour polygons and erodes noise away
    def __fill_and_erode(self, img, contours):
        cv2.fillPoly(img, pts=contours, color=(255, 255, 255))
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10), (-1, -1))
        img = cv2.erode(img, element)
        return img


    # Checks which contours are possible paintings
    def __detect_candidates(self, copy, poly_copy, contours):
        polygons = []

        # First, find polygon with biggest area in the image
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area


        # Now, take all contours with area equal to or bigger than biggest area * 0.35
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= max_area * 0.02:
                max_area = area
                # For each such contour, calculate number of corners in approximated polygon
                approx = cv2.approxPolyDP(cnt, 0.07 * cv2.arcLength(cnt, True), True)

                # If number of corners equals 4, contour is very likely to be painting
                if len(approx) == 4:
                    array = np.array([[[approx[0][0][0], approx[0][0][1]],
                                    [approx[1][0][0], approx[1][0][1]],
                                    [approx[2][0][0], approx[2][0][1]],
                                    [approx[3][0][0], approx[3][0][1]]]]
                                    , np.int32)
                    cv2.polylines(copy, array, True, (255, 0, 0), 3)
                    # self.warp_image(poly_copy, array)
                    polygons.append(array)
        return polygons

    def find_paintings(self, img):
        copy = img.copy()
        poly_copy = img.copy()

        edged = self.__apply_filter_and_canny(img)
        dilated_img = self.__dilate_edges(edged, 1)
        
        # Find contours in dilated image
        contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Fill up all contour polygons, invert grayscale image and dilate again
        inv_dilate = self.__fill_and_erode(dilated_img.copy(), contours)

        # Now, look for contours again
        contours, hierarchy = cv2.findContours(inv_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Check which contours are possible paintings
        polygons = self.__detect_candidates(copy, poly_copy, contours)

        return polygons
