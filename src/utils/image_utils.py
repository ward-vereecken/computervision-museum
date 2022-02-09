import numpy as np
import cv2

class ImageUtils:

    @staticmethod
    def order_points(pts):
        # Sort the points based on their x-coordinates
        x_sorted = pts[np.argsort(pts[:, 0]), :]

        # Grab the left-most and right-most points from the sorted x-coordinates points
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]

        # Sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left points, respectively
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most

        # Sort the right-most coordinates according to their
        # y-coordinates so we can grab the top-right and bottom-right points, respectively
        right_most = right_most[np.argsort(right_most[:, 1]), :]
        (tr, br) = right_most

        # Return the coordinates in correct order
        return np.array([tl, tr, br, bl], dtype="float32")
    
    @staticmethod
    def warp_image(img, polygon):
        src = np.array(polygon[0], np.float32)
        src = ImageUtils.order_points(src)
        (tl, tr, br, bl) = src

        # Compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordinates or the top-right and top-left x-coordinates
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # Compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], np.float32)

        # Compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (max_width, max_height))
        return warped
    
    #https://en.wikipedia.org/wiki/Shoelace_formula
    @staticmethod
    def validateMinArea(polygon):
        corners = polygon[0]
        n = len(corners) # of corners
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs(area) / 2.0
        return area > (85 * 85)