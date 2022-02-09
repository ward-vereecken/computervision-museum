from typing import List
import cv2
from models.database_image import DatabaseImage
from painting_matching.keypoint_extraction import KeypointExtraction

from models.match import Match

ratio_thresh = 0.75


# Returns an array with the matches found for a given painting in descending order
# The best found match is found at index 0
def match_painting(database: List[DatabaseImage],img) -> List[Match]:
    painting_desc, painting_keypoints = KeypointExtraction.calculate_descriptors_and_keypoints(img)
    # painting_hist = calculate_hist(img)
    matches = []

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for object in database:
        keypoint_matches = calculate_best_desc_matches(painting_desc, object.descriptors)
        # hist_distance = calculate_best_hist_matches(painting_hist, object.hist)
        # if (len(keypoint_matches) > 1):
        matches.append(Match(sum(i.distance for i in keypoint_matches), object))

    return sorted(matches, key=lambda match: match.matches_count, reverse=True)
    # return sorted(matches, key=lambda match: match.hist_distance)


# Returns the distances between every pair of descriptors in the images
def calculate_best_desc_matches(desc_img1, desc_img2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(desc_img1, desc_img2)
    # print(len(matches))
    
    # Filter matches for those that are very close to there respective matched counterpart
    matches = [match for match in matches if match.distance < 40]
    matches = sorted(matches, key=lambda x: x.distance)
    return matches



def calculate_best_hist_matches(hist_img1, hist_img2):
    distance = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
    return distance



def match_painting_sift(database: List[DatabaseImage], img) -> List[Match]:
    img = cv2.bilateralFilter(img,7,30,30)
    painting_desc, painting_keypoints = KeypointExtraction.calculate_descriptors_and_keypoints_sift(img)
    matches = []

    for painting in database:
        keypoint_matches = calculate_best_matches_sift_BF(painting_desc, painting.descriptors)
        matches.append(Match(len(keypoint_matches), painting))

    return sorted(matches, key=lambda match: match.matches_count, reverse=True)


# Calculates the best matches given 2 descriptors of images
# Uses the Lowe's ratio test to filter good matches 
# Each keypoint of the first image is matched with a number of keypoints from the second image.
# The two best matches are kept (k=2) and the ratio test checks if the two distances are sufficiently different.
# If they are not, then the keypoint is eliminated and will not be used for further calculations
def calculate_best_matches_sift_BF(desc_img1, desc_img2):
    bf = cv2.BFMatcher()
    # Apply ratio test
    matches = bf.knnMatch(desc_img1, desc_img2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio_thresh*n.distance:
            good.append([m])
    return good


# Flann makes the program very slow and less accurate than with BFMatcher
def calculate_best_matches_sift_flann(desc_img1,desc_img2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=5)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc_img1,desc_img2,k=2) 
    good = []
    for m,n in matches:
        if m.distance < ratio_thresh*n.distance:
            good.append([m])
    
    return good

# A method that draws the keypoint matches 
def drawMatches(img1, kp1, desc_img1, img2, kp2, desc_img2):
    matches = calculate_best_matches_sift_BF(desc_img1, desc_img2)
    img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.namedWindow("keypoint_matches", cv2.WINDOW_NORMAL)
    cv2.imshow("keypoint_matches", img_matches)
    cv2.waitKey(0)