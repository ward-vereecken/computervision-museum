import cv2 as cv
from tqdm import tqdm

ORB_OPTIONS = {
    "nfeatures": 500
}

class KeypointExtraction:

    # Calculates the keypoints and descriptors of the image using ORB (FAST keypoint detector and BRIEF descriptor)
    @staticmethod
    def calculate_descriptors_and_keypoints(img, options=ORB_OPTIONS):
        # Grayscale
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create(**options)
        # find the keypoints and descriptors with ORB
        kp, des = orb.detectAndCompute(gray_img, None)
        return des,kp

    # Returns an array of the descriptors from all images in a specific PATH
    # Used to generate the database
    @staticmethod
    def generate_descriptors_from_img_paths(img_paths):
        all_desc = []
        for img_path in tqdm(img_paths):
            # Load the img
            img = cv.imread(img_path)
            # Calculate descriptors
            desc,kp = KeypointExtraction.calculate_descriptors_and_keypoints(img)

            all_desc.append(desc)
        return all_desc


    # Calculates the keypoints and descriptors of the image using SIFT
    @staticmethod
    def calculate_descriptors_and_keypoints_sift(img, options=ORB_OPTIONS):
        # Grayscale
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create(**options)
        kp, des = sift.detectAndCompute(gray_img, None)
        return des, kp


    # Returns an array of the descriptors from all images in a specific PATH
    # Used to generate the database
    @staticmethod
    def generate_descriptors_from_img_paths_sift(img_paths):
        all_desc = []
        all_keypoints = []
        for img_path in tqdm(img_paths):
            # Load the img
            img = cv.imread(img_path)
            # Calculate descriptors
            desc, kp = KeypointExtraction.calculate_descriptors_and_keypoints_sift(img)
            all_keypoints.append(kp)
            all_desc.append(desc)
        return all_desc,all_keypoints

    
