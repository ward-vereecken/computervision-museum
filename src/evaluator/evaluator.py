import cv2
import numpy as np
import pandas as pd
import glob
import ast
import re
from statistics import mean
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm
from classifier.classifier import Classifier

from painting_matching.descriptors_database import DEFAULT_DATABASE_FILENAME, load_database
from models.database_image import DatabaseImage
from painting_detection.painting_detector import PaintingDetector
from models.dataset_entry import DataSetEntry
from painting_matching.painting_matching import match_painting_sift
from utils.image_utils import ImageUtils


class Evaluator:

    descriptors_db: List[DatabaseImage]

    def __init__(self, img_dir, dataset_dir):
        self.df = pd.read_csv('resources/database_log.csv')
        self.img_dir = img_dir
        self.dataset_dir = dataset_dir
        self.painting_detector = PaintingDetector()

    def __get_dataset(self, database_path = DEFAULT_DATABASE_FILENAME) -> List[DataSetEntry]:
        img_abs_paths_in_dataset_path = glob.glob(f"./{self.dataset_dir}/**/*.png") + glob.glob(f"./{self.dataset_dir}/**/*.jpg") + glob.glob(f"./{self.dataset_dir}/**/*.jpeg")
        og_path_regex = r"^.+[\/\\]([^\/\\]+)\..+$"
        og_paths = [re.match(og_path_regex, abs_path).group(1) for abs_path in img_abs_paths_in_dataset_path]
        abs_and_og_paths = zip(img_abs_paths_in_dataset_path, og_paths)

        db = load_database(database_path)

        dataset = []

        for abspath, ogpath in abs_and_og_paths:
            matching_db_entries =  [entry for entry in db if entry.og_path == ogpath]
            dataset_entry = DataSetEntry(abspath, matching_db_entries)
            dataset.append(dataset_entry)

        return dataset

    def evaluate_painting_detection(self):
        df_grouped = self.df.groupby(['Photo', 'Room'])

        intersection_over_union = []
        # Paintings missed entirely
        false_negatives = 0
        # Non-paintings extracted
        false_positives = 0

        clf = Classifier(True)

        # Resize size
        width = 600
        height = 800

        rooms = []
        iou_room = []

        previous = ""

        # Iterate over each group
        for image_name, df_group in tqdm(df_grouped):
            #print('Evaluate: {} in {}'.format(image_name[0], image_name[1]))

            if not previous:
                previous = image_name[1]

            if previous != image_name[1]:
                rooms.append(previous)
                iou_room.append(round(np.sum(intersection_over_union)/len(intersection_over_union), 2))
                intersection_over_union.clear()
                previous = image_name[1]

            img = cv2.imread(self.dataset_dir+'/'+image_name[1]+'/'+image_name[0]+'.jpg')

            resize_factor_w = width / img.shape[1]
            resize_factor_h = height / img.shape[0]

            img = cv2.resize(img, (width, height))

            # Get all contours as a array of polygons (4 points)
            predicted_polys = self.painting_detector.find_paintings(img)

            # Sort poly points clockwise: [TopLeft, BottomLeft, BottomRight, TopRight]
            sorted_poly = []
            for poly in predicted_polys:
                if clf.test(ImageUtils.warp_image(img, poly)):
                    sorted_poly.append(ImageUtils.order_points(poly[0]))

            predicted_polys = np.array(sorted_poly)


            if len(predicted_polys) != 0:

                matched_paintings = np.full(len(predicted_polys), False)
                for row_index, row in df_group.iterrows():
                    ground_truth_poly = np.array([ast.literal_eval(row['Top-left']),
                                                  ast.literal_eval(row['Bottom-left']),
                                                  ast.literal_eval(row['Bottom-right']),
                                                  ast.literal_eval(row['Top-right'])])
                    # Apply resize_factor
                    for j in range(ground_truth_poly.shape[0]):
                        ground_truth_poly[j][0] *= resize_factor_w
                        ground_truth_poly[j][1] *= resize_factor_h

                    best_match, place = 0, -1
                    # Let's look for the best matching between ground_truth and detected painting
                    for i in range(len(predicted_polys)):

                        if not matched_paintings[i]:
                            blank = np.zeros(img.shape[0:2])
                            # Copy each of the contours and fill it with 1
                            img1 = cv2.drawContours(blank.copy(), [np.array(predicted_polys[i]).astype(int)], 0, 1, thickness=cv2.FILLED)
                            img2 = cv2.drawContours(blank.copy(), [ground_truth_poly], 0, 1, thickness=cv2.FILLED)

                            # Now AND the two together to get intersection
                            intersection = np.logical_and(img1, img2)
                            # and OR to get union
                            union = np.logical_or(img1, img2)

                            # Calculate IoU
                            iou = np.sum(intersection)/np.sum(union)
                            # Check if best_match yet
                            if iou > best_match:
                                best_match = iou
                                place = i

                    # Check if we found a match and mark as found
                    if place > -1:
                        matched_paintings[place] = True

                intersection_over_union.append(best_match)
                # Calculate false_negatives and positives
                false_negatives += df_group.shape[0] - np.sum(matched_paintings)
                # Count false in matched_array which is equal to count false_positives
                false_positives += np.size(matched_paintings) - np.count_nonzero(matched_paintings)

            else:
                # Missed all paintings
                false_negatives += df_group.shape[0]

        rooms.append(previous)
        iou_room.append(round(np.sum(intersection_over_union) / len(intersection_over_union), 2))
        #print("The average intersection over union: ", np.sum(intersection_over_union)/len(intersection_over_union))
        print("False negatives: ", false_negatives)
        print("False positives: ", false_positives)
        print("Total paints: ", self.df.shape[0])
        fig = plt.figure(figsize=(10, 5))

        # creating the bar plot
        plt.bar(rooms, iou_room, width=0.4)
        plt.xticks(rotation=90)
        plt.xlabel("Room")
        plt.ylabel("IoU ratio")
        plt.title("The IoU ratio per room")
        plt.show()

    def evaluate_matching(self, database_path = DEFAULT_DATABASE_FILENAME, debug = False):

        correct_match_at_nth_positions = []
        counter_1st_position = 0
        counter_top_10_position = 0
        counter_bad_extraction = 0
        counter_no_matches_found = 0
        average_score_correct_match = 0

        painting_detector = PaintingDetector()
        dataset = self.__get_dataset(database_path = database_path)
        db = load_database(database_path)

        # Only test on dataset images that contain one image, otherwise not certain which painting need to be matched
        for entry in [entry for entry in dataset if len(entry.matching_db_entries) == 1]:
            dataset_img = cv2.imread(entry.dataset_img_abs_path)
            polygons = painting_detector.find_paintings(dataset_img)
            polygons = list(filter(ImageUtils.validateMinArea, polygons))
            found_paintings = [ImageUtils.warp_image(dataset_img, poly) for poly in polygons]
            if len(found_paintings) > 0:
                # counter_to_max_testing += 1
                wanted_match = entry.matching_db_entries[0]
                # TODO expecting only one painting so only looking at the first
                # painting = found_paintings[0]
                best_result_index = 801
                for painting in found_paintings:
                    matches = match_painting_sift(db, painting)
                    correct_match_at_nth_position = [match.image.og_path for match in matches].index(wanted_match.og_path) + 1
                    if correct_match_at_nth_position < best_result_index:
                        # Holds the values of the best matched painting + matches
                        best_result_index = correct_match_at_nth_position
                        best_result_matches = matches
                if best_result_index == 1:
                    counter_1st_position += 1
                    average_score_correct_match += (best_result_matches[best_result_index - 1].matches_count/500) * 100
                if best_result_index <= 10:
                    counter_top_10_position += 1
                
                # print(f"{best_result_index}: {best_result_matches[best_result_index - 1].hist_distance} -- {entry.dataset_img_abs_path}")
                if (best_result_matches[best_result_index - 1].matches_count > 0):
                    print(f"Best result index: {best_result_index} \n\t Matches found: {best_result_matches[best_result_index - 1].matches_count} \n\t Amount of keypoints in ori image: {len(wanted_match.keypoints)} \n\t Score on match: {(best_result_matches[best_result_index - 1].matches_count/500) * 100}%  \n\t Image Path: {entry.dataset_img_abs_path}")
                else:
                    print(f"Geen matches gevonden voor absoluut PATH: {entry.dataset_img_abs_path}")
                    counter_no_matches_found += 1
                
                # Through debug tool user can mark bad extractions 

                if (debug == True and best_result_index > 1):
                    cv2.namedWindow("found painting", cv2.WINDOW_NORMAL)
                    cv2.imshow("found painting", painting)
                    cv2.namedWindow("match_painting", cv2.WINDOW_NORMAL)
                    match_img = cv2.imread(matches[best_result_index-1].image.absolute_img_path)
                    cv2.imshow("match_painting", match_img)
                    cv2.waitKey(0)
                    answer_to_bad_extraction = input("Is the painting visible in the extraction? Type Y or N \n")
                    if(answer_to_bad_extraction == "N"):
                        counter_bad_extraction += 1                   
                    cv2.destroyAllWindows()

                correct_match_at_nth_positions.append(best_result_index)

        # print([entry.matching_db_entries for entry in dataset])

        print(f"For every image in dataset with one painting, the correct match was found on average at position {mean(correct_match_at_nth_positions)}")
        print(f"First place: {counter_1st_position}")
        print(f"Top 10 matches: {counter_top_10_position}")
        if (debug == True):
            print(f"Bad extractions: {counter_bad_extraction}")
        print(f"Average score on correct match: {average_score_correct_match/counter_1st_position}%")
        print(f"Number of extracted paintings with zero matches: {counter_no_matches_found}")
        print(f"Number of evaluator checks: {len(correct_match_at_nth_positions)}")
