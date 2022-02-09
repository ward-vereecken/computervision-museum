import cv2
import ast
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
import tqdm
from joblib import dump, load
from skimage.feature import hog
import warnings

warnings.filterwarnings("ignore")

# Block size of the image patch
BLOCK_SIZE = 32

class Classifier:

    def __init__(self, shouldLoad=False, database_path = ""):
        self.database_path = database_path
        # Use the pretrained classifier and scaler
        if shouldLoad:
            self.model = load('resources/classifier.joblib')
            self.scaler = load('resources/scaler.joblib')

    def test(self, img):
        feature_vectors = []

        # Calculate the feature vector of the painting for each pixel
        for i in range(0, img.shape[0] - BLOCK_SIZE, BLOCK_SIZE):
            for j in range(0, img.shape[1] - BLOCK_SIZE, BLOCK_SIZE):
                feature_vectors.append(calculate_hist_vector(img, i, j))
        # If feature vector to small return as wall
        if len(feature_vectors) == 0:
            return 0
        # Transform (normalize)
        feature_vectors = self.scaler.transform(feature_vectors)
        # Predict
        predictions = self.model.predict(feature_vectors)

        # Count the amount of wall pixels and painting pixels in the painting
        painting_features = np.count_nonzero(predictions == 0)
        wall_features = np.count_nonzero(predictions == 1)
        # Threshold to return prediction
        return painting_features*0.6 > wall_features


    def train(self):
        # Get the trainingsset
        csvfile = "resources/database_log.csv"
        df = pd.read_csv(csvfile).head(310)
        df_grouped = df.groupby(['Photo', 'Room'])

        # Resize size
        width = 600
        height = 800

        wall_features = []
        painting_features = []
        # Iterate over each group
        for image_name, df_group in tqdm.tqdm(df_grouped):
            #print('Evaluate: {} in {}'.format(image_name[0], image_name[1]))
            img = cv2.imread(self.database_path + '/' + image_name[1] + '/' + image_name[0] + '.jpg')
            resize_factor_w = width / img.shape[1]
            resize_factor_h = height / img.shape[0]

            img = cv2.resize(img, (width, height))

            mask_image = np.zeros(img.shape[0:2])
            # Make a mask the image to get the ground_truth (serves as label)
            for row_index, row in df_group.iterrows():
                ground_truth_poly = np.array([ast.literal_eval(row['Top-left']),
                                              ast.literal_eval(row['Bottom-left']),
                                              ast.literal_eval(row['Bottom-right']),
                                              ast.literal_eval(row['Top-right'])])
                # Apply resize_factor
                for j in range(ground_truth_poly.shape[0]):
                    ground_truth_poly[j][0] *= resize_factor_w
                    ground_truth_poly[j][1] *= resize_factor_h

                # Make mask
                cv2.drawContours(mask_image, [ground_truth_poly], 0, 1, thickness=cv2.FILLED)

            # Calculate the feature vector of each pixel and save
            for i in range(0, img.shape[0] - BLOCK_SIZE, BLOCK_SIZE):
                for j in range(0, img.shape[1] - BLOCK_SIZE, BLOCK_SIZE):
                    # Check ground_truth to append to the correct save
                    if mask_image[i, j] == 0:
                        wall_features.append(calculate_hist_vector(img, i, j))
                    else:
                        painting_features.append(calculate_hist_vector(img, i, j))

        # Subsample => (good practice for training the classifier)
        length = min([len(painting_features), len(wall_features)])

        # Shuffle
        np.random.shuffle(painting_features)
        np.random.shuffle(wall_features)

        # Subsample
        painting_features = painting_features[0:length-1]
        wall_features = wall_features[0:length-1]
        all_features = painting_features.copy()
        all_features.extend(wall_features)

        # Normalize the data
        scaler = StandardScaler()
        scaler.fit(all_features)
        # Save normalizer
        dump(scaler, 'resources/scaler.joblib')
        wall_features = scaler.transform(wall_features)
        painting_features = scaler.transform(painting_features)

        # Using the Support Vector Machine classifier
        clf = OneVsRestClassifier(svm.SVC(gamma='scale', kernel='rbf', verbose=1))

        X = np.vstack([painting_features, wall_features])

        # Labels
        y = list(np.full(length - 1, 1)) + list(np.full(length - 1, 0))
        y = label_binarize(y, classes=[1, 0])

        # Train test split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Now fitting SVM on training set...')
        # Training SVM
        y_score = clf.fit(x_train, y_train).decision_function(x_test)
        print(y_score)

        print('Now testing on unseen testing set...')
        print(clf.score(x_test, y_test))

        # Predict the labels of the test dataset
        y_pred = clf.predict(x_test)
        # Save classifier
        dump(clf, 'resources/classifier.joblib')
        # Print report
        print(precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1]))
        print(classification_report(y_test, y_pred, target_names=["wall", "painting"]))


# Calculate the feature vector for a give image patch
def calculate_hist_vector(img, i, j):
    imageBlock = img[i: i + BLOCK_SIZE, j: j + BLOCK_SIZE]
    # Color histogram bin size 5
    hist_channel_0 = cv2.calcHist([imageBlock], [0], None, [5], [0, 256])
    hist_channel_1 = cv2.calcHist([imageBlock], [1], None, [5], [0, 256])
    hist_channel_2 = cv2.calcHist([imageBlock], [2], None, [5], [0, 256])

    # Histogram if oriented gradients (hog)
    fd, hog_img = hog(imageBlock, orientations=10, pixels_per_cell=(BLOCK_SIZE, BLOCK_SIZE), cells_per_block=(1, 1),
                      feature_vector=True, visualize=True, block_norm='L2-Hys')
    # Normalize
    hist_channel_0 = cv2.normalize(hist_channel_0, hist_channel_0, norm_type=cv2.NORM_MINMAX)
    hist_channel_1 = cv2.normalize(hist_channel_1, hist_channel_1, norm_type=cv2.NORM_MINMAX)
    hist_channel_2 = cv2.normalize(hist_channel_2, hist_channel_2, norm_type=cv2.NORM_MINMAX)

    # Add values to feature vector
    hist_vector = []
    hist_vector.extend(hist_channel_0)
    hist_vector.extend(hist_channel_1)
    hist_vector.extend(hist_channel_2)
    hist_vector.extend(fd)

    return hist_vector
