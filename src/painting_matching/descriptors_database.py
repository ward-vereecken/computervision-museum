import base64
from collections import namedtuple

import cv2
import pickle
from typing import List

import pandas as pd
from pandas.core.frame import DataFrame
import pathlib
import re
from warnings import warn

from painting_matching.keypoint_extraction import KeypointExtraction
from models.database_image import DatabaseImage

# In this file the database is created and saved to a JSON object file
# The database contains all the information about the paintings
# Each painting has the following attributes:
# Room, ogPath, paintingNmbr, imagePath, descriptors, keypoints
# Each painting is transformed to a DatabaseImage object

FILENAME_PATTERN = r"[zZ]aal_((?:I+)|(?:[0-9]+)|(?:[A-Z]))__(.+)__([0-9]+)"
DEFAULT_DATABASE_FILENAME = "resources/desc_dataset.json"

def generate_and_save_descriptors_dataset(img_dir):
    df = generate_descriptors_and_hist_dataframe(img_dir)
    df.to_json(DEFAULT_DATABASE_FILENAME)

# Converts the dataframe data to DatabaseImages
def convert_dataframe_to_DatabaseImages(df):
    return [DatabaseImage(row["room"], row["ogPath"], row["paintingNmbr"], row["imagePath"], row["descriptors"], row["keypoints"]) for index, row in df.iterrows()]

# Returns a List of DatabaseImage objects for every image in the database
def load_database(in_dataset_file = DEFAULT_DATABASE_FILENAME) -> List[DatabaseImage]:
    df = load_descriptors_and_hist_dataframe(in_dataset_file)
    return convert_dataframe_to_DatabaseImages(df)
        
# Returns the JSON object to df
def load_descriptors_and_hist_dataframe(in_dataset_file) -> DataFrame:
    df = pd.read_json(in_dataset_file)
    df = load_descriptors(df)
    # df = load_hist(df)
    df = load_keypoints(df)
    return df


def load_descriptors(df):
    df["descriptors"] = [pickle.loads(base64.b64decode(encodedDescriptors)) for encodedDescriptors in df["descriptors"]]
    return df

def load_hist(df):
    df["hist"] = [pickle.loads(base64.b64decode(encodedHist)) for encodedHist in df["hist"]]
    return df

def load_keypoints(df):
    df["keypoints"] = [convertTupletoKeypoint(tuplekp) for tuplekp in df["keypoints"]]
    return df

def convertKeypointToTuple(kp: cv2.KeyPoint):
    return [[kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in kp]

def convertTupletoKeypoint(tup):
    return [cv2.KeyPoint(x=tup[0][0], y=tup[0][1], _size=tup[1], _angle=tup[2], _response=tup[3], _octave=tup[4], _class_id=tup[5]) for tup in tup]

def generate_descriptors_and_hist_dataframe(img_dir):
    imgDataInfos = get_images_info(img_dir)
    descriptors, keypts = KeypointExtraction.generate_descriptors_from_img_paths_sift([imgDataInfo.imagePath for imgDataInfo in imgDataInfos])
    # Create dataframe which will hold all the data
    df = pd.DataFrame(imgDataInfos, columns=['room', 'ogPath', 'paintingNmbr', 'imagePath'])

    df["descriptors"] = [base64.b64encode(pickle.dumps(descriptor)).decode('ascii') for descriptor in descriptors]
    # df["hist"] = [base64.b64encode(pickle.dumps(hist)).decode('ascii') for hist in hists]
    df["keypoints"] = [convertKeypointToTuple(kp) for kp in keypts]
    return df


# Extracts all the usefull info for each image in a given directory (DATABASE)
# The info is embedded in the name of the image file
def get_images_info(dir_path):
    Fileinfo = namedtuple("Fileinfo", ['room', 'ogPath', 'paintingNmbr', 'imagePath'])
    currentDirectory = pathlib.Path(dir_path)

    imageInfos = []

    # Iterate over all files in the given directory
    for currentFile in currentDirectory.iterdir():
        match = re.match(FILENAME_PATTERN, currentFile.stem)

        if not match:
            warning = f"{currentFile.name} does not match the expected pattern"
            warn(warning)
        else:
            room, og_path, paintingNmbr = match.groups()
            imageInfos.append(Fileinfo(room, og_path, paintingNmbr, str(currentFile.absolute())))
    return imageInfos