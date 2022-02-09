from classifier.classifier import Classifier
from evaluator.evaluator import Evaluator
from painting_matching.descriptors_database import generate_and_save_descriptors_dataset, load_database
from painting_detection.painting_detector import PaintingDetector
from painting_matching.painting_matching import match_painting_sift
from video.video_demo import VideoDemo
from imutils.video import FileVideoStream
import time
import argparse
import os

from video.calibration import Calibration

parser = argparse.ArgumentParser(description='Recognition based indoor positioning')
subparsers = parser.add_subparsers(dest='subparser_name')

parser_video = subparsers.add_parser('video', help='Start Demo Video')
parser_video.add_argument("-f", help="Path to video", dest="videopath", required=True)
parser_video.add_argument("-c", help="Path to calibration file", dest="calibrationpath")

parser_databse = subparsers.add_parser('generate_database', help='Generate keypoints database')
parser_databse.add_argument("-d", help="Path to image database directory", dest="databasepath", required=True)

parser_calibration = subparsers.add_parser('calibrate', help='calibrate gopro video distortion')
parser_calibration.add_argument("-f", help="Path to calibration video", dest="calibrationpath", required=True)
parser_calibration.add_argument("-x", help="Amount of squares on the x-axis", dest="x", required=True)
parser_calibration.add_argument("-y", help="Amount of squares on the y-axis", dest="y", required=True)

parser_evaluate = subparsers.add_parser('evaluate_matcher_on_imgs_with_one_painting', help="")
parser_evaluate.add_argument("-d", help="Path to image database directory", dest="databasepath", required=True)
parser_evaluate.add_argument("-m", help="Path to dataset directory", dest="datasetpath", required=True)

parser_classifier = subparsers.add_parser('train_classifier', help='Train classifier')
parser_classifier.add_argument("-d", help="Path to image database directory", dest="databasepath", required=True)

parser_evaluate = subparsers.add_parser('evaluate_painting_extraction', help="")
parser_evaluate.add_argument("-d", help="Path to image database directory", dest="databasepath", required=True)
parser_evaluate.add_argument("-m", help="Path to dataset directory", dest="datasetpath", required=True)

args = parser.parse_args()

#py .\src\main.py video -f Computervisie_2020_Project_Database/video/MSK_15.mp4 -c resources/calibration_matrix.yaml
#py .\src\main.py video -f Computervisie_2020_Project_Database/video/MSK_15.mp4
if args.subparser_name == "video":
    if not os.path.isfile(args.videopath):
        print("Video file doesn't exist!")
        os._exit(1)

    if args.calibrationpath is not None and not os.path.isfile(args.calibrationpath):
        print("Calibration file doesn't exist!")
        os._exit(1)
            
    vs = FileVideoStream(args.videopath).start()
    time.sleep(2.0)
    pba = VideoDemo(vs, args.calibrationpath)
    pba.root.mainloop()

#py .\src\main.py generate_database -d Computervisie_2020_project_Database/Database
elif args.subparser_name == "generate_database":
    if not os.path.isdir(args.databasepath):
        print("Database directory doesn't exist!")
        os._exit(1)

    generate_and_save_descriptors_dataset(args.databasepath)

#py .\src\main.py calibrate -f .\Computervisie_2020_Project_Database\video\calibration\calibration_W.mp4 -x 10 -y 6
elif args.subparser_name == "calibrate":
    if not os.path.isfile(args.calibrationpath):
        print("Calibration file doesn't exist!")
        os._exit(1)

    if not args.x.isdigit() or not args.y.isdigit():
        print("Invalid checkboard size provided")
        os._exit(1)

    calibration = Calibration()
    calibration.calibrate(args.calibrationpath, (int(args.y), int(args.x)))

#py .\src\main.py evaluate_matcher_on_imgs_with_one_painting -d Computervisie_2020_project_Database/Database -m Computervisie_2020_project_Database/dataset_pictures_msk
elif args.subparser_name == "evaluate_matcher_on_imgs_with_one_painting":
    if not os.path.isdir(args.databasepath):
        print("Database directory doesn't exist!")
        os._exit(1)

    if not os.path.isdir(args.datasetpath):
        print("Dataset directory doesn't exist!")
        os._exit(1)
    
    evaluator = Evaluator(args.databasepath, args.datasetpath)
    evaluator.evaluate_matching(debug=False)

#py .\src\main.py train_classifier -d Computervisie_2020_project_Database/dataset_pictures_msk
elif args.subparser_name == "train_classifier":
    if not os.path.isdir(args.databasepath):
        print("Database directory doesn't exist!")
        os._exit(1)

    classifier = Classifier(False, args.databasepath)
    classifier.train()

#py .\src\main.py evaluate_painting_extraction -d Computervisie_2020_project_Database/Database -m Computervisie_2020_project_Database/dataset_pictures_msk
elif args.subparser_name == "evaluate_painting_extraction":
    if not os.path.isdir(args.databasepath):
        print("Database directory doesn't exist!")
        os._exit(1)

    evaluator = Evaluator(args.databasepath, args.datasetpath)
    evaluator.evaluate_painting_detection()

os._exit(0)