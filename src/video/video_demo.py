from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import imutils
import cv2
from models.demo_match import DemoMatch
from models.match import Match
from painting_matching.descriptors_database import load_database
from painting_localisation.localisation import Localisation
from painting_detection.painting_detector import PaintingDetector
from painting_matching.painting_matching import match_painting_sift
from utils.image_utils import ImageUtils
import numpy as np
import yaml

from video.frame import Frame
from video.floorplan import Floorplan
from utils.canvas_utils import CanvasUtils

# This class is the start point for the demo application
# The TKinter python library is used to visualise the input video,
# extracted paintings and its potential matches.

class VideoDemo:
    def __init__(self, vs, calibration_file = None):
        self.db = load_database()
        self.localisation = Localisation(self.db)
        self.floorplan = Floorplan("resources/floorplan_map.csv")
        self.PaintingDetector = PaintingDetector()
        self.camera_calibration_enabled = False
        self.vs = vs
        self.frameCount = -1
        self.root = tki.Tk()
        self.panel = None
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.__video_loop, args=())
        self.root.geometry("1400x950")
        self.root.configure(bg="#757474")
        self.root.wm_title("Computer Vision: Project assignment")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        self.labels = [None] * 10
        self.labels2 = [None] * 10
        self.buffer = []
        self.images = []
        self.thread.start()

        if calibration_file is not None:
            with open(calibration_file, "r") as f:
                data = yaml.full_load(f)
                self.cameraMatrix = np.array(data["camera_matrix"])
                self.distCoeff = np.array(data["dist_coeff"])
                self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.cameraMatrix, self.distCoeff, np.eye(3), self.cameraMatrix, (1280, 720), cv2.CV_16SC2)
                self.camera_calibration_enabled = True

    #Video initialisation 
    def __video_loop(self):
        #Initialize current room String
        roomVar = tki.StringVar()
        roomLabel = tki.Label(self.root, textvariable=roomVar, font=("Arial", 18))
        roomLabel.grid(row=3, column=0)
        roomVar.set("Current Room: 1")

        #Initialize floorplan
        canvas = tki.Canvas(width=540, height=385)
        canvas.grid(row=2, column=0)
        floorplan_image = tki.PhotoImage(file='resources/floorplan.png')

        try:
            while not self.stopEvent.is_set():
                self.frameCount += 1

                image = self.vs.read()
                if self.camera_calibration_enabled:
                    image = self.__correctFishEye(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                #Init custom Frame object
                frame = Frame(image)
                self.buffer.append(frame)

                #Check whether the frame meets all minimum requirements
                if frame.isValidFrame():
                    #Find paintings on frame & Filter out detections that don't meet the minimum size requirement 
                    polygons = frame.findPolygons(self.PaintingDetector)

                    contoured_image = cv2.drawContours(image.copy(), polygons, -1, (0, 255, 0), thickness=8)

                    #Show frame in tkinter window
                    self.__show_video_frame(contoured_image)
                else:
                    self.__show_video_frame(image)
                
                #Running matching algorithm every x frames & Buffer contains atleast 1 valid frame
                if self.frameCount % 60 == 0 and len(self.buffer) > 0:
                    #Remove existing painting extractions + matches
                    self.__clear_labels()
                    
                    valid_frames = list(filter(lambda x: x.isValidFrame(), self.buffer))
                    if (valid_frames is None or len(valid_frames) < 1):
                        continue
                    
                    #Sort buffer by 'Bluriness'
                    valid_frames.sort(key=lambda x: x.lapVar, reverse=True)
                    best_frame = valid_frames[0]

                    #Warp polygons found on frame
                    found_painitings = [ImageUtils.warp_image(best_frame.frame, poly) for poly in best_frame.polygons]

                    threading.Thread(target=self.__calculate_matches, args=(found_painitings, self.labels, canvas, self.images, floorplan_image, roomVar)).start()
                    
                    #Clear buffer
                    self.buffer.clear()      
        except RuntimeError:
            print("[ERROR] caught a RuntimeError")

    def __calculate_matches(self, found_painitings, labels, canvas, images, floorplan_image, roomVar):
        found_matches = []
        top_matches = None

        for i in range(len(found_painitings)):
            painting = found_painitings[i]
            height, width = painting.shape[:2]
            matches = match_painting_sift(self.db, painting)
            matches_count = matches[0].matches_count
            img_match = None
            #Apply threshold to avoid matches based on a small set of similiraties
            if (matches_count >= 30):
                # Prints the "correctness of the match"
                # print(f"{(matches[0].matches_count / 500)* 100} %")
                img_match = cv2.imread(matches[0].image.absolute_img_path)
                img_match = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
                img_match = cv2.resize(img_match, (width, height))

                if top_matches is None or matches_count > top_matches[0].matches_count:
                    top_matches = matches
            found_matches.append(DemoMatch(matches_count, img_match, painting))
        
        if len(found_matches) > 0:
            found_matches.sort(key=lambda x: x.matches_count, reverse=True)
            for i in range(len(found_matches)):
                match = found_matches[i]
                self.__show_painting(match.painting, labels, i, i+1, 1)
                if match.image is not None:
                    self.__show_painting(match.image, self.labels2, i, i+1, 2)

        if (top_matches is None):
            return
        
        #Forward found matches to the Hidden Markov Model
        localisation = self.localisation.find_location(top_matches)
        top5 = localisation.sort_values(by=['chance'], ascending=False).head(5)

        canvas.delete("all")
        canvas.create_image(0, 0, image=floorplan_image, anchor=tki.NW)
        for index, row in top5.iterrows():
            room = self.floorplan.get_room(index)
            if room is not None:
                CanvasUtils.create_polygon(self.root, canvas, images, room.points, fill='blue', alpha=row['chance'])
            else:
                print(f"[ERROR] Couldn't find room with name: {index}")
        accuracy = round(top5.iloc[0]['chance'] * 100, 4)
        roomVar.set(f"Predicted Room: {top5.index[0]} ({accuracy}%)")

    def __show_painting(self, image, labels, index, x, y):
        scale_percent = 40 # percentage of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 
        tkinter_image = Image.fromarray(resized)
        tkinter_image = ImageTk.PhotoImage(tkinter_image)
        if labels[index] is None:
            labels[index] = tki.Label(self.root, image=tkinter_image, background="#757474")
            labels[index].grid(row=x, column=y)
        else:
            labels[index].configure(image=tkinter_image)
        labels[index].image = tkinter_image

    def __clear_labels(self):
        for label in self.labels + self.labels2:
            if label is not None:
                label.image = None
                label.configure(image=None)

    def __show_video_frame(self, frame):
        frame = imutils.resize(frame, width=600)
        video_frame = Image.fromarray(frame)
        video_frame = ImageTk.PhotoImage(video_frame)

        if self.panel is None:
            self.panel = tki.Label(image=video_frame)
            self.panel.image = video_frame
            self.panel.grid(column=0, row=1, padx=5, pady=5)
        else:
            self.panel.configure(image=video_frame)
            self.panel.image = video_frame

    def __correctFishEye(self, frame):
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def onClose(self):
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()
        self.root.destroy()