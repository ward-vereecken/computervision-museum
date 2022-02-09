import cv2

from utils.image_utils import ImageUtils

MINIMUM_CONTRAST = 27.5
MINIMUM_LAPLACIAN = 5

class Frame:

    def __init__(self, frame):
        self.frame = frame
        self.valid_frame = True
        self.__verifyParameters()

    def __verifyParameters(self):
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.__calculateBlurLevel()
        self.__calculateContrastLevel()
        if self.contrastlevel < MINIMUM_CONTRAST:
            self.valid_frame = False

        if self.lapVar < MINIMUM_LAPLACIAN:
            self.valid_frame = False        
    
    #Calculate variance of frame -> Assumption: High variance = Wide spread of responses = Sharp image
    def __calculateBlurLevel(self):
        self.lapVar = cv2.Laplacian(self.gray, cv2.CV_64F).var()

    #Fucntion to calculate the contrast level of a frame (based on the standard deviation)
    def __calculateContrastLevel(self):
        self.contrastlevel = self.gray.std()

    def isValidFrame(self):
        return self.valid_frame

    def findPolygons(self, painting_detector):
        self.polygons = painting_detector.find_paintings(self.frame)
        #Remove polygons that are too small to be a painting
        self.polygons = list(filter(ImageUtils.validateMinArea, self.polygons))
        return self.polygons