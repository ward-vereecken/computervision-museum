import cv2
import numpy as np
import yaml

#This class is responsible for calculating the parameters intrinsic to the less
#of a provided calibrataion video.

class Calibration:

    def __init__(self):
        self.frames = []

    def calibrate(self, file_path, checkboardSize):
        #Open provided video file
        cap = cv2.VideoCapture(file_path)
        i=0
        #Since only a small number of images is required to calculate the parameters, build a subset of video frames
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            if (i % 60 == 0):
                self.frames.append(frame)
            i+=1

        CHECKERBOARD = checkboardSize
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW

        #Object points = how the object should look in the real world
        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        _img_shape = None
        for img in self.frames:
            if _img_shape == None:
                _img_shape = img.shape[:2]
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find corners on the chessboard where 2 black squares touch each other
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)

        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        #Use the built-in opencv function to calculate the intrinsic camera parameters
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")

        # transform the matrix and distortion coefficients to writable lists
        data = {'camera_matrix': np.asarray(K).tolist(),
                'dist_coeff': np.asarray(D).tolist()}

        with open("resources/calibration_matrix.yaml", "w") as f:
            yaml.dump(data, f)

        cap.release()
        cv2.destroyAllWindows()