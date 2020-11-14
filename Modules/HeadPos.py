#!usr/bin/env python3
# __author__ = 3van0
# 2020-8-16

from cv2 import cv2
import numpy as np
import dlib
import math
from Modules.Stabilizer import Stabilizer

class HeadPos:

    detector = dlib.get_frontal_face_detector()
    posStabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=1) for _ in range(6)]
    boxStabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.05,
    cov_measure=0.05) for _ in range(8)]

    def __init__(self, predictor, cap, useStabilizer = True, pointsNum = 68):
        self.predictor = predictor
        self.cap = cap
        self.useStabilizer = useStabilizer
        self.POINTS_NUM_LANDMARK = pointsNum
        self.im = None

    # fetch 6 feature points from all points
    def getImagePointsFromLandmarkShape(self, landmarkShape):
        if landmarkShape.num_parts != self.POINTS_NUM_LANDMARK:
            # print("ERROR:landmarkShape.num_parts-{}".format(landmarkShape.num_parts))
            return -1, None
        
        #2D image points. If you change the image, you need to change vector
        imagePoints = np.array([
                                    (landmarkShape.part(30).x, landmarkShape.part(30).y),     # Nose tip
                                    (landmarkShape.part(8).x, landmarkShape.part(8).y),     # Chin
                                    (landmarkShape.part(36).x, landmarkShape.part(36).y),     # Left eye left corner
                                    (landmarkShape.part(45).x, landmarkShape.part(45).y),     # Right eye right corne
                                    (landmarkShape.part(48).x, landmarkShape.part(48).y),     # Left Mouth corner
                                    (landmarkShape.part(54).x, landmarkShape.part(54).y)      # Right mouth corner
                                ], dtype="double")

        return 0, imagePoints

    # fetch points
    def getImagePoints(self, img):
        dets = self.detector( img, 0 )

        if 0 == len( dets ):
            # print( "ERROR: found no face" )
            return -1, None
        largestIndex = self._largestFace(dets)
        faceRectangle = dets[largestIndex]
        landmarkShape = self.predictor(img, faceRectangle)
        self.currentLandmark = landmarkShape

        return self.getImagePointsFromLandmarkShape(landmarkShape)

    def update(self):
        # read from camera and return head pose.
        self.isUpdated = False
        if (self.cap.isOpened()):
            
            # Read Image
            ret, self.im = self.cap.read()
            if ret != True:
                # print('read frame failed')
                return False, None, None, None
            size = self.im.shape
            
            if size[0] > 10000:
                h = size[0] / 3
                w = size[1] / 3
                self.im = cv2.resize(self.im, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
                size = self.im.shape
         
            ret, imagePoints = self.getImagePoints(self.im)
            self.currentPoints = imagePoints
            if ret != 0:
                # print('getImagePoints failed')
                return False, None, None, None
            
            ret, rotationVector, translationVector, cameraMatrix, dist_coeffs = self.getPoseEstimation(size, imagePoints)
            
            if ret != True:
                # print('getPoseEstimation failed')
                return False, None, None, None
            
            if self.useStabilizer:
                # Stabilize pose
                pose = (rotationVector, translationVector)
                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, self.posStabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])
                steady_pose = np.reshape(steady_pose, (-1, 3))
                rotationVector = steady_pose[0]
                translationVector = steady_pose[1]
            
            ret, pitch, yaw, roll = self.getEulerAngle(rotationVector)
            if ret != 0:
                # print('getEulerAngle failed')
                return -1, None, None, None
            
            self.isUpdated = True
            self.currentData = (rotationVector, translationVector, [pitch, yaw, roll])
            return True, rotationVector, translationVector, [pitch, yaw, roll]

    def getCurrentImage(self):
        if self.im is not None:
            return True, self.im
        else:
            return False, None
    
    def getFaceBox(self, expand=False):
        if (self.isUpdated):
            rotation_vector, translation_vector, eulerAngle = self.currentData
            im = self.im
            points = self.currentPoints.astype(np.int)
            all_marks = self.currentLandmark.parts()
            #all_marks = np.array(all_marks)
            all_points = np.empty((0,2), dtype="int")

            for mark in all_marks:
                #print(mark)
                all_points = np.append(all_points, [[mark.x, mark.y]], axis=0)

            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print("box: ",box)
            print("deg: ",rect[2])
            deg = rect[2]
            if(rect[2]<-45):
                box_tmp = np.empty((4,2), dtype="int")
                box_tmp[0]=box[1]
                box_tmp[1]=box[2]
                box_tmp[2]=box[3]
                box_tmp[3]=box[0]
                box = box_tmp
                # print(box)
            
            if(rect[2]<-45):
                height = rect[1][0]
                width = rect[1][1]
            else:
                width = rect[1][0]
                height = rect[1][1]

            if(expand):
                height = height*3/2
                # print(height, width)

                exp_point1, exp_point2 = self.expandWidth(box[0], box[3], 20, deg)
                exp_point3, exp_point4 = self.expandHeight(exp_point1, exp_point2, height, deg)
                box_tmp = np.empty((4,2), int)
                box_tmp[0] = exp_point1
                box_tmp[1:3] = [exp_point3, exp_point4]
                box_tmp[3] = exp_point2
                box = box_tmp

            
            steady_box = []
            box = box.flatten()
            
            for i in range(8):
                stb = self.boxStabilizers[i]
                stb.update([box[i]])
                steady_box.append(stb.state[0])
            
            steady_box = np.int0(np.reshape(steady_box, (4,2)))
            return True, steady_box
        else:
            return False, None
    
    @staticmethod
    # fetch biggest face
    def _largestFace(dets):
        if len(dets) == 1:
            return 0

        faceAreas = [ (det.right()-det.left())*(det.bottom()-det.top()) for det in dets]

        largestArea = faceAreas[0]
        largestIndex = 0
        for index in range(1, len(dets)):
            if faceAreas[index] > largestArea :
                largestIndex = index
                largestArea = faceAreas[index]

        # print("largest_face index is {} in {} faces".format(largestIndex, len(dets)))

        return largestIndex

    @staticmethod
    # calculate rotation vector and translation vector
    def getPoseEstimation(imgSize, imagePoints ):
        # 3D model points.
        modelPoints = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                 
                                ])
         
        # Camera internals
        focalLength = imgSize[1]
        center = (imgSize[1]/2, imgSize[0]/2)
        cameraMatrix = np.array(
                                 [[focalLength, 0, center[0]],
                                 [0, focalLength, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
         
         
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotationVector, translationVector) = cv2.solvePnP(modelPoints, imagePoints, cameraMatrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )
     
        return success, rotationVector, translationVector, cameraMatrix, dist_coeffs

    @staticmethod
    # transfer to Euler angle
    def getEulerAngle(rotationVector):
        # calculate rotation angles
        theta = cv2.norm(rotationVector, cv2.NORM_L2)
        
        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2)*rotationVector[0] / theta
        y = math.sin(theta / 2)*rotationVector[1] / theta
        z = math.sin(theta / 2)*rotationVector[2] / theta
        
        ysqr = y * y
        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        # print('t0:{}, t1:{}'.format(t0, t1))
        pitch = math.atan2(t0, t1)
        
        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)
        
        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)
        
        # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
        
        # convert to degree
        Y = int((pitch/math.pi)*180)
        X = int((yaw/math.pi)*180)
        Z = int((roll/math.pi)*180)
        
        return 0, Y, X, Z

    @staticmethod
    # expand box width
    def expandWidth(point1, point2, diff, deg):
        if(point1[1]==point2[1]):
            return([int(point1[0] - diff), int(point1[1])], [int(point2[0] + diff), int(point2[1])])
        elif(deg<-45):
            point3 = [int(point1[0] + diff*math.sin(deg*math.pi/180)), int(point1[1] + diff*math.cos(deg*math.pi/180))]
            point4 = [int(point2[0] - diff*math.sin(deg*math.pi/180)), int(point2[1] + diff*math.cos(deg*math.pi/180))]
        else:
            point3 = [int(point1[0] - diff*math.cos(deg*math.pi/180)), int(point1[1] - diff*math.sin(deg*math.pi/180))]
            point4 = [int(point2[0] + diff*math.cos(deg*math.pi/180)), int(point2[1] - diff*math.sin(deg*math.pi/180))]
        return(point3, point4)

    @staticmethod
    # expand box height
    def expandHeight(point1, point2, height, deg):
        if(point1[1]==point2[1]):
            return([int(point1[0]), int(point1[1]-height)], [int(point2[0]), int(point2[1]-height)])
        elif(deg<-45):
            point3 = [int(point1[0] + height*math.cos(deg*math.pi/180)), int(point1[1] + height*math.sin(deg*math.pi/180))]
            point4 = [int(point2[0] + height*math.cos(deg*math.pi/180)), int(point2[1] + height*math.sin(deg*math.pi/180))]
        else:
            point3 = [int(point1[0] + height*math.sin(deg*math.pi/180)), int(point1[1] - height*math.cos(deg*math.pi/180))]
            point4 = [int(point2[0] + height*math.sin(deg*math.pi/180)), int(point2[1] - height*math.cos(deg*math.pi/180))]
        return(point3, point4)
    
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    hp = HeadPos(dlib.shape_predictor("shape_predictor_68_face_landmarks.dat"), cap)
    while (True):
        
        img = hp.im
     
        ret, imagePoints = hp.getImagePoints(img)
        ret, rotationVector, translationVector, eulerAngle = hp.readHeadPos()
        ret, pitch, yaw, roll = hp.getEulerAngle(rotationVector)
        eulerAngle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
        print(eulerAngle_str)
        
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
         
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotationVector, translationVector, cameraMatrix, dist_coeffs)
         
        for p in imagePoints:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
         
         
        p1 = ( int(imagePoints[0][0]), int(imagePoints[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
         
        cv2.line(im, p1, p2, (255,0,0), 2)
         
        # Display image
        #cv2.putText( im, str(rotationVector), (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1 )
        cv2.putText( im, eulerAngle_str, (0, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1 )
        cv2.imshow("Output", im)
        cv2.waitKey(1)
