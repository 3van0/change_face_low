#! usr/bin/env python3
# __author__ = 3van0
# 2020-8-16

from cv2 import cv2
import numpy as np
import dlib
import math
from Modules.Stabilizer import Stabilizer



class HeadPose:

    detector = dlib.get_frontal_face_detector()
    POINTS_NUM_LANDMARK = 68
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=1) for _ in range(6)]
    im = None

    def __init__(self, predictor, cap, useStabilizer = True):
        self.predictor = predictor
        self.cap = cap
        self.useStabilizer = useStabilizer


    # fetch biggest face
    def _largest_face(self, dets):
        if len(dets) == 1:
            return 0

        face_areas = [ (det.right()-det.left())*(det.bottom()-det.top()) for det in dets]

        largest_area = face_areas[0]
        largest_index = 0
        for index in range(1, len(dets)):
            if face_areas[index] > largest_area :
                largest_index = index
                largest_area = face_areas[index]

        # print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

        return largest_index

    # fetch 6 feature points from all points
    def get_image_points_from_landmark_shape(self, landmark_shape):
        if landmark_shape.num_parts != self.POINTS_NUM_LANDMARK:
            # print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
            return -1, None
        
        #2D image points. If you change the image, you need to change vector
        image_points = np.array([
                                    (landmark_shape.part(30).x, landmark_shape.part(30).y),     # Nose tip
                                    (landmark_shape.part(8).x, landmark_shape.part(8).y),     # Chin
                                    (landmark_shape.part(36).x, landmark_shape.part(36).y),     # Left eye left corner
                                    (landmark_shape.part(45).x, landmark_shape.part(45).y),     # Right eye right corne
                                    (landmark_shape.part(48).x, landmark_shape.part(48).y),     # Left Mouth corner
                                    (landmark_shape.part(54).x, landmark_shape.part(54).y)      # Right mouth corner
                                ], dtype="double")

        return 0, image_points
    
    # fetch points
    def get_image_points(self, img):
                                
        #gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )  # 图片调整为灰色
        dets = self.detector( img, 0 )

        if 0 == len( dets ):
            # print( "ERROR: found no face" )
            return -1, None
        largest_index = self._largest_face(dets)
        face_rectangle = dets[largest_index]
        self.currentFace = face_rectangle
        landmark_shape = self.predictor(img, face_rectangle)
        self.current_landmark = landmark_shape


        return self.get_image_points_from_landmark_shape(landmark_shape)


    # calculate rotation vector and translation vector
    def get_pose_estimation(self, img_size, image_points ):
        # 3D model points.
        model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                 
                                ])
         
        # Camera internals
         
        focal_length = img_size[1]
        center = (img_size[1]/2, img_size[0]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
         
         
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )
     
        return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs

    # transfer to Euler angle
    def get_euler_angle(self, rotation_vector):
        # calculate rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)
        
        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2)*rotation_vector[0] / theta
        y = math.sin(theta / 2)*rotation_vector[1] / theta
        z = math.sin(theta / 2)*rotation_vector[2] / theta
        
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
        
        # 单位转换：将弧度转换为度
        Y = int((pitch/math.pi)*180)
        X = int((yaw/math.pi)*180)
        Z = int((roll/math.pi)*180)
        
        return 0, Y, X, Z


    def readHeadPose(self):
        # require cap as input
        # read from camera and return head pose.

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
         
            ret, image_points = self.get_image_points(self.im)
            self.currentPoints = image_points
            if ret != 0:
                # print('get_image_points failed')
                return False, None, None, None
            
            ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = self.get_pose_estimation(size, image_points)
            
            if ret != True:
                # print('get_pose_estimation failed')
                return False, None, None, None
            
            if self.useStabilizer:
                # Stabilize pose
                pose = (rotation_vector, translation_vector)
                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, self.pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])
                steady_pose = np.reshape(steady_pose, (-1, 3))
                rotation_vector = steady_pose[0]
                translation_vector = steady_pose[1]
            
            ret, pitch, yaw, roll = self.get_euler_angle(rotation_vector)
            if ret != 0:
                # print('get_euler_angle failed')
                return -1, None, None, None
            
            return True, rotation_vector, translation_vector, (pitch, yaw, roll)
        
if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    # Introduce scalar stabilizers for pose.
    hp = HeadPose(dlib.shape_predictor("shape_predictor_68_face_landmarks.dat"), cap)
    while True:
        
        ret, rotation_vector, translation_vector, euler_angle = hp.readHeadPose()
    
        print(translation_vector, euler_angle)
    
