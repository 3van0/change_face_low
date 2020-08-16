from cv2 import cv2
from Modules.HeadPose import HeadPose
from Modules.Stabilizer import Stabilizer
import dlib
import numpy as np


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    hp = HeadPose(dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat"), cap)
    while True:
        ret, rotation_vector, translation_vector, euler_angle = hp.readHeadPose()

        if ret:
            im = hp.im
            points = hp.currentPoints.astype(np.int)
            area = hp.currentFace
            left, top, right, bottom = area.left(), area.top(), area.right(), area.bottom()
            # print(rotation_vector)
            R, _ = cv2.Rodrigues(rotation_vector)
            # print(R)
            rot_rect = np.dot(R, np.array([[left, left, right, right],[top, bottom, top, bottom],[0,0,0,0]]))
            rot_rect = np.abs(rot_rect.astype(np.int))

            rot_rect = np.delete(rot_rect, 2, axis=0).T

            print(rot_rect)

            cv2.polylines(im, [rot_rect], True, (255,0,0))

            
           # cv2.rectangle(im, (d_left, d_top), (d_right, d_bottom), (255, 0, 0), -1)

            for point in points:
                cv2.circle(im, (int(point[0]), int(
                    point[1])), 1, (255, 0, 0), thickness=-1)

            cv2.imshow("image-with-points", im)
            cv2.waitKey(1)
    cv2.destroyAllWindow()
