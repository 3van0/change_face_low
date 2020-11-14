from cv2 import cv2
from Modules.HeadPose import HeadPose
from Modules.Stabilizer import Stabilizer
import dlib
import numpy as np
import math


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    hp = HeadPose(dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat"), cap)
    while True:
            
            ret,_,_,_ = hp.update()
            if ret != False:
                _, im = hp.getCurrentImage()
                _, steady_box = hp.getFaceBox()
                print("steady:",steady_box)
                print(type(im))
                cv2.drawContours(im, [steady_box], 0, (0, 0, 255), 3)
            
            
            # x_min = np.min(box[:][0])
            # x_max = np.max(box[:][0])
            # y_min = np.min

            
           # cv2.rectangle(im, (d_left, d_top), (d_right, d_bottom), (255, 0, 0), -1)

            # for point in points:
            #     cv2.circle(im, (int(point[0]), int(
            #         point[1])), 1, (255, 0, 0), thickness=-1)

                cv2.imshow("image-with-points", im)
                cv2.waitKey(1)
    cv2.destroyAllWindow()
