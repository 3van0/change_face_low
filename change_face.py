from cv2 import cv2
from Modules.HeadPos import HeadPos
from Modules.Stabilizer import Stabilizer
from Modules.superpose import *
import dlib
import numpy as np
import math


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    hp = HeadPos(dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat"), cap)
    img_target = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)

    while True:
        ret, _, _, _ = hp.update()
        if ret != False:
            _, im = hp.getCurrentImage()
            _, steady_box, _ = hp.getFaceBox(expand=True, w_rate=1.2, h_rate=1.5)
            im = superpose(im, steady_box, img_target)
            # cv2.drawContours(im, [steady_box], 0, (0, 0, 255, 100), 3)
            # cv2.drawContours(im, [newbox], 0, (255, 255, 255, 100), 3)
            # cv2.drawContours(im, [normbox], 0, (0, 255, 255, 100), 3)
            # cv2.imshow("target_org", img_target)
            # cv2.imshow("frame", src_frame)
            # cv2.imshow("target", img_target_rt)
            # cv2.rectangle(im, (d_left, d_top), (d_right, d_bottom), (255, 0, 0), -1)

            # for point in points:
            #     cv2.circle(im, (int(point[0]), int(
            #         point[1])), 1, (255, 0, 0), thickness=-1)
        else:
            _, im = hp.getCurrentImage()
        cv2.imshow("TEST", im)
        cv2.imwrite("test_add.png", im)
        cv2.waitKey(1)
    cv2.destroyAllWindow()
