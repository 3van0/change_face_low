from cv2 import cv2
from Modules.HeadPos import HeadPos
from Modules.Stabilizer import Stabilizer
import dlib
import numpy as np
import math
from superpose import *

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    hp = HeadPos(dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat"), cap)
    img_target = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)

    while True:
        ret, _, _, _ = hp.update()
        if ret != False:
            _, im = hp.getCurrentImage()
            im_b, im_g, im_r = cv2.split(im)
            im_alpha =  np.ones(im_b.shape, dtype=im_b.dtype)*255
            im = cv2.merge((im_b, im_g, im_r, im_alpha))
            _, steady_box, deg = hp.getFaceBox(expand=True, w_rate=1.2, h_rate=1.4)
            # cv2.drawContours(im, [steady_box], 0, (0, 0, 255, 100), 3)

            deg = getBoxDegree(steady_box)
            newbox, box_width, box_height = getNewBox(steady_box, 483, 494, deg)
            img_target_rt = cv2.resize(img_target, (int(box_width), int(box_height)))
            if deg < -45:
                img_target_rt = rotateImg(img_target_rt, 90+deg)
            else:
                img_target_rt = rotateImg(img_target_rt, deg)
            sp_target = img_target_rt.shape
            tg_wd = sp_target[0]
            tg_ht = sp_target[1]
            _,_,_,mask = cv2.split(img_target_rt)
            mask = 255 - mask
            mask = cv2.normalize(mask, mask, 0, 1, cv2.NORM_MINMAX)
            # cv2.imshow("mask",mask)
            normbox = getSrcArea(im, newbox)
            src_frame = im[normbox[1]:normbox[1]+tg_wd, normbox[0]:normbox[0]+tg_ht]
            #frame_add = np.multiply(src_frame, mask) + img_target_rt
            src_frame_b, src_frame_g, src_frame_r, src_frame_alpha = cv2.split(src_frame)
            src_frame_b = src_frame_b * mask
            src_frame_g = src_frame_g * mask
            src_frame_r = src_frame_r * mask
            src_frame_alpha = src_frame_alpha * mask
            src_frame = cv2.merge((src_frame_b, src_frame_g, src_frame_r, src_frame_alpha))
            frame_add = src_frame + img_target_rt
            im[normbox[1]:normbox[1]+tg_wd, normbox[0]:normbox[0]+tg_ht] = frame_add

            normbox = np.array([[normbox[0], normbox[1]], [normbox[2], normbox[1]], [
                                normbox[2], normbox[3]], [normbox[0], normbox[3]]])
            # cv2.drawContours(im, [newbox], 0, (255, 255, 255, 100), 3)
            # cv2.drawContours(im, [normbox], 0, (0, 255, 255, 100), 3)
            cv2.imshow("target_org", img_target)
            cv2.imshow("frame", src_frame)
            cv2.imshow("target", img_target_rt)
        # x_min = np.min(box[:][0])
        # x_max = np.max(box[:][0])
        # y_min = np.min

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
