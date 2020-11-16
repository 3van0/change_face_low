from cv2 import cv2
import numpy as np
import math


def superpose(im, box, img_target):

    im_b, im_g, im_r = cv2.split(im)
    im_alpha =  np.ones(im_b.shape, dtype=im_b.dtype)*255
    im = cv2.merge((im_b, im_g, im_r, im_alpha))

    deg = getBoxDegree(box)
    newbox, box_width, box_height = getNewBox(box, 483, 494, deg)
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

    return im


def getBoxDegree(box, origin_deg=None):
    if origin_deg is not None:
        return False
    else:
        point1 = box[0]
        point2 = box[3]
        deg = math.atan2(point2[1]-point1[1], point2[0]-point1[0])*180/math.pi
        if deg < 0:
            deg = deg
        elif deg >0:
            deg = -90+deg
        else:
            deg = 0.0
    return deg

def getBoxWH(box):
    point1 = box[0]
    point2 = box[3]
    point3 = box[1]
    box_width = np.sqrt(np.power(point1[0]-point2[0],2) + np.power(point1[1]-point2[1], 2))
    box_height = np.sqrt(np.power(point1[0]-point3[0],2) + np.power(point1[1]-point3[1], 2))
    return box_width, box_height

def getNewBox(box, target_width, target_height, deg):
    box_width, box_height = getBoxWH(box)
    wh_box = box_width/box_height
    wh_target = target_width/target_height
    if wh_box > wh_target:
        new_box = expandFromBottom(box, box_width, box_width/wh_target, deg)
    else:
        new_box = expandFromBottom(box, box_height*wh_target, box_height, deg)

    box_width, box_height = getBoxWH(new_box)
    return new_box, box_width, box_height

def getSrcArea(img, box):
    xy_min = box.min(axis=0)
    x_min = xy_min[0]
    y_min = xy_min[1]

    xy_max = box.max(axis=0)
    x_max = xy_max[0]
    y_max = xy_max[1]

    return [x_min, y_min, x_max, y_max]

def expandFromBottom(box, width, height, deg):
    point1 = box[0]
    point2 = box[3]
    dist = np.sqrt(np.power(point1[0]-point2[0],2) + np.power(point1[1]-point2[1], 2))
    diff = (width -  dist) / 2
    if(point1[1] == point2[1]):
        box[0] = [int(point1[0] - diff), int(point1[1])]
        box[3] = [int(point2[0] + diff), int(point2[1])]
        point1 = box[0]
        point2 = box[3]
        box[1] = [int(point1[0]), int(point1[1]-height)]
        box[2] = [int(point2[0]), int(point2[1]-height)]
    elif(deg < -45):
        box[0] = [int(point1[0] + diff*math.sin(deg*math.pi/180)),
                    int(point1[1] + diff*math.cos(deg*math.pi/180))]
        box[3] = [int(point2[0] - diff*math.sin(deg*math.pi/180)),
                    int(point2[1] + diff*math.cos(deg*math.pi/180))]
        point1 = box[0]
        point2 = box[3]
        box[1] = [int(point1[0] + height*math.cos(deg*math.pi/180)),
                    int(point1[1] + height*math.sin(deg*math.pi/180))]
        box[2] = [int(point2[0] + height*math.cos(deg*math.pi/180)),
                    int(point2[1] + height*math.sin(deg*math.pi/180))]
    else:
        box[0] = [int(point1[0] - diff*math.cos(deg*math.pi/180)),
                    int(point1[1] - diff*math.sin(deg*math.pi/180))]
        box[3] = [int(point2[0] + diff*math.cos(deg*math.pi/180)),
                    int(point2[1] - diff*math.sin(deg*math.pi/180))]
        point1 = box[0]
        point2 = box[3]
        box[1] = [int(point1[0] + height*math.sin(deg*math.pi/180)),
                    int(point1[1] - height*math.cos(deg*math.pi/180))]
        box[2] = [int(point2[0] + height*math.sin(deg*math.pi/180)),
                    int(point2[1] - height*math.cos(deg*math.pi/180))]
    
    return box


def rotateImg(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255,0))
 