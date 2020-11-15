from cv2 import cv2
import numpy as np
import math

# def superpose(img, box, target):

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
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255,0))
 