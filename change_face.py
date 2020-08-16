from cv2 import cv2
from Modules.HeadPose import HeadPose
from Modules.Stabilizer import Stabilizer
import dlib
import numpy as np
import math

box_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.05,
    cov_measure=0.05) for _ in range(8)]


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
    hp = HeadPose(dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat"), cap)
    while True:
        ret, rotation_vector, translation_vector, euler_angle = hp.readHeadPose()

        if ret:
            im = hp.im
            points = hp.currentPoints.astype(np.int)
            all_marks = hp.current_landmark.parts()
            #all_marks = np.array(all_marks)
            all_points = np.empty((0,2), dtype="int")

            for mark in all_marks:
                #print(mark)
                all_points = np.append(all_points, [[mark.x, mark.y]], axis=0)
            #print(all_points)



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
                print(box)
            
            if(rect[2]<-45):
                height = rect[1][0]
                width = rect[1][1]
            else:
                width = rect[1][0]
                height = rect[1][1]


            height = height*3/2
            print(height, width)

            exp_point1, exp_point2 = expandWidth(box[0], box[3], 20, deg)
            exp_point3, exp_point4 = expandHeight(exp_point1, exp_point2, height, deg)
            box_tmp = np.empty((4,2), int)
            box_tmp[0] = exp_point1
            box_tmp[1:3] = [exp_point3, exp_point4]
            box_tmp[3] = exp_point2
            box = box_tmp
            
            steady_box = []
            box = box.flatten()
            
            for i in range(8):
                stb = box_stabilizers[i]
                stb.update([box[i]])
                steady_box.append(stb.state[0])
            
            steady_box = np.int0(np.reshape(steady_box, (4,2)))
            
            print("steady:",steady_box)
            
            cv2.drawContours(im,[steady_box], 0, (0, 0, 255), 3)
            
            
            x_min = np.min(box[:][0])
            x_max = np.max(box[:][0])
            y_min = np.min

            
           # cv2.rectangle(im, (d_left, d_top), (d_right, d_bottom), (255, 0, 0), -1)

            for point in points:
                cv2.circle(im, (int(point[0]), int(
                    point[1])), 1, (255, 0, 0), thickness=-1)

            cv2.imshow("image-with-points", im)
            cv2.waitKey(1)
    cv2.destroyAllWindow()
