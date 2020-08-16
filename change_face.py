from cv2 import cv2
from Modules.HeadPose import HeadPose
from Modules.Stabilizer import Stabilizer
import dlib
import numpy as np

box_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1) for _ in range(8)]

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
            print("box",box)
            
            steady_box = []
            box = box.flatten()
            for i in range(8):
                stb = box_stabilizers[i]
                stb.update([box[i]])
                steady_box.append(stb.state[0])
            
            steady_box = np.int0(np.reshape(steady_box, (4,2)))
            print("steady:",steady_box)
            
            cv2.drawContours(im,[steady_box], 0, (0, 0, 255), 3)
            

            
           # cv2.rectangle(im, (d_left, d_top), (d_right, d_bottom), (255, 0, 0), -1)

            for point in points:
                cv2.circle(im, (int(point[0]), int(
                    point[1])), 1, (255, 0, 0), thickness=-1)

            cv2.imshow("image-with-points", im)
            cv2.waitKey(1)
    cv2.destroyAllWindow()
