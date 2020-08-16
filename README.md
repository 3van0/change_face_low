# Overlay Face

Overlay a face with  different target images according to the position and posture of the face in an image or video, based on OpenCV and dlib. Simply call it *change face (low level)*.

While there have already been well-performing algorithms for changing faces in an image or a video, the traditional way of overlaying another picture on the face still has its advantages especially when the second picture is not a face image. For example, if the goal is to change a face/head into a cartoon character or a pig head, most AI face changing algorithms won't work well. Another problem with most AI face changing methods is that they do not focus on changing the hair while sometimes hair the hair of the original head is not needed (also look at the case of changing a real man's head to a cartoon character). People have to turn to overlaying the face with target picture in these cases.

This project aims at making this overlaying process easier. 