
"""
Comparison of Face Detectors
"""
import os

import cv2
#import dlib
from mtcnn.mtcnn import MTCNN
import numpy as np

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

detector1 = MTCNN()
#detector2 = dlib.get_frontal_face_detector()
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
classifier2 = cv2.CascadeClassifier('models/haarcascade_frontalface2.xml')

mtcount = 0
dlcount = 0
dnncount = 0
hrcount = 0

#Webcam
#cap = cv2.VideoCapture(0)
#Linked Video
cap = cv2.VideoCapture('vdo.mp4')

font = cv2.FONT_HERSHEY_SIMPLEX
while (True):
    ret, img = cap.read()
    if ret == True:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        height, width = img.shape[:2]
        img1 = img.copy()
        img2 = img.copy()
        img3 = img.copy()
        # detect faces in the image
        faces1 = detector1.detect_faces(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #faces2 = detector2(gray, 1)
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                     1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces3 = net.forward()
        faces4 = classifier2.detectMultiScale(img)

        # display faces on the original image
        for result in faces1:
            x, y, w, h = result['box']
            x1, y1 = x + w, y + h
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img, 'mtcnn', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # for result in faces2:
        #     x = result.left()
        #     y = result.top()
        #     x1 = result.right()
        #     y1 = result.bottom()
        #     cv2.rectangle(img1, (x, y), (x1, y1), (0, 0, 255), 2)
        # cv2.putText(img1, 'dlib', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        for i in range(faces3.shape[2]):
            confidence = faces3[0, 0, i, 2]
            if confidence > 0.5:
                box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img2, 'dnn', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        for result in faces4:
            x, y, w, h = result
            x1, y1 = x + w, y + h
            cv2.rectangle(img3, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img3, 'haar', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        h1 = cv2.hconcat([img3, img1])
        h2 = cv2.hconcat([img, img2])
        fin = cv2.vconcat([h1, h2])

        # Total Num of MTCNN Faces
        cv2.putText(img, 'N: ' + str(len(faces1)), (200, 80), font, 1, (255, 0, 0), 2)
        if int(len(faces1)) == 1:
            mtcount = mtcount + 1
            print('MTCNN:', mtcount)

        # Total Num of DLIB Faces
        # cv2.putText(img1, 'N: ' + str(len(faces2)), (200, 80), font, 1, (255, 0, 0), 2)
        # if int(len(faces2)) == 1:
        #      dlcount = dlcount + 1
        #      print('DLIB:', dlcount)

        # Total Num of DNN Faces
        cv2.putText(img2, 'N: ' + str(len(faces3)), (200, 80), font, 1, (255, 0, 0), 2)
        if int(len(faces3)) == 1:
            dnncount = dnncount + 1
            print('DNN:', dnncount)

        # Total Num of HAAR Faces
        cv2.putText(img3, 'N: ' + str(len(faces4)), (200, 80), font, 1, (255, 0, 0), 2)
        if int(len(faces4)) == 1:
            hrcount = hrcount + 1
            print('HAAR:', hrcount)

        cv2.imshow("MTCNN", img)
        #cv2.imshow("DLIB", img1)
        cv2.imshow("DNN", img2)
        cv2.imshow("HAAR", img3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('Total DNN:', dnncount)

        print('Total MTCNN:', mtcount)

        print('Total HAAR:', hrcount)

        # Total Num of Frames
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Total Number of Frames:', length)
        break


cap.release()
cv2.destroyAllWindows()
