from my_issue import my_yolov61
import cv2
#import math
#from playsound import playsound
from pygame import mixer


#playsound("audio/warn.mp3",True)
def video_detection1(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    #frame_width=int(cap.get(3))
    #frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

    model=my_yolov61("weights/bestfire.pt","cpu","data/mydatafire.yaml", 640, True)
    #classNames = ['T_shirt', 'camera', 'cap', 'car', 'cell phone', 'clock', 'headphone', 'jean', 'keyboard', 'laptop', 'mouse', 'pant_short', 'pillow', 'trouser', 'truck']
    while True:
        ret, img_src = cap.read()
        if ret:
            img_src , det=model.infer(img_src,conf_thres=0.5, iou_thres=0.35)
            if det!=0:
                model.alert()
                break
        yield img_src
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()