# from my_YOLOv6 import my_yolov6
# import cv2
#
# def video_detection(path_x):
#     video_capture = path_x
#     cap=cv2.VideoCapture(video_capture)
#
#     model=my_yolov6("weights/best_ckpt.pt","cpu","data/mydataset.yaml", 640, True)
#     while True:
#         ret, img_src = cap.read()
#         if ret:
#             img_src , det=model.infer(img_src,conf_thres=0.4, iou_thres=0.35)
#         yield img_src
# cv2.destroyAllWindows()

# from my_yolonew import Inferer
# import cv2
# args  = {
#     "weights": "weights/best_ckpt.pt", # Path to weights file default weights are for nano model
#     "source" : "data/images/image2.jpg", #Path to image file or it can be a directory of image
#     "yaml"   : "data/mydataset.yaml",
#     "img-size": 640, # default image size
#     "conf-thres": 0.25, # confidence threshold for inference.
#     "iou-thres" : 0.45, # NMS IoU threshold for inference.
#     "max-det" : 1000,  # maximal inferences per image
#     "device" : 0,  # device to run our model i.e. 0 or 0,1,2,3 or cpu
#     "save-img" : False,  # save visuallized inference results.
#     "classes" : None, # filter detection by classes
#     "agnostic-nms": False,  # class-agnostic NMS
#     "half" : False,   # whether to use FP16 half-precision inference.
#     "hide-labels" : False,  # hide labels when saving visualization
#     "hide-conf" : False # hide confidences.
#
# }
# inferer = Inferer(weights = args['weights'], device = args['device'], yaml = args['yaml'], img_size = args['img-size'],half = args['half'], conf_thres= args['conf-thres'], iou_thres= args['iou-thres'],classes = args['classes'],
#                   agnostic_nms = args['agnostic-nms'], max_det= args['max-det'])
#
#
# def video_detection(path_x):
#     video_capture = path_x
#     cap = cv2.VideoCapture(video_capture)
#     while True:
#         ret, img_src = cap.read()
#         if ret:
#             img_src1, det = inferer.infer(img_src, conf_thres=0.4, iou_thres=0.35)
#         yield img_src1
#
# cv2.destroyAllWindows()

import cv2
from newyolo import Inferer
import time
import numpy as np

args = {
    "weights": "weights/best_ckpt.pt", # Path to weights file default weights are for nano model
    "source" : "data/images/image2.jpg", #Path to image file or it can be a directory of image
    "yaml"   : "data/mydataset.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.45, # confidence threshold for inference.
    "iou-thres" : 0.5, # NMS IoU threshold for inference.
    "max-det" : 1000,  # maximal inferences per image
    "device" : 0,  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "save-img" : False,  # save visuallized inference results.
    "classes" : None, # filter detection by classes
    "agnostic-nms": False,  # class-agnostic NMS
    "half" : False,   # whether to use FP16 half-precision inference.
    "hide-labels" : False,  # hide labels when saving visualization
    "hide-conf" : False # hide confidences.

}

inferer = Inferer(weights = args['weights'], device = args['device'], yaml = args['yaml'], img_size = args['img-size'],half = args['half'], conf_thres= args['conf-thres'], iou_thres= args['iou-thres'],classes = args['classes'],
                  agnostic_nms = args['agnostic-nms'], max_det= args['max-det'])

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)

    #output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'),30 , (img_src.shape[1],img_src.shape[0]))
    while True:
        ret, img_src = cap.read()
        if ret:
            start = time.time()
            img, img_src = inferer.precess_image(img_src, inferer.img_size, inferer.model.stride, args['half'])
            det = inferer.infer(img, img_src)
            end = time.time() - start
            fps_txt =  "{:.2f}".format(1/end)
            for *xyxy, conf, cls in reversed(det):

              class_num = int(cls)  # integer class
              label = None if args['hide-labels'] else (inferer.class_names[class_num] if args['hide-conf'] else f'{inferer.class_names[class_num]} {conf:.2f}')
              inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=inferer.generate_colors(class_num, True), fps = fps_txt)

            image = np.asarray(img_src)
        yield img_src
cv2.destroyAllWindows()


