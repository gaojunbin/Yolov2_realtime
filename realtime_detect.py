import sys
import time
from PIL import Image, ImageDraw
from utils import *
from darknet import Darknet
import cv2

def detect_cv2(m,imgfile,use_cuda):
    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    if use_cuda:
        m.cuda()

    img = imgfile
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    finish = time.time()
    print('Predicted in %f seconds.' % (finish-start))

    class_names = load_class_names(namesfile)
    return plot_boxes_cv2(img, boxes, class_names=class_names)


def real_time():
    cap = cv2.VideoCapture(0)
    cfgfile = "cfg/yolov2.cfg"
    weightfile = "weights/yolo.weights"
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    ret = True
    while ret:
        ret,frame = cap.read()
        frame = detect_cv2(m, frame,False)
        cv2.imshow('实时目标检测',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    real_time()