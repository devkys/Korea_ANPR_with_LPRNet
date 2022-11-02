import os
import cv2
import random
import numpy as np
import tensorflow as tf

from core.utils import read_class_names
from core.config import cfg

from lpr.model import LPRNet
from lpr.loader import resize_and_normailze

# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            #cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + '_' + str(counts[class_name]) + '.jpg'
            img_path = os.path.join(path, img_name)
            # save image
            cv2.imwrite(img_path, cropped_img)
            lpr(img_path)
            print(img_path)
            print("lprlprlpr: " + lpr(img_path))
            return lpr(img_path)
        else:
            continue


classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
              "가", "나", "다", "라", "마", "거", "너", "더", "러",
              "머", "버", "서", "어", "저", "고", "노", "도", "ho",
              "모", "보", "소", "오", "조", "구", "누", "두", "루",
              "무", "부", "수", "우", "주", "허", "하", "호"
              ]

def lpr(img):
    # t = time()
    args = {'weights' : './lpr/weights_best.pb'}

    #tf.compat.v1.enable_eager_execution()
    net = LPRNet(len(classnames) + 1)
    net.load_weights(args["weights"])

    img = cv2.imread(img)
    #print(img)

    x = np.expand_dims(resize_and_normailze(img), axis=0)

    print("predict: ", net.predict(x, classnames))
    result = net.predict(x, classnames)
    str_result = ' '.join(s for s in result)
    print("str_result type: ", type(str_result))
    return str_result

    #cv2.imshow("lp", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

