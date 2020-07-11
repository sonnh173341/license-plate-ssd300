from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.utils.misc import Timer
import cv2

import sys
import time
import torch
import os
from keras.models import load_model
import numpy as np
print(torch.version.__version__)


def load_model_detect(model_path, label_path):
    class_names = [name.strip() for name in open(label_path).readlines()]

    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    net.load(model_path)

    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    return predictor

def load_model_classify(model_path):
    model= load_model(model_path)
    return model


def sort_pos_boxes(boxes):
    boxes = boxes[boxes[:, 1].argsort()]
    boxes = boxes.numpy()
    boxes1 = []
    boxes2 = []
    boxes1.append(boxes[0, :])
    for i in range(1, boxes.shape[0]):
        x1, y1, x2, y2 = boxes[1, :]
        xx1, yy1, xx2, yy2 = boxes[i, :]
        if yy1 - y1 >= (y2 - y1)/1.2:
            boxes2.append(boxes[i, :])
        else:
            boxes1.append(boxes[i, :])
            
    boxes1 = np.array(boxes1)
    boxes1 = boxes1[boxes1[:, 0].argsort()]
    boxes2 = np.array(boxes2)

    if boxes2.shape[0] != 0:
        boxes2 = boxes2[boxes2[:, 0].argsort()]

    return boxes1, boxes2
    
def main():

    model_plate = "models/mb1-ssd-plate-Epoch-120-Loss-1.4717974662780762.pth"
    model_kt = 'models/mb1-ssd-kt-Epoch-95-Loss-0.9465917199850082.pth'
    model_classify = 'models/model_kt.h5'

    label_path_plate = 'models/open-images-model-plate.txt'
    label_path_kt = 'models/open-images-model-kt.txt'

    model_detect_plate = load_model_detect(model_plate, label_path_plate)
    model_detect_kt = load_model_detect(model_kt, label_path_kt)

    model_classifier = load_model_classify(model_classify)
    path_img = os.listdir('test_image')
    for path in path_img :
        print(path)

        orig_image = cv2.imread('test_image/' + path)
        # orig_image = cv2.resize(orig_image,None, fx= 0.5, fy= 0.5)
        # cv2.imshow('origin', orig_image)
        # cv2.waitKey()
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        a = time.time()

        boxes, _, probs1 = model_detect_plate.predict(image, 2, 0.2)
        b = time.time()
        print(b-a)
        print(f"Found {len(probs1)} objects.")
        kts = ['F', 'Z', '5', '7', '0', 'H', '3', 'N', 'E', '2', 'U', 'S', 'K', 'L', 'D', 'T', 'B', '9', 'X', 'Y', '8', 'C', '1', 'R', 'V', 'A', '4', 'Q', '6', 'P', 'G', 'M'] 

        for i in range(boxes.size(0)):
            bo_x = boxes[i, :]
            bo_x0 = int(bo_x[0])
            bo_x1 = int(bo_x[1])
            bo_x2 = int(bo_x[2])+1
            bo_x3 = int(bo_x[3])+1
            img_plate = image[bo_x1:bo_x3, bo_x0:bo_x2]
            img_plate = cv2.resize(img_plate,None, fx= 2, fy= 2)
            label_plate_1 = []
            label_plate_2 = []
            # cv2.imshow('img_plate', img_plate)
            # cv2.waitKey()

            boxes1, _, probs1 = model_detect_kt.predict(img_plate, 9, 0.14)
            boxes_1, boxes_2 = sort_pos_boxes(boxes1)
            
            #dong 1
            for j in range(boxes_1.shape[0]):
                box = boxes_1[j, :]
                box0 = int(box[0])
                box1 = int(box[1])
                box2 = int(box[2])+1
                box3 = int(box[3])+1
                img_kt = img_plate[box1:box3, box0:box2]
                # cv2.imshow('kt', img_kt)
                # cv2.waitKey()
                img_kt = cv2.cvtColor(img_kt, cv2.COLOR_RGB2GRAY)
                img_kt = cv2.resize(img_kt, (32, 64))
                
                
                img_kt = img_kt.astype('float32')
                img_kt /= 255
                y_predict = model_classifier.predict(img_kt.reshape(1, 64, 32, 1))

                label_plate_1.append(str(kts[np.argmax(y_predict)]))
            
            #dong 2
            for j in range(boxes_2.shape[0]):
                box = boxes_2[j, :]
                box0 = int(box[0])
                box1 = int(box[1])
                box2 = int(box[2])+1
                box3 = int(box[3])+1
                img_kt = img_plate[box1:box3, box0:box2]
                # cv2.imshow('kt', img_kt)
                # cv2.waitKey()
                img_kt = cv2.cvtColor(img_kt, cv2.COLOR_RGB2GRAY)
                img_kt = cv2.resize(img_kt, (32, 64))
                
                img_kt = img_kt.astype('float32')
                img_kt /= 255
                y_predict = model_classifier.predict(img_kt.reshape(1, 64, 32, 1))

                label_plate_2.append(str(kts[np.argmax(y_predict)]))

            cv2.rectangle(orig_image, (bo_x0, bo_x1), (bo_x2, bo_x3), (255, 255, 0), 1)
            label1 = ""
            label2 = ""
            for text in label_plate_1 :
                label1 = label1 + text
            
            for text in label_plate_2 :
                label2 = label2 + text

            cv2.putText(orig_image, label1, (bo_x[0], bo_x[1] - 30), cv2.FONT_HERSHEY_PLAIN, 1.4, (222, 115, 18), 2)  # line type
            cv2.putText(orig_image, label2, (bo_x[0] - 10, bo_x[1] - 15), cv2.FONT_HERSHEY_PLAIN, 1.4, (222, 115, 18), 2)  # line type
            cv2.imshow('img_success', orig_image)
            cv2.waitKey()

            # bg = np.zeros((100, 250, 3), np.uint8)
            # bg[:] = [255, 255, 255]
            # cv2.putText(bg, label1, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, (222, 115, 18), 2)  # line type
            # cv2.putText(bg, label2, (10, 80), cv2.FONT_HERSHEY_PLAIN, 2.5, (222, 115, 18), 2)  # line type
            # cv2.namedWindow("success")
            # cv2.imshow("success", bg)
            # cv2.waitKey()
            

    
cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

            
        


    

