import os 
import cv2
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.utils.misc import Timer
import sys

model_path = 'models/mb1-ssd-kt-Epoch-95-Loss-0.9465917199850082.pth'
label_path = 'models/open-images-model-kt.txt'

class_names = [name.strip() for name in open(label_path).readlines()]

net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)

predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

list_image_path = os.listdir('./data/label')

j = 1

for path in list_image_path:
    path = './data/label/' + path

    # print(path)

    img = cv2.imread(path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 9, 0.3)

    # print(boxes.numpy(), type(boxes.numpy()))
    
    if boxes.numpy().size ==0 :
        continue
    
    for i in range(boxes.size(0)):
         
         box = boxes[i, :]
         box0 = int(box[0])+1
         box1 = int(box[1])+1
         box2 = int(box[2])+1
         box3 = int(box[3])+1
         img_label = img[box1:box3, box0:box2]
         
         height, width = img_label.shape[:2]

         if width == 0 or height ==0:
             continue

         img_label = cv2.resize(img_label, (width, height))
        
         path_label = './data/kt2/' + str(j)+'.jpg'
        #  print(path_label)
         j += 1
         cv2.imwrite(path_label, img_label)


