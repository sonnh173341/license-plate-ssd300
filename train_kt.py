# Import Libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np
import pickle
import os
import cv2
np.random.seed(1234)

def read_data(root):
    label_listdir = os.listdir(root)
    train_data = []
    test_data = []
    train_label = []
    test_label = []

    for label in label_listdir :
        path_list_img = os.listdir(root+ '/' + str(label))

        for path_img in path_list_img :
            s = root+ '/' + str(label) + '/' + str(path_img)
            img = cv2.imread(s, 0)
            img = cv2.resize(img, (64, 64))
            img = img.astype('float32')
            img /= 255

            a = np.random.randint(100)
            if a <= 80 :
                train_data.append(img)
                train_label.append(str(label))
            else :
                test_data.append(img)
                test_label.append(str(label))

    return  train_data, train_label, test_data, test_label

def label_one_hot(phase_label, labels):
    label_phase = []
    for label1 in phase_label:
        index = 0
        zeros = np.zeros((32, ))
        for label in labels:
            if label1 == label:
                zeros[index] = 1
                label_phase.append(zeros)
            else:
                index +=1
    return label_phase

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = read_data('./data/data-kt')

    labels = []
    paths = os.listdir('./data/data-kt')
    for path in os.listdir('./data/data-kt'):
        labels.append(str(path))
    print(labels)
    #['F', 'Z', '5', '7', '0', 'H', '3', 'N', 'E', '2', 'U', 'S', 'K', 'L', 'D', 'T', 'B', '9', 'X', 'Y', '8', 'C', '1', 'R', 'V', 'A', '4', 'Q', '6', 'P', 'G', 'M']
    train_label = label_one_hot(train_label, labels)
    test_label = label_one_hot(test_label, labels)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    pickle.dump(train_data, open("./data/pickle/train_data.pkl","wb"))
    pickle.dump(train_label, open("./data/pickle/train_label.pkl","wb"))
    pickle.dump(test_data, open("./data/pickle/test_data.pkl","wb"))
    pickle.dump(test_label, open("./data/pickle/test_label.pkl","wb"))



    # train_data = pickle.load(open("./data/pickle/train_data.pkl","rb"))
    # train_label = pickle.load(open("./data/pickle/train_label.pkl","rb"))
    # test_data = pickle.load(open("./data/pickle/test_data.pkl","rb"))
    # test_label = pickle.load(open("./data/pickle/test_label.pkl","rb"))
    # print(train_data.shape)
    # print(train_label.shape)
    # print(test_data.shape)
    # print(test_label.shape)

