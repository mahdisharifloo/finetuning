# -*- coding: utf-8 -*-

import numpy as np
import os
from keras.models import load_model 
from keras.preprocessing import image
import matplotlib.pyplot as plt

#import global variables
img_path = input("[INPUT] image path >>> ")
train_path = 'dataset/train'
#import pretrained model from hdf5 file
model = load_model('mobilenet_finetuned.hdf5')


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()
    return img_tensor




def run():
    new_image = load_image(img_path)
    #create list of training lables
    # train_labels = os.listdir(train_path)
    train_labels = ['سیب', 'موز', 'انبه', 'پرتقال', 'توت فرنگی']
    # sort the training labels
    # train_labels.sort()
    preds = model.predict(new_image)
    pred_lable , pred_percent = preds.argmax(),preds.max()
    prediction = train_labels[pred_lable]
    return prediction,pred_percent

if __name__ == '__main__':
    prediction , percent = run()
    print( prediction ,': ', str(percent*100),'%')
