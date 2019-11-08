#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:30:38 2019

@author: localuser
"""
from PIL import Image

from model import *
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from matplotlib import pyplot as plt

from keras.utils import plot_model
from keras.losses import binary_crossentropy
from keras import backend as K

import numpy as np


model=load_model('/home/localuser/unet/unet_membrane.hdf5')
plot_model(model, to_file='model_unet.png')

#%matplotlib inline
# load the image
img = load_img('data/membrane/train/image/0.png',color_mode='grayscale')
imgResized = img.resize((256, 256), Image.BICUBIC) 
label = load_img('data/membrane/train/label/0.png',color_mode='grayscale')
labelResized = label.resize((256, 256), Image.BICUBIC) 
plt.figure()
plt.imshow(imgResized,cmap="gray")

plt.figure()
plt.imshow(labelResized,cmap="gray")

img_array= img_to_array(imgResized).reshape(1,256,256,1)/255.0
print(img_array)

out=model.predict(img_array)
print(out.shape)
out=out.reshape(256,256,1)
print('min='+str(np.min(out)))
print('max='+str(np.max(out)))
print('mean='+str(np.mean(out)))

label_array = img_to_array(labelResized)/255.0
print(label_array)
y_true = K.variable(label_array)
y_pred = K.variable(out)

error = K.eval(binary_crossentropy(y_true,y_pred))

plt.figure()
plt.hist(out.ravel(),256,[0,255])
plt.title('Histogram for output picture')

plt.figure()
plt.imshow(array_to_img(out),cmap="gray")

print(error)
print(error.shape)
print(np.min(error))
print(np.max(error))
print(np.mean(error))
outResized = array_to_img(out).resize((512, 512), Image.BICUBIC)
#save_img('predict.png',outResized)
