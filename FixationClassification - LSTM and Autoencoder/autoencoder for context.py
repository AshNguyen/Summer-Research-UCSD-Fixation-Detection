# Autoencoder
import numpy as np
import cv2
from keras.models import Model, load_model

def preprocess_image_1():
    images_data = np.zeros(shape=(1, 512, 512, 3))
    path = '/Users/ash/Desktop/test.jpg'
    image = cv2.imread(path) / 225
    image = cv2.resize(image, (512, 512))
    images_data[0] = image
    return images_data

def preprocess_image_2(position):
    images_data = np.zeros(shape=(1, 512, 512, 3))
    path = '/home/radlab/Desktop/Video_2/_Image_{0:05d}.jpg'.format(position+1)
    image = cv2.imread(path) / 225
    image = cv2.resize(image, (512, 512))
    images_data[0] = image
    return images_data


autoencoder = load_model('/Users/ash/Downloads/DATA_RNN/autoencoder.h5')
encoder = Model(input=autoencoder.inputs, output=autoencoder.layers[15].output)

o_image = preprocess_image_1()
p_image = autoencoder.predict(o_image)

cv2.imwrite('/Users/ash/Desktop/p_image.png', p_image[0]*225)
cv2.imwrite('/Users/ash/Desktop/o_image.png', o_image[0]*225)

# n = 26547
# frames_1 = np.zeros(shape=(n, 16*16*4))
#
# for _ in range(n):
#     o_image = preprocess_image_1(_)
#     encoded = encoder.predict(o_image)
#     frames_1[_,:] = np.reshape(encoded, (16*16*4))
#
#
# m = 45712
# frames_2 = np.zeros(shape=(m, 16*16*4))
#
# for _ in range(m):
#     o_image = preprocess_image_2(_)
#     encoded = encoder.predict(o_image)
#     frames_2[_,:] = np.reshape(encoded, (16*16*4))
#
# frames = np.concatenate((frames_1, frames_2), axis=0)
# frames.dump('/Users/ash/Downloads/frames')