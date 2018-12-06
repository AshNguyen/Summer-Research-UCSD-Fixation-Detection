from keras.models import load_model
from keras import layers
from keras.layers import Reshape, Conv2D
import keras.backend as K
from keras.models import Model
import math
import keras
import numpy as np
import tensorflow as tf


model = load_model('/Users/ash/Downloads/floydhub/floydhub_datasets/yolo.h5')

new_output_0 = Conv2D(filters=33, kernel_size=(1,1), padding='same', activation='linear', name='cus_1')(model.layers[269].output)
new_output_0 = Reshape((1,13,13,3,11), name='output_1')(new_output_0)
new_output_1 = Conv2D(filters=33, kernel_size=(1,1), padding='same', activation='linear', name='cus_2')(model.layers[270].output)
new_output_1 = Reshape((1,26,26,3,11), name='output_2')(new_output_1)
new_output_2 = Conv2D(filters=33, kernel_size=(1,1), padding='same', activation='linear', name='cus_3')(model.layers[271].output)
new_output_2 = Reshape((1,52,52,3,11), name='output_3')(new_output_2)
new_output = [new_output_0,new_output_1,new_output_2]
customized_yolov3 = Model(inputs=model.input, outputs=new_output)

customized_yolov3.summary()
#
# lambda_noob = 0.5
# lambda_localize = 5.0
# batch_size = 1000
#
# def loss_1(truth, pred, batch=batch_size):
#     track = truth[:,1]
#     compl = tf.ones(shape=(batch,13,13,3,11)) - track
#     confidence_loss = tf.reduce_sum(
#         (truth[:, 0, :, :, :, 0] - tf.multiply(pred[:, 0, :, :, :, 0], track[:, :, :, :, 0])) ** 2, axis=[1, 2, 3]) \
#                       + lambda_noob * tf.reduce_sum(
#         (tf.multiply(pred[:, 0, :, :, :, 0], compl[:, :, :, :, 0]) - truth[:, 0, :, :, :, 0]) ** 2, axis=[1, 2, 3])
#     localization_loss = tf.reduce_sum(
#         (truth[:, 0, :, :, :, 1:5] - tf.multiply(pred[:, 0, :, :, :, 1:5], track[:, :, :, :, 1:5])) ** 2,
#         axis=[1, 2, 3, 4])
#     classification_loss = tf.reduce_sum(
#         (truth[:, 0, :, :, :, 5:] - tf.multiply(pred[:, 0, :, :, :, 5:], track[:, :, :, :, 5:])) ** 2,
#         axis=[1, 2, 3, 4])
#     return confidence_loss + classification_loss + lambda_localize * localization_loss
#
# def loss_2(truth, pred, batch=batch_size):
#     confidence_loss = tf.reduce_sum(
#         (truth[:, 0, :, :, :, 0] - tf.multiply(pred[:, 0, :, :, :, 0], track[:, :, :, :, 0])) ** 2, axis=[1, 2, 3]) \
#                       + lambda_noob * tf.reduce_sum(
#         (tf.multiply(pred[:, 0, :, :, :, 0], compl[:, :, :, :, 0]) - truth[:, 0, :, :, :, 0]) ** 2, axis=[1, 2, 3])
#     localization_loss = tf.reduce_sum(
#         (truth[:, 0, :, :, :, 1:5] - tf.multiply(pred[:, 0, :, :, :, 1:5], track[:, :, :, :, 1:5])) ** 2,
#         axis=[1, 2, 3, 4])
#     classification_loss = tf.reduce_sum(
#         (truth[:, 0, :, :, :, 5:] - tf.multiply(pred[:, 0, :, :, :, 5:], track[:, :, :, :, 5:])) ** 2,
#         axis=[1, 2, 3, 4])
#     return confidence_loss + classification_loss + lambda_localize * localization_loss
#
#
# def loss_3(truth, pred, batch=batch_size):
#     track = truth[:, 1]
#     compl = tf.ones(shape=(batch, 52, 52, 3, 11)) - track
#     confidence_loss = tf.reduce_sum(
#         (truth[:, 0, :, :, :, 0] - tf.multiply(pred[:, 0, :, :, :, 0], track[:, :, :, :, 0])) ** 2, axis=[1, 2, 3]) \
#                       + lambda_noob * tf.reduce_sum(
#         (tf.multiply(pred[:, 0, :, :, :, 0], compl[:, :, :, :, 0]) - truth[:, 0, :, :, :, 0]) ** 2, axis=[1, 2, 3])
#     localization_loss = tf.reduce_sum(
#         (truth[:, 0, :, :, :, 1:5] - tf.multiply(pred[:, 0, :, :, :, 1:5], track[:, :, :, :, 1:5])) ** 2,
#         axis=[1, 2, 3, 4])
#     classification_loss = tf.reduce_sum(
#         (truth[:, 0, :, :, :, 5:] - tf.multiply(pred[:, 0, :, :, :, 5:], track[:, :, :, :, 5:])) ** 2,
#         axis=[1, 2, 3, 4])
#     return confidence_loss + classification_loss + lambda_localize * localization_loss
#     track = truth[:, 1]
#     compl = tf.ones(shape=(batch, 26, 26, 3, 11)) - track



#
#
# for layer in customized_yolov3.layers[0:272]:
#     layer.trainable = False
#
# customized_yolov3.compile(optimizer='adam', loss={'output_1': loss_1, 'output_2': loss_2, 'output_3': loss_3},
#                           loss_weights={'output_1':1., 'output_2':1., 'output_3':1.})
#
#
#
# # x_train = np.load('/Users/ash/Downloads/images')
# # y_1 = np.load('/Users/ash/Downloads/label_13')
# # y_2 = np.load('/Users/ash/Downloads/label_26')
# # y_3 = np.load('/Users/ash/Downloads/label_52')
# #
# # customized_yolov3.fit(x_train, {'output_1': y_1, 'output_2':y_2, 'output_3': y_3}, batch_size= 10, epochs=5)


