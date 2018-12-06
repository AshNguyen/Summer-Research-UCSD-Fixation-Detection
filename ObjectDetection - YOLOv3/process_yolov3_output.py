from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import time

class YOLO:
    def __init__(self, o_thres, c_thres):
        self.othresh = o_thres
        self.cthresh = c_thres

        batch_size = 1000
        lambda_noob = 0.5
        lambda_localize = 5.0

        def loss_1(truth, pred, batch=batch_size):
            track = truth[:, 1]
            compl = tf.ones(shape=(batch, 13, 13, 3, 11)) - track
            confidence_loss = tf.reduce_sum(
                (truth[:, 0, :, :, :, 0] - tf.multiply(pred[:, 0, :, :, :, 0], track[:, :, :, :, 0])) ** 2,
                axis=[1, 2, 3]) \
                              + lambda_noob * tf.reduce_sum(
                (tf.multiply(pred[:, 0, :, :, :, 0], compl[:, :, :, :, 0]) - truth[:, 0, :, :, :, 0]) ** 2,
                axis=[1, 2, 3])
            localization_loss = tf.reduce_sum(
                (truth[:, 0, :, :, :, 1:5] - tf.multiply(pred[:, 0, :, :, :, 1:5], track[:, :, :, :, 1:5])) ** 2,
                axis=[1, 2, 3, 4])
            classification_loss = tf.reduce_sum(
                (truth[:, 0, :, :, :, 5:] - tf.multiply(pred[:, 0, :, :, :, 5:], track[:, :, :, :, 5:])) ** 2,
                axis=[1, 2, 3, 4])
            return confidence_loss + classification_loss + lambda_localize * localization_loss

        def loss_2(truth, pred, batch=batch_size):
            track = truth[:, 1]
            compl = tf.ones(shape=(batch, 26, 26, 3, 11)) - track
            confidence_loss = tf.reduce_sum(
                (truth[:, 0, :, :, :, 0] - tf.multiply(pred[:, 0, :, :, :, 0], track[:, :, :, :, 0])) ** 2,
                axis=[1, 2, 3]) \
                              + lambda_noob * tf.reduce_sum(
                (tf.multiply(pred[:, 0, :, :, :, 0], compl[:, :, :, :, 0]) - truth[:, 0, :, :, :, 0]) ** 2,
                axis=[1, 2, 3])
            localization_loss = tf.reduce_sum(
                (truth[:, 0, :, :, :, 1:5] - tf.multiply(pred[:, 0, :, :, :, 1:5], track[:, :, :, :, 1:5])) ** 2,
                axis=[1, 2, 3, 4])
            classification_loss = tf.reduce_sum(
                (truth[:, 0, :, :, :, 5:] - tf.multiply(pred[:, 0, :, :, :, 5:], track[:, :, :, :, 5:])) ** 2,
                axis=[1, 2, 3, 4])
            return confidence_loss + classification_loss + lambda_localize * localization_loss

        def loss_3(truth, pred, batch=batch_size):
            track = truth[:, 1]
            compl = tf.ones(shape=(batch, 52, 52, 3, 11)) - track
            confidence_loss = tf.reduce_sum(
                (truth[:, 0, :, :, :, 0] - tf.multiply(pred[:, 0, :, :, :, 0], track[:, :, :, :, 0])) ** 2,
                axis=[1, 2, 3]) \
                              + lambda_noob * tf.reduce_sum(
                (tf.multiply(pred[:, 0, :, :, :, 0], compl[:, :, :, :, 0]) - truth[:, 0, :, :, :, 0]) ** 2,
                axis=[1, 2, 3])
            localization_loss = tf.reduce_sum(
                (truth[:, 0, :, :, :, 1:5] - tf.multiply(pred[:, 0, :, :, :, 1:5], track[:, :, :, :, 1:5])) ** 2,
                axis=[1, 2, 3, 4])
            classification_loss = tf.reduce_sum(
                (truth[:, 0, :, :, :, 5:] - tf.multiply(pred[:, 0, :, :, :, 5:], track[:, :, :, :, 5:])) ** 2,
                axis=[1, 2, 3, 4])
            return confidence_loss + classification_loss + lambda_localize * localization_loss

        self.model = load_model('/Users/ash/Downloads/trained_model.h5',
                             custom_objects={'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3})
    def predict(self, image):
        start = time.time()
        out = self.model.predict(image)

        def process_output(out, l_confi, c_confi):
            out_1 = out[0]
            out_2 = out[1]
            out_3 = out[2]
            boxes = []
            classes = []
            for i in range(52):
                for j in range(52):
                    if out_3[0, 0, i, j, 0, 0] >= l_confi and np.max(out_3[0, 0, i, j, 0, 5:]) >= c_confi:
                        boxes.append(out_3[0, 0, i, j, 0, 1:5])
                        classes.append(out_3[0, 0, i, j, 0, 5:])

            for i in range(26):
                for j in range(26):
                    if out_2[0, 0, i, j, 0, 0] >= l_confi and np.max(out_2[0, 0, i, j, 0, 5:]) >= c_confi:
                        boxes.append(out_2[0, 0, i, j, 0, 1:5])
                        classes.append(out_2[0, 0, i, j, 0, 5:])

            for i in range(13):
                for j in range(13):
                    if out_1[0, 0, i, j, 0, 0] >= l_confi and np.max(out_1[0, 0, i, j, 0, 5:]) >= c_confi:
                        boxes.append(out_1[0, 0, i, j, 0, 1:5])
                        classes.append(out_1[0, 0, i, j, 0, 5:])
            return boxes, classes

        boxes, classes = process_output(out, self.othresh, self.cthresh)
        print('Time', time.time()-start)
        return boxes, classes

model = YOLO(0.1, 0.5)
img = cv2.imread('/Users/ash/Desktop/test_2.jpg')
image = cv2.imread('/Users/ash/Desktop/test_2.jpg')
image = cv2.resize(image, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
image = np.array(image, dtype='float32')
image /= 255.
image = np.expand_dims(image, axis=0)
boxes, classes = model.predict(image)

def draw(image, boxes, classes):
    for box, score in zip(boxes,classes):
        x, y, w, h = box[0], box[1], box[2], box[3]
        x, y, w, h = x * 1280, y * 720, (w**2) * 1280, (h**2) * 720
        top = max(0, np.floor(x - w/2 + 0.5).astype(int))
        left = max(0, np.floor(y + h/2 +  0.5).astype(int))
        right = min(1280, np.floor(x + w/2 + 0.5).astype(int))
        bottom = min(720, np.floor(y - h/2 + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (251, 125, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(np.argmax(score), np.max(score)),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

draw(img, boxes, classes)

cv2.imwrite('/Users/ash/Downloads/detected.png', img)