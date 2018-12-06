import pandas as pd
import numpy as np

def create_train_label(num_frame, data):
    processed = np.zeros(shape=(num_frame, 8))
    for _ in range(len(data[0])):
        if data[2][_] == 'Key':
            processed[data[0][_] - 1:data[1][_], 0] = np.ones(data[1][_] - data[1][_] + 1)
        elif data[2][_] == 'Dice':
            processed[data[0][_] - 1:data[1][_], 1] = np.ones(data[1][_] - data[1][_] + 1)
        elif data[2][_] == 'Spider':
            processed[data[0][_] - 1:data[1][_], 2] = np.ones(data[1][_] - data[1][_] + 1)
        elif data[2][_] == 'Cards':
            processed[data[0][_] - 1:data[1][_], 3] = np.ones(data[1][_] - data[1][_] + 1)
        elif data[2][_] == 'Phone':
            processed[data[0][_] - 1:data[1][_], 4] = np.ones(data[1][_] - data[1][_] + 1)
        elif data[2][_] == 'Map':
            processed[data[0][_] - 1:data[1][_], 5] = np.ones(data[1][_] - data[1][_] + 1)
        elif data[2][_] == 'Face':
            processed[data[0][_] - 1:data[1][_], 6] = np.ones(data[1][_] - data[1][_] + 1)
    for i in range(num_frame):
        if sum(processed[i,:]) == 0:
            processed[i, 7] = 1
    return processed


data = pd.read_csv('/Users/ash/Desktop/rnn_train_1.csv')


processed = np.array([data['On '], data['Off '], data['Label']])

train_label = create_train_label(26547, processed)

train_label.dump('/Users/ash/Downloads/rnn_train_label_1')



# def create_train_input(num_frame, data):
#     processed = np.zeros(shape=(num_frame,3))
#     current_frame = 0
#     x_pos = data[0, 2]
#     y_pos = data[0, 3]
#     confidence = data[0, 1]
#     count = 1
#     for _ in range(data.shape[0]):
#         if data[_,0] == current_frame+1:
#             confidence = confidence + data[_, 1]
#             x_pos = x_pos + data[_, 2]
#             y_pos = y_pos + data[_, 3]
#             count += 1
#         else:
#             processed[current_frame, 0] = confidence / count
#             processed[current_frame, 1] = x_pos / count
#             processed[current_frame, 2] = y_pos / count
#             x_pos = data[_, 2]
#             y_pos = data[_, 3]
#             confidence = data[_, 1]
#             current_frame+=1
#             count = 1
#     return processed
#
# gaze = pd.read_csv('/Users/ash/Desktop/gaze_positions.csv')
# pupil = pd.read_csv('/Users/ash/Desktop/pupil_positions.csv')
#
# processed = np.zeros(shape=(gaze.shape[0],4))
# processed[:,0] = np.array(gaze['index'])
# processed[:,1] = np.array(gaze['confidence'])
# processed[:,2] = np.array(gaze['norm_pos_x'])
# processed[:,3] = np.array(gaze['norm_pos_y'])
#
# processed_p = np.zeros(shape=(pupil.shape[0],4))
# processed_p[:,0] = np.array(pupil['index'])
# processed_p[:,1] = np.array(pupil['diameter'])
# processed_p[:,2] = np.array(pupil['norm_pos_x'])
# processed_p[:,3] = np.array(pupil['norm_pos_y'])
#
#
# rnn_train_input_0 = create_train_input(45712, processed)
#
# rnn_train_input_1 = create_train_input(45712, processed_p)
#
#
# rnn_train_input = np.concatenate((rnn_train_input_0, rnn_train_input_1), axis=1)
#
#
# rnn_train_input.dump('/Users/ash/Downloads/rnn_train_input_2')