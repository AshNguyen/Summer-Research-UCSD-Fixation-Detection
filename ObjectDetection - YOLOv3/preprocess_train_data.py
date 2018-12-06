import numpy as np
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import string
import math

'''
Process training images
'''



'''
Process training labels
'''

# key, dice, spider, cards, phone, map

height = 720
width = 1280


def cvat_xml_parser(xml_path):  # label_map_path):

    '''
	:param: xml_path, label_map_path
	:output: frames_data, size

	Finds the path to the xml file outputted by CVAT annotation tool and parses through.
	Organizes the bounding box coordinates by xmin, ymin, xmax, class_text, and class number, which
	is then put in a dictionary sorting the arrays into a dictionary for each frame. The dictionary is
	then put into an array consisting of all the frames ordered by frame number.
    '''

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # label_map_dict = label_map_util.get_label_map_dict(label_map_path)#gets the labels from label map
    size = int(root.find('meta').find('task').find('size').text)  # parses through xml file to find number of frames
    frames_data = [None] * (size + 1)  # the +1 is for interpolation mode

    # initializes list of dictionaries
    for i in range(0, size + 1):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []

        # each index of list will have a dictionary holding information about the bounding boxes in that frame
        frame_dic = {'xmins': xmins, 'xmaxs': xmaxs, 'ymins': ymins, 'ymaxs': ymaxs,
                     'classes_text': classes_text}

        frames_data[i] = frame_dic

    # parses through the xml file to acquire data from each bounding box
    for track in root.iter('track'):

        # get the label for each track
        label = track.attrib['label'].lower().translate(str.maketrans('', '', string.punctuation))

        # get all the bboxes in the track and append to the specified index in the list of dictionaries
        for bbox in track.findall('box'):
            frame_id = int(bbox.attrib['frame'])
            xmin = float(bbox.attrib['xtl']) / width
            xmax = float(bbox.attrib['xbr']) / width
            ymin = float(bbox.attrib['ytl']) / height
            ymax = float(bbox.attrib['ybr']) / height
            class_text = label
            class_text = class_text.encode('utf-8')
            frames_data[frame_id]['xmins'].append(xmin)
            frames_data[frame_id]['xmaxs'].append(xmax)
            frames_data[frame_id]['ymins'].append(ymin)
            frames_data[frame_id]['ymaxs'].append(ymax)
            frames_data[frame_id]['classes_text'].append(class_text)

    return frames_data, size

def preprocess_label(box_data, N):
    processed = np.zeros(shape=(2,N,N,3,11))
    col_row = np.floor(box_data[:,0:2] * N)
    for _ in range(box_data.shape[0]):
        processed[0, int(col_row[_,0]), int(col_row[_,1]), 0, :] = box_data[_,:]
        processed[1, int(col_row[_,0]), int(col_row[_,1]), 0, :] = np.ones(shape=(1,11))
    return processed

def xywh_transform(raw):
    processed = np.zeros(shape=raw.shape)
    processed[:, 0] = (raw[:, 0] + raw[:, 1]) / 2
    processed[:, 1] = (raw[:, 2] + raw[:, 3]) / 2
    processed[:, 2] = np.sqrt(np.absolute(raw[:, 1] - raw[:, 0]))
    processed[:, 3] = np.sqrt(np.absolute(raw[:, 3] - raw[:, 2]))
    processed[:, 4:] = raw[:, 4:]
    return processed

def manual_label(label):
    processed = np.zeros(shape=6)
    if label == b'key':
        processed[0] = 1
    if label == b'dice':
        processed[1] = 1
    if label == b'spider':
        processed[2] = 1
    if label == b'cards':
        processed[3] = 1
    if label == b'phone':
        processed[4] = 1
    if label == b'map':
        processed[5] = 1
    return processed

def dict_parse(item):
    length = len(item['xmins'])
    data = np.zeros(shape=(length, 11))
    for _ in range(1,length,2):
        data[_, 1] = item['xmins'][_]
        data[_, 2] = item['xmaxs'][_]
        data[_, 3] = item['ymins'][_]
        data[_, 4] = item['ymaxs'][_]
        data[_, 0] = 1
        data[_, 5:] = manual_label(item['classes_text'][_])
    return data

def process_label(frame_data, size, N):
    processed = np.zeros(shape=(size,2,N,N,3,11))
    for _ in range(1,size+1,2):
        raw_0 = dict_parse(frame_data[_])
        raw_1 = xywh_transform(raw_0)
        raw_2 = preprocess_label(raw_1, N)
        processed[_] = raw_2
    return processed

frames, size = cvat_xml_parser('/Users/ash/Downloads/19_andy.xml')
print(size)
print(frames[0])
print(frames[24510])
# size = 100.
#
# label_13 = process_label(frames, int(size), 13)
# label_13.dump('/Users/ash/Downloads/label_13')
# label_26 = process_label(frames, int(size), 26)
# label_26.dump('/Users/ash/Downloads/label_26')
# label_52 = process_label(frames, int(size), 52)
# label_52.dump('/Users/ash/Downloads/label_52')
#
# data = preprocess_image(int(size))
# data.dump('/Users/ash/Downloads/images')

# print(dict_parse(frames[0]))
# print(xywh_transform(dict_parse(frames[0])))