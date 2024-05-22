import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from grabscreen import grab_screen
import cv2
import urllib.request

import keys as k
import time

keys=k.Keys({})

import tensorflow_text
tf.contrib

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

def download_and_extract_model():
    if not os.path.exists(PATH_TO_CKPT):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
        print("Model extracted.")
    else:
        print("Model already exists. No need to extract.")

download_and_extract_model()

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_width, im_height, 3)).astype(np.uint8)

WIDTH = 480
HEIGHT = 270

#take the mouse and aim it towards of object
#where we are looking for the screen on our monitor
#relatively how much do we need to move our mouse around
def determine_movement(mid_x, mid_y, width=WIDTH, height=HEIGHT-15):
    #the center of the screen : 0.5, 0.5
    #the center of the object : mid_x, mid_y
    x_move = 0.5 - mid_x  # the distance from the center of the object
    y_move = 0.5 - mid_y

    #how much to move
    hm_x = x_move / 0.5
    hm_y = y_move / 0.5

    # Move the mouse based on the calculations 
        #decimal to integer
                                                                        #(X, Y)
                                                                        #the reason why negative is flipping the sign of both tof values
                                                                        #that should be where remove the mouse
    keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x000, -1*int(hm_x*width), -1*int(hm_y*width)))

if not os.path.exists(PATH_TO_CKPT):
    print("Model file not found")
else:
    print("Model file found, loading...")

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.70)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        stolen = False
        while True:
            screen = cv2.resize(grab_screen(region=(0, 45, 1280, 768)), (1280, 720))
            image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            #x1,y1, x2, y2
            #zero to one
            #boxes[0] : [[3.51673365e-03 4.09537554e-03 9.89943802e-01 1.00000000e+00], ...] 100th, contain the coordinates
            #scores[0] : [0.32287654 0.22373985 0.1553109  0.14345004 0.11081722 0.10204417, ...] 100th
            #classes[0] : [77. 84. 73. 84. 72. 14. ...] 100th
            #num_detections : 100.0 100.0...

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            #check the classes of those boxes
            #if you know the exact size of that object, you could calculate the exact distance of that object
            #measure the amount of pixels in between x1 to x2 that how wide is that
            #from there we can get a relative determination of how far away that object is
            vehicle_dict = {}
            for i, b in enumerate(boxes[0]):

                #https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
                #car, bus, truck
                if classes[0][i] in [3, 6, 8]:
                    if scores[0][i] > 0.5:

                        #where is the middle point of this object
                        #middle point : (boxes[0][i][3]+boxes[0][i][1])/2
                        mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2  # percentages
                        mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2

                        #the reason why one minus is it'll be smaller as it gets closer
                        #the reason why power of 4 is that will give a litt le more granularity
                        #height : boxes[0][i][3]-boxes[0][i][1]
                        #the reason why round 3 is we are less likely to get duplicate distances.
                        apx_distance = round((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4, 3)  # percentages
                        vehicle_dict[apx_distance] = [mid_x, mid_y, scores[0][i]]
                        
                        #need to put pixel, not percentage, int won't take a float
                        #put the approximate relative distance on the car
                        #cv2 works in BGR
                        cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x * WIDTH), int(mid_y * HEIGHT)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        if apx_distance <= 0.5:
                            if mid_x > 0.3 and mid_x < 0.7:
                                cv2.putText(image_np, 'WARNING!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            #Acquiring a Vehicle for the Agent
            #if the agent does not have a vehicle, we want to be able to steal a vehicle : find the vehicle and steal it
            if len(vehicle_dict) > 0:
                if not stolen:
                    closest = sorted(vehicle_dict.keys())[0]
                    vehicle_choice = vehicle_dict[closest]

                    #approach the car
                    determine_movement(mid_x=vehicle_choice[0], mid_y=vehicle_choice[1], width=WIDTH, height=HEIGHT - 15)
                    if closest < 0.1:
                        keys.directKey("w", keys.key_release)
                        keys.directKey("f")
                        time.sleep(0.05)
                        keys.directKey("f", keys.key_release)
                        stolen = True
                    else:
                        keys.directKey("w")

            cv2.imshow('window', cv2.resize(image_np, (1280, 720)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
