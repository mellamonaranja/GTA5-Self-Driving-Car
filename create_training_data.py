import numpy as np
import cv2
import time
import pyautogui
from PIL import ImageGrab

from grabscreen import grab_screen
from getkeys import key_check
import os


# def keys_to_output(keys):
#     # [A, W, D, S]
#     output=[0,0,0,0]

#     if 'A' in keys:
#         output[0]=1
#     elif 'W' in keys:
#         output[1]=1
#     elif 'D' in keys:
#         output[2]=1
#     elif 'S' in keys:
#         output[3]=1
       
#     return output

def keys_to_output(key):
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    key_to_index = {'A': 0, 'W': 1, 'D': 2, 'S': 3, 'WA': 4, 'WD': 5, 'SA': 6, 'SD': 7, 'NO_KEY': 8}
    if key in key_to_index:
        output[key_to_index[key]] = 1
    return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name), dtype=object)
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():


    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


    paused = False
    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,45,800,600))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120)) #(120, 160) [[159 119 239 ... 132 239 239]...]
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = np.array(keys_to_output(keys), dtype=object) #output=[0,0,0,0]
            training_data.append([screen,output]) #[[array([[159, 119, 239, ..., 239, 222, 239]...],array([0, 0, 0, 1])
            print('Frame took {} seconds'.format(time.time()-last_time))
            
            if len(training_data) % 50 == 0: #need to be changed once the dataset gets bigger
                print(len(training_data))
                np.save(file_name,training_data)

        if 'T' in keys:
            if paused:
                paused=False
                print('unpaused!')
                time.sleep(0.1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main()