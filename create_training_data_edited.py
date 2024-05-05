import numpy as np
import cv2
import time
import pyautogui
from PIL import ImageGrab

from grabscreen import grab_screen
from getkeys import key_check
import os

WIDTH=480
HEIGHT=270

def keys_to_output(*args):
    output = [0, 0, 0, 0]
    key_to_index = {'A': 0, 'W': 1, 'D': 2, 'S' : 3}
    
    # Flatten args if a single list is passed
    if len(args) == 1 and isinstance(args[0], list):
        keys = args[0]
    else:
        keys = args

    for key in keys:
        if key in key_to_index:
            output[key_to_index[key]] = 1

    return output




starting_value=1
while True:
    file_name='city_training_data/training_data-{}-{}-{}.npy'.format(starting_value,WIDTH, HEIGHT)
    if os.path.isfile(file_name):
        print('File exists')
        # print(file_name)
        starting_value+=1
    else:
        print('File does not exist')
        break

# def get_processed_screen():
#     screen = grab_screen(region=(0, 45, 800, 600))
#     screen = cv2.resize(screen, (WIDTH, HEIGHT))
#     screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
#     return screen

def main(file_name, starting_value):
    file_name=file_name
    starting_value=starting_value
    training_data = []
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)



    last_time = time.time()

    paused = False
    while(True):

        if not paused:
            # 800x600 windowed mode
            # screen = grab_screen(region=(0,45,800,600))
            screen = grab_screen(region=(0,45,1920,1120))

            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH,HEIGHT))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            # get_processed_screen()
           
            # resize to something a bit more acceptable for a CNN

            keys = key_check()
            # output = np.array(keys_to_output(keys), dtype=object) #output=[0,0,0,0,0,0,0,0,0]
            output = np.array(keys_to_output(keys))

            # training_data.append([get_processed_screen(),output])
            training_data.append([screen,output]) #[[array([[159, 119, 239, ..., 239, 222, 239]...],array([0, 0, 0, 1, 0, 0, 0, 0, 0])
            # print('Frame took {} seconds'.format(time.time()-last_time))
            
            if len(training_data) % 100 == 0: #need to be changed once the dataset gets bigger
                print(len(training_data))

                if len(training_data)==200: #need to be changed
                    np.save(file_name,training_data)
                    print('saved', starting_value)
                    training_data=[]
                    starting_value+=1
                    file_name='city_training_data/training_data-{}-{}-{}.npy'.format(starting_value,WIDTH, HEIGHT)
                    
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused=False
                print('unpaused!') 
                time.sleep(0.1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

        # if 'Y' in keys:
        #     if paused:
        #         paused=False
        #         time.sleep(0.1)
        #         cv2.destroyAllWindows()
        #         break  


main(file_name, starting_value)
