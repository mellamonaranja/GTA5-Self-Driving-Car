import numpy as np
import cv2
import time
import pyautogui

from grabscreen import grab_screen
from getkeys import key_check
import os
from models import alexnet, inception_v3
from directkeys import ReleaseKey, PressKey, W, A, S, D

WIDTH=160
HEIGHT=120
lr=1e-3
EPOCHS=10
OUTPUT=4
# MODEL_NAME='pygta5-car-{}-{}-{}-epochs.model'.format(lr, 'alexnetv2', EPOCHS)
MODEL_NAME='pygta5-car-{}-{}-{}-epochs.model'.format(lr, 'inception_v3', EPOCHS)

t_time=1

def straight():
    PressKey(W) 
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(0.4)
    ReleaseKey(W)

def left():
    PressKey(W) 
    PressKey(A) 
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(1)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D) 
    ReleaseKey(A)
    ReleaseKey(S)
    time.sleep(1)
    ReleaseKey(D)

def slow_accel():
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(W)

def brake():
    ReleaseKey(W) 
    ReleaseKey(A)
    ReleaseKey(D)

    PressKey(S) 
    time.sleep(0.2) 
    ReleaseKey(S) 

# def accel():
#     ReleaseKey(S)
#     PressKey(W)

# model=alexnet(WIDTH, HEIGHT, lr, output=OUTPUT)
model=inception_v3(WIDTH, HEIGHT, lr, output=OUTPUT)

model.load(MODEL_NAME)


def main():


    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(t_time)

    
    paused = False
    start_time = time.time()
    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,45,800,600))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH, HEIGHT)) #(120, 160) [[159 119 239 ... 132 239 239]...]
            # resize to something a bit more acceptable for a CNN
           
            # print('Frame took {} seconds'.format(time.time()-last_time))
    
            #predict(x, batch_size=None, verbose='auto', steps=None, callbacks=None)
            prediction=model.predict([screen.reshape(WIDTH, HEIGHT,1)])[0] #predict the list of features
            #you will get a list of predictions back
    
            moves=list(np.around(prediction))
            print(moves, prediction)
            # print('predicion : {}'.format(predicion))
            # print('moves : {}'.format(moves))
            # turn_thresh=.60
            turn_right=.80
            turn_left=.80
            fwd_thresh=.85
            max_pred=np.argmax(prediction)
            actions = {
                0: left,
                1: straight,
                2: right,
                3: slow_accel
            }

            # if predicion[0]>turn_left:
            #     left()
            # elif predicion[1]>fwd_thresh:
            #     straight()
            # elif predicion[2]>turn_right:
            #     right()
            # elif predicion[3]>fwd_thresh:
            #     brake()
            # elif predicion[0]<turn_left and predicion[1]<fwd_thresh and predicion[2]<turn_right and predicion[3]<fwd_thresh:
            #      slow_accel()
            # else:
            #     actions[max_pred]()
            
            # else:
            #     slow_accel()


            if all(value<=0.4 for value in prediction):
                brake()

            else:

                actions[max_pred]()

        keys=key_check()

                                    
        if 'T' in keys:
            if paused:
                paused=False
                time.sleep(0.1)
                cv2.destroyAllWindows()
                break        
            else:                            
                paused=True
                ReleaseKey(A)
                ReleaseKey(D)
                ReleaseKey(W)
                ReleaseKey(S)
                time.sleep(t_time)    

            # if 'Q' in keys:
            #     if paused==False:
            #         cv2.destroyAllWindows()
            #         break  
            
        # if time.time()-start_time>10:
        #     paused=False
        #     time.sleep(0.1) 
        #     cv2.destroyAllWindows()
        #     break
        
main()