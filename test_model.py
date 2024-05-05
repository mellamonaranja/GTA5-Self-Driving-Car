import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from tf_models import inception_v3 
from getkeys import key_check
from collections import deque, Counter
import random
from statistics import mode,mean
import numpy as np
from motion import motion_detection


WIDTH = 480
HEIGHT = 270
EPOCHS = 10
lr=1e-3
frame_count=3
OUPUT=3
GAME_WIDTH = 1920
GAME_HEIGHT = 1080

log_len = 25

motion_req = 800
motion_log = deque(maxlen=log_len)



t_time=1

def straight():
    # if random.randrange(0,3) == 1:
    #     PressKey(W)
    #     time.sleep(0.1)
    # else:
    #     ReleaseKey(W)
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    # time.sleep(0.4)
    # ReleaseKey(W)

def left():
    if random.randrange(0,3)==1:
        PressKey(W) 
    else:
        ReleaseKey(W)
    PressKey(A) 
    time.sleep(0.4)
    ReleaseKey(D)
    ReleaseKey(S)
    # ReleaseKey(A)

def right():
    if random.randrange(0,3)==1:
        PressKey(W)
    else:
        ReleaseKey(W)   
    PressKey(D) 
    time.sleep(0.4)

    ReleaseKey(A)
    ReleaseKey(S)
    # ReleaseKey(D)

def straight_left():
    # if random.randrange(0, 3) == 1:
    #     PressKey(W)
    # else:
    #     ReleaseKey(W)
    PressKey(W)
    PressKey(A)
    # time.sleep(0.4)
    ReleaseKey(S)
    ReleaseKey(D)


def straight_right():
    # if random.randrange(0, 3) == 1:
    #     PressKey(W)
    # else:
    #     ReleaseKey(W)
    PressKey(W)
    PressKey(D)
    # time.sleep(0.4)
    ReleaseKey(S)
    ReleaseKey(A)

def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(W)

def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)



def slow_accel():
    # ReleaseKey(W)
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)


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

model=inception_v3(WIDTH, HEIGHT, frame_count, lr, output=OUPUT)
MODEL_NAME='model_sentnet_color-1350'
model.load(MODEL_NAME)

def main():


    for i in list(range(2))[::-1]:
        print(i+1)
        # time.sleep(t_time)

    
    paused = False
    start_time = time.time()

    screen = grab_screen(region=(0,45,GAME_WIDTH,GAME_HEIGHT+40))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen, (WIDTH, HEIGHT)) 

    t_minus=prev
    t_now=prev
    t_plus=prev

    while(True):

        if not paused:

            screen = grab_screen(region=(0,45,GAME_WIDTH,GAME_HEIGHT+40))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))
            # print('Frame took {} seconds'.format(round(time.time()-start_time),3))

            delta_count_last=motion_detection(screen, t_minus, t_now, t_plus)

            t_minus=t_now
            t_now=t_plus
            t_plus=screen
            t_plus=cv2.blur(t_plus,(4,4))
    
            #predict(x, batch_size=None, verbose='auto', steps=None, callbacks=None)
            #you will get a list of predictions back
            prediction=model.predict([screen.reshape(1,WIDTH, HEIGHT,3)][0]) #predict the list of features
            prediction=np.array(prediction)*np.array([0.35, 0.0, 0.1])

            # print(prediction)
            # prediction=np.array(prediction)
            # prediction=np.around(np.array(prediction))
            # moves=list(np.array(prediction))

            max_pred=np.argmax(prediction)
            #forwards+augmented_straight_left+augmented_straight_right

            actions = {
                0: straight,
                1: straight_left,
                2: straight_right
                
            }
            actions[max_pred]()

            print(max_pred)




            # motion_log.append(delta_count_last)
            # motion_avg=round(mean(motion_log),3)
            # print('Motion : {}, Choice : {}'.format(motion_avg, actions[max_pred]()))


        keys=key_check()

        print(keys)                       
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

            if 'Q' in keys:
                if paused==False:
                    cv2.destroyAllWindows()
                    break  
            
        if time.time()-start_time>10:
            paused=False
            time.sleep(0.1) 
            cv2.destroyAllWindows()
            break
        
main()