import numpy as np
import random
from tf_models import alexnet, inception_v3, sentnet_color
import cv2
import os
import tflearn
import pandas as pd
from collections import Counter

WIDTH = 480
HEIGHT = 270
EPOCHS = 30
lr=1e-3
OUPUT=3
MODEL_NAME = ''
frame_count=3
FILE_I_END = 16 ### +1

model=inception_v3(WIDTH, HEIGHT, frame_count, lr, output=OUPUT, model_name=MODEL_NAME)

for e in range(EPOCHS):
    data_order=[i for i in range(1,FILE_I_END)] 
    random.shuffle(data_order)
    
    for count, i in enumerate(data_order):
        
        try:
            
            # file_name='training_data/training_data_{}_{}.npy'.format(WIDTH, HEIGHT)
            file_name='training_data/training_data-{}-{}-{}.npy'.format(i,WIDTH, HEIGHT)
            
            if os.path.isfile(file_name):
                train_data=np.load(file_name,allow_pickle=True)# [   [    [FRAMES], CHOICE   ]    ]
                # print('File exists')
                # print('training_data-{}.npy'.format(i), len(train_data))

                #modify train data
                train_data = [[item[0], item[1][:3]] for item in train_data]

            else:
                # print('File does not exist')
                break




            np.random.shuffle(train_data)

            # train, test=train_data[:-80,:], train_data[-80:,:]
            train = train_data[:-50]
            test = train_data[-50:]

            X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3) 
            Y = [i[1] for i in train]
            test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            test_y = [i[1] for i in test]      


            try:

                model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
                        snapshot_step=500, show_metric=True, run_id=MODEL_NAME) #n_epoch?
                
                if count%10==0:
                    print('SAVING MODEL!')
                    model.save(MODEL_NAME)

            except IndexError as err:

                print("IndexError: ", err)

        
        except Exception as e:
            print("ExceptionError: ",str(e))
                    

#tensorboard --logdir=C:/Users/motoko/Desktop/WORKSPACE/GTA5AI/log