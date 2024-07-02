import keras
import os
import pickle
import numpy as np
from tf_models import inception_v3

validation_dir="training_data"

USE_WEIGHTS=True

DECAY=0.9
#        w0     s1     a2    d3    wa4     wd5   sa6     sd7     nk8
WEIGHTS=[1.0,  1.0,   1.0,  1.0,   1.0,    1.0,  1.0,    1.0,    1.0]
#WEIGHTS =[0.030903154382632643, 1000.0, 0.020275559590445278, 0.013302794647291147, 0.0225283995449392, 0.025031555049932444, 1000.0, 1000.0, 0.016423203268260675]

mapping_dict = {0: "W",
                1: "S",
                2: "A",
                3: "D",
                4: "WA",
                5: "WD",
                6: "SA",
                7: "SD",
                8: "NK",}

#if the model predicts 'A' bur it was meant to be a 'WA'
#then that's better than if the model predicts other keys
close_dict = {
              0: {4: 0.3, 5: 0.3, 8: 0.05},  # Should be W, but said WA or WD NK nbd
              1: {6: 0.3, 7: 0.3, 8: 0.05},  # Should be S, but said SA OR SD, NK nbd
              2: {4: 0.3, 6: 0.3},           # Should be A, but SA or WA
              3: {5: 0.3, 7: 0.3},           # Shoudl be D, but SD or WD
              4: {2: 0.5},                   # Should be WA, but A
              5: {3: 0.5},                   # Should be WD, but D
              6: {1: 0.3, 2: 0.3},           # Should be SA, but S or A
              7: {1: 0.3, 3: 0.3},           # Should be SD, but S or D
              8: {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05, 7: 0.05, },  # should be NK... but whatever.
              }

WIDTH = 480
HEIGHT = 270
EPOCHS = 30
lr=1e-3
OUPUT=3
MODEL_NAME = ''
frame_count=3
FILE_I_END = 16 ### +1

model=inception_v3(WIDTH, HEIGHT, frame_count, lr, output=OUPUT)
MODEL_NAME='model_sentnet_color-1350'
model.load(MODEL_NAME)

while True:
    dist_dict={0:0,
               1:0,
               2:0,
               3:0,
               4:0,
               5:0,
               6:0,
               7:0,
               8:0
               }
    
    total=0
    correct=0
    closeness=0

    for f in os.listdir(validation_dir):
        if ".pickle" in f:
            chunk=pickle.load(open(os.path.jois(validation_dir, f),'rb'), allow_pickle=True)
            
            for data in chunk:
                total+=1
                X=data[1]
                X=X/255.0
                y=data[0]

                prediction=model.predict([X.reshape(-1, X.shape[0], X.shape[1], X.shape[2])])[0]

                if USE_WEIGHTS:
                    prediction=np.array(prediction)*np.array(WEIGHTS)

                dist_dict[np.argmax(prediction)]+=1

                if np.argmax(prediction)==np.argmax(y):
                    correct+=1
                    closeness+=1
                else:
                    if np.argmax(prediction) in close_dict[np.argmax(y)]:
                        closeness+=close_dict[np.argmax(y)][np.argmax(prediction)]
            
            print(30*'_')
            print("weights:", WEIGHTS)
            print(f"Accuracy: {round(correct/total,3)}. Accuracy considering 'closeness':{round(closeness/total,3)}")
            print(dist_dict)

            
            largest_key=max(dist_dict, key=dist_dict.get)
            
            with open('log.txt', 'a') as f:
                f.write('Weights: '+str(WEIGHTS))
                f.write('\n')
                f.write(f"Real mobile2-32-batch-0001.hdf5 accuracy: {round(correct/total, 3)}. Accuracy considering 'closeness': {round(closeness/total, 3)}\n")
                f.write("Distribution: "+str(dist_dict))
                f.write("\n")
                f.write("\n")
                f.write("\n")

            WEIGHTS[largest_key]*=DECAY