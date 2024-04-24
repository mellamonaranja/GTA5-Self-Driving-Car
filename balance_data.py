import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('training_data.npy', allow_pickle=True)
#[[array([[159, 119, 239, ..., 239, 222, 239]...],array([0, 0, 0, 1])


df = pd.DataFrame(train_data) #0  [[155, 81, 135, 136, 136, 135, 135, 136, 136, ...  [0, 0, 0, 1]
print(df.head())
# print(Counter(df[1].apply(str))) #Counter({'[0 0 0 1]': 437, '[0 1 0 0]': 269, '[1 0 0 0]': 144, '[0 0 1 0]': 50})

lefts=[]
forwards=[]
rights=[]
slow=[]

forward_key=np.array([0,1,0,0])
left_key=np.array([1,0,0,0])
right_key=np.array([0,0,1,0])
slow_key=np.array([0,0,0,1])

for data in train_data:
    img = data[0]
    choice = data[1]

   
    if np.array_equal(choice,left_key):
        lefts.append([img, choice])
    elif np.array_equal(choice,forward_key):
        forwards.append([img, choice])
    elif np.array_equal(choice,right_key):
        rights.append([img, choice])
    elif np.array_equal(choice,slow_key):
        slow.append([img, choice])
    else:
        pass

# forwards=forwards[:len(lefts)]
# lefts=lefts[:len(forwards)]
# rights=rights[:len(lefts)]
# slow=slow[:len(lefts)]


# print(lefts)
# print(len(rights))
# print(len(forwards))
# print(len(slow))

desired_size = len(forwards)
right_indices = np.random.choice(len(rights), desired_size, replace=True)
slow_indices = np.random.choice(len(slow), desired_size, replace=True)
left_indices = np.random.choice(len(lefts), desired_size, replace=True)

left_balanced = [lefts[i] for i in left_indices]
right_balanced = [rights[i] for i in right_indices]
slow_balanced = [slow[i] for i in slow_indices]

print(len(left_balanced))
print(len(right_balanced))
print(len(slow_balanced))

# final_data=lefts+forwards+rights+slow
final_data=left_balanced+forwards+right_balanced+slow_balanced
shuffle(final_data)
np.save('training_data_v4.npy', final_data)