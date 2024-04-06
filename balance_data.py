import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

# train_data=np.load('training_data.npy')
# #[[array([[159, 119, 239, ..., 239, 222, 239]...],array([0, 0, 0, 1])

# for data in train_data:
#     img=data[0]
#     choice=data[1]
#     cv2.imshow('test',img)
#     print(choice)

#     if cv2.waitKey(25) & 0xFF==ord('q'): #miliseconds, 5000=5seconds
#         cv2.destroyAllWindows()
#         break

train_data = np.load('training_data.npy', allow_pickle=True)#[[array([[159, 119, 239, ..., 239, 222, 239]...],array([0, 0, 0, 1])


df = pd.DataFrame(train_data) #0  [[155, 81, 135, 136, 136, 135, 135, 136, 136, ...  [0, 0, 0, 1]
# print(df.head())
# print(Counter(df[1].apply(str))) #Counter({'[0 0 0 1]': 437, '[0 1 0 0]': 269, '[1 0 0 0]': 144, '[0 0 1 0]': 50})

lefts=[]
forwards=[]
rights=[]
slow=[]

# print(df[1][:5])



# for index, row in df.iterrows():
#     img=row[0]
#     choice=row[1]
    
#     if choice==np.array(left_key).all():
#         lefts.append([img, choice])
#     print(lefts)


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
        print('no match')

lefts=lefts[:len(forwards)]
forwards=forwards[:len(lefts)][:len(rights)]
rights=rights[:len(forwards)]
slow=slow[:len(forwards)]

final_data=lefts+forwards+rights+slow
shuffle(final_data)
np.save('training_data_v2.npy', final_data)

print(len(final_data))
print(final_data)
# print(len(rights))
# print(len(forwards))
# print(len(slow))
