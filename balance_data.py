import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

starting_value=0
WIDTH=480
HEIGHT=270

#get the training data
#[[array([[159, 119, 239, ..., 239, 222, 239]...],array([0, 0, 0, 1])
starting_value=1
for i in list(range(1,16))[::-1]:

    file_name='training_data-{}-{}-{}.npy'.format(i,WIDTH, HEIGHT)
    train_data=np.load(file_name, allow_pickle=True)
    if os.path.isfile(file_name):
        # print('File exists')
        # print(file_name)
        i+=1
    else:
        print('File does not exist')
        break

#Originally the training data output has shape (,9) because I put the 9 keys but I want to get only 3 keys from the first element
modified_train_data = [[item[0], item[1][:3]] for item in train_data]
# print(modified_train_data)


df = pd.DataFrame(modified_train_data) #0  [[155, 81, 135, 136, 136, 135, 135, 136, 136, ...  [0, 0, 1]
# print('counter:{}'.format(Counter(df[1].apply(str)))) 
#Counter({'[0 1 0]': 165, '[1 1 0]': 18, '[0 1 1]': 17})

# define the keys
lefts=[]
rights=[]
forwards=[]
straight_left=[]
straight_right=[]
# # slow=[]

left_key=np.array([1,0,0])
right_key=np.array([0,0,1])
forward_key=np.array([0,1,0])
straight_left_key=np.array([1,1,0])
straight_right_key=np.array([0,1,1])
# slow_key=np.array([0,0,0,1])

#mapping the keys if it match from the choice of train_data

for data in modified_train_data:
    img = np.array(data[0])  # Assuming data[0] is the image
    choice = np.array(data[1])  # Assuming data[1] is the choice array

    # Append data based on the choice
    if np.array_equal(choice, forward_key):
        forwards.append([img, choice])
    elif np.array_equal(choice, straight_left_key):
        straight_left.append([img, choice])
    elif np.array_equal(choice, straight_right_key):
        straight_right.append([img, choice])


#data augment-straight_left, straight_right

# Initialize the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to perform augmentation
def augment_images(image_list, augment_times=3):
    augmented_images = []
    for img, choice in image_list:
        img = img.reshape((1,) + img.shape)  # Reshape for generator (adding batch dimension)
        i = 0
        for x in datagen.flow(img, batch_size=1):
            augmented_images.append([x[0], choice])  # Append the augmented image and choice
            i += 1
            if i >= augment_times:
                break  # Stop after generating the required number of images
    return augmented_images

# Augment straight_left and straight_right lists
augmented_straight_left = augment_images(straight_left, augment_times=3)
augmented_straight_right = augment_images(straight_right, augment_times=3)

# Check the number of images augmented
# print("Augmented straight_left count:", len(augmented_straight_left)) #54
# print("Augmented straight_right count:", len(augmented_straight_right)) #51

#create the final dataset
final_data=forwards+augmented_straight_left+augmented_straight_right
shuffle(final_data)


np.save('training_data/training_data_{}_{}.npy'.format(WIDTH, HEIGHT), final_data)