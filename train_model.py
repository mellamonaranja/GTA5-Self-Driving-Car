import numpy as np
from models import alexnet, inception_v3

WIDTH=160
HEIGHT=120
lr=1e-3
EPOCHS=10
OUPUT=4

# MODEL_NAME='pygta5-car-{}-{}-{}-epochs.model'.format(lr, 'alexnetv2', EPOCHS)
MODEL_NAME='pygta5-car-{}-{}-{}-epochs.model'.format(lr, 'inception_v3', EPOCHS)
# model=alexnet(WIDTH, HEIGHT, lr, output=OUPUT)
model=inception_v3(WIDTH, HEIGHT, lr, output=OUPUT)

train_data = np.load('training_data_v4.npy',allow_pickle=True)


# train = train_data[:-50] #need to be changed
# test = train_data[-50:]

np.random.shuffle(train_data)
train, test=train_data[:-80,:], train_data[-80:,:]


X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1) #(2332, 80, 60, 1)
# print(X[0])
# Y = np.array([i[1] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]
# print(Y[0])

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

try:

    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    
except IndexError as err:

    print("IndexError: ", err)

model.save(MODEL_NAME)

#tensorboard --logdir=C:/Users/motoko/Desktop/WORKSPACE/GTA5AI/log