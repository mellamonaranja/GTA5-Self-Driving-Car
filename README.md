# Self Driving Car and Object detection on GTA V

### 1. Self Driving Car

### Simulate Python keypresses for controlling a game

![keypress-ezgif com-video-to-gif-converter](https://github.com/mellamonaranja/ObjectDetection_and_SelfDriving_GTA5/assets/55770526/73346bc8-8c6a-4f13-9d04-94663a1d6fb7)


### Generate the dataset and labeling

- Driving over the Circuit 
- Record display

<img width="334" alt="image" src="https://github.com/mellamonaranja/ObjectDetection_and_SelfDriving_GTA5/assets/55770526/27e5c2ec-88ec-4da0-bfe9-b31469ab7b15">

### Balancing the dataset

- Distribution of the unbalanced Steering dataset 

##### Before :

<img width="607" alt="image" src="https://github.com/mellamonaranja/ObjectDetection_and_SelfDriving_GTA5/assets/55770526/91308e2f-f0b4-4df2-8684-0b56d8d69821">

##### After :

- Distribution of the balanced Steering dataset 

<img width="594" alt="image" src="https://github.com/mellamonaranja/ObjectDetection_and_SelfDriving_GTA5/assets/55770526/1b076974-96c4-4797-9ec0-42f709797b75">

#####

- The balanced dataset is artificially expended using image augmentation
- Random shifts, random brightness, and random zoom

<img width="602" alt="image" src="https://github.com/mellamonaranja/ObjectDetection_and_SelfDriving_GTA5/assets/55770526/b2ca7a21-28ec-46b3-ab08-32447445a6ab">

### Training neural network

- Training with model Alexnet 

<img width="690" alt="image" src="https://github.com/mellamonaranja/ObjectDetection_and_SelfDriving_GTA5/assets/55770526/eea4df44-ce87-45e3-8193-60708a616f26">

- Training with model inception_v3

<img width="264" alt="image" src="https://github.com/mellamonaranja/ObjectDetection_and_SelfDriving_GTA5/assets/55770526/f2a72f51-e84b-4891-8ac6-cc4d6eecabf8">

### Test model

![Untitled-ezgif com-video-to-gif-converter (2)](https://github.com/mellamonaranja/ObjectDetection_and_SelfDriving_GTA5/assets/55770526/aedd8982-e1d1-4338-8aff-11d4acc9b28e)


### development

1. Download code

```
git clone https://github.com/mellamonaranja/ObjectDetection_and_SelfDriving_GTA5.git
```

2. Install packages

```
pip install -r requirements.txt
```

### USAGE EXAMPLE

```
if __name__ == "__main__":

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

            max_pred=np.argmax(prediction)

            actions = {
                0: straight,
                1: straight_left,
                2: straight_right
                
            }
            actions[max_pred]()

            print(max_pred)

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
```

### 2. Object detection

- Acquiring a Vehicle for the Agent

![2024-05-2309-46-50-Trim-ezgif com-video-to-gif-converter](https://github.com/mellamonaranja/ObjectDetection_and_SelfDriving_GTA5/assets/55770526/b2a8399d-e04c-4edc-9f40-3129130766d2)

### USAGE EXAMPLE

```
#Acquiring a Vehicle for the Agent
      #if the agent does not have a vehicle, we want to be able to steal a vehicle : find the vehicle and steal it
      if len(vehicle_dict) > 0:
        closest = sorted(vehicle_dict.keys())[0]
        vehicle_choice = vehicle_dict[closest]
        print('CHOICE:',vehicle_choice)
        if not stolen:
          
          #approach the car
          determine_movement(mid_x = vehicle_choice[0], mid_y = vehicle_choice[1], width=1280, height=705)
          if closest < 0.1:
            keys.directKey("w", keys.key_release)
            keys.directKey("f")
            time.sleep(0.05)          
            keys.directKey("f", keys.key_release)
            stolen = True
          else:
            keys.directKey("w")
```

### Ref:
https://pythonprogramming.net/tensorflow-object-detection-api-self-driving-car/
