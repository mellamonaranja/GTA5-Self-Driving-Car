import numpy as np
import cv2
import time
import pyautogui
from PIL import ImageGrab

from grabscreen import grab_screen
from getkeys import key_check
import os

# def keys_to_output(keys):
#     # [A, W, D, S]
#     output=[0,0,0,0]

#     if 'A' in keys:
#         output[0]=1
#     elif 'W' in keys:
#         output[1]=1
#     elif 'D' in keys:
#         output[2]=1
#     else:
#         output[3]=1
    
#     return output

# file_name='training_data.npy'

# if os.path.isfile(file_name):
#     print('File exists, loading previous data')
#     training_data=list(np.load(file_name))
# else:
#     print('File does not exist, starting fresh')
#     training_data=[]
        


# def roi(img, vertices):
    
#     #blank mask:
#     mask = np.zeros_like(img)   
    
#     #filling pixels inside the polygon defined by "vertices" with the fill color    
#     cv2.fillPoly(mask, vertices, 255)
    
#     #only the ROI is left, only show the area that is the mask
#     #returning the image only where mask pixels are nonzero
#     masked = cv2.bitwise_and(img, mask)
#     return masked

# def process_img(image):
#     original_image = image
#     # convert to gray
#     processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # edge detection
#     processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    
#     # apply gauissian blur after edge detection
#     # because edge detection algorithum drawing the line again
#     processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
    
#     #returns the array of arrays that contain the lines 
#     vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
#                          ], np.int32)

#     processed_img = roi(processed_img, [vertices])

#     # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
#     #                                     rho   theta   thresh  min length, max gap:        
#     lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180,      20,       15)
    
#     m1=0
#     m2=0
#     try:
#         l1, l2, m1,m2 = draw_lanes(original_image,lines)
        
#         #find the two final most common slopes
#         cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
#         cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
#     except Exception as e:
#         print(str(e))
#         pass

#     try:
#         for coords in lines:
#             coords = coords[0]
#             try:
#                 cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                
                
#             except Exception as e:
#                 print(str(e))
#     except Exception as e:
#         pass

#     return processed_img,original_image, m1, m2

# def straight():
#     ReleaseKey(A)
#     ReleaseKey(D)


# def left():

#     ReleaseKey(D)
#     PressKey(A) 


# def right():

#     ReleaseKey(A)
#     PressKey(D) 

# def slow_accel():
#     ReleaseKey(W)  


# def brake():
#     ReleaseKey(W) 
#     PressKey(S) 

# def accel():
#     ReleaseKey(S)
#     PressKey(W)


# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)

# def main():
#     for i in list(range(4))[::-1]:
#         print(i+1)
#         time.sleep(1)

#     last_time = time.time()

#     while True: 
#         screen =  grab_screen(region=(0,45,800,650))
#         screen=cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#         screen=cv2.resize(screen,(80,60))
#         keys=key_check()
#         output=keys_to_output(keys)
#         training_data.append([screen, output])
#         print('Frame took {} seconds'.format(time.time()-last_time))
#         last_time = time.time()



#         # new_screen,original_image, m1, m2 = process_img(screen)
#         # # cv2.imshow('window', new_screen)
#         # cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
#         # if m1<0 and m2<0:
#         #     right()
#         # elif m1>0 and m2>0:
#         #     left()
#         # else:
#         #     accel() 


#         #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break

#         if len(training_data)%500==0:
#             print(len(training_data))
#             np.save(file_name, training_data)

# def main():

#     for i in list(range(4))[::-1]:
#         print(i+1)
#         time.sleep(1)
        
#     while(True):
#         # 800x600 windowed mode
#         screen = grab_screen(region=(0,40,800,640))
#         last_time = time.time()
#         screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#         # resize to something a bit more acceptable for a CNN
#         screen = cv2.resize(screen, (80,60))
#         keys = key_check()
#         output = keys_to_output(keys)
#         training_data.append([screen,output])
        
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break

#         if len(training_data) % 500 == 0:
#             print(len(training_data))
#             np.save(file_name,training_data)

########################

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [A,W,D] boolean values.
    '''
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name), dtype=object)
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


    paused = False
    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,640))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = np.array(keys_to_output(keys), dtype=object)
            training_data.append([screen,output])
            print('Frame took {} seconds'.format(time.time()-last_time))
            
            if len(training_data) % 50 == 0:
                print(len(training_data))
                np.save(file_name,training_data)
print(np.__version__)


main()