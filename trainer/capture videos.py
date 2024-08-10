import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from tqdm import tqdm
import cv2
import pose_module as pm
folder = "data/gesture/"
gestures = {0: "Nodding", 1: "Stop sign", 2: "Thumbs down", 3: "Waving", 4: "Pointing",
            5: "Calling someone", 6: "Thumbs up", 7: "Wave someone away", 8: "Shaking head",
            9: "Others"}

for i in range(len(gestures)):
    if not os.path.exists(folder+"/"+str(i)):
        os.makedirs(folder+"/"+str(i))

cap = cv2.VideoCapture(0)#+ cv2.CAP_DSHOW)
if (cap.isOpened() == False): 
    print("Error reading video file")
#cap.set(3, 1280)
#cap.set(4, 720)
capture=False
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
pos = pm.PoseDetector(model_path="")
frame_no = 300
vid_name = 0
est=True
print("Initializing")
while 1:
    class_int = input("enter class no.: ")
    if int(class_int)>len(gestures)-1:
        print("wrong class")
        break
    pathn = folder+str(class_int)+"/"+str(class_int)+"_"+str(vid_name)+'.avi'
    while os.path.exists(pathn):
        vid_name +=1
        pathn = folder+str(class_int)+"/"+str(class_int)+"_"+str(vid_name)+'.avi'
    f=0
    p_bar = tqdm(range(frame_no),position=0, leave=True)
    result = cv2.VideoWriter(pathn, cv2.VideoWriter_fourcc(*'MJPG'),10, size)
    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        image= frame.copy()
        #if i%2==0:
        if capture:
            result.write(frame)
            f=f+1
            p_bar.update(1)
            #p_bar.refresh()
        if est:
            results1, results2, results3 = pos.mediapipe_detection(frame)
            #pose, lh, rh = pos.data2array(image, results1, results2)
            #print(pose)
            #print(lh)
            #print(rh)
            image = pos.draw_landmarks(image, results1, results2, results3)
        fps = pos.get_fps()
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('OpenCV Feed', image)
        i=i+1
        if f==frame_no:
            result.release()
            cv2.destroyAllWindows()
            capture=False
            est = True
            print("-> Capture Stoped")
            p_bar.close()
            break
        # Break gracefully
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            print("-> Saving Video Stream")
            result.release()
            cv2.destroyAllWindows()
            capture=False
            est = True
            print("-> Capture Stoped")
            p_bar.close()
            break  
        elif k == ord('p'): #pause stream
            print("-> Pausing Video Stream")
            print("-> Press any key to continue Video Stream")
            cv2.waitKey(-1) #wait until any key is pressed
        elif k == ord('c'): #pause stream
            if capture==False:
                capture=True
                print("-> Started to capture")
            else:
                capture=False
                print("-> Capture Stoped")
        elif k == ord('q'): #pause stream
            if est==False:
                est=True
                print("-> Started pose est.")
            else:
                est=False
                print("-> Stoped pose est.")
cap.release()
cv2.destroyAllWindows()
'''cap = cv2.VideoCapture("video.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )'''