import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import cv2
import csv
import numpy as np
import pose_module as pm

folder = "datas/"
rewrite = 1
pos = pm.PoseDetector(model_path="")
gestures = {0: "Nodding", 1: "Stop sign", 2: "Thumbs down", 3: "Waving", 4: "Pointing",
            5: "Calling someone", 6: "Thumbs up", 7: "Wave someone away", 8: "Shaking head",
            9: "Others"}

folder_list = os.listdir(folder)
for folder_i in folder_list:
    for i in range(len(gestures)):
        if not os.path.exists(folder+folder_i+"/keys/"+str(i)):
            os.makedirs(folder+folder_i+"/keys/"+str(i))
    
    for i in range(len(gestures)):
        dir_list = os.listdir(folder+folder_i+"/gesture/"+str(i))
        for v in dir_list:
            try:
                if (not os.path.exists(folder+folder_i+"/keys/"+str(i)+"/"+os.path.splitext(v)[0]+".csv")) or rewrite:
                    with open(folder+folder_i+"/keys/"+str(i)+"/"+os.path.splitext(v)[0]+".csv", 'w', newline="") as csv_file:
                        pass
                    with open(folder+folder_i+"/keys/"+str(i)+"/"+os.path.splitext(v)[0]+"f.csv", 'w', newline="") as csv_file:
                        pass
                    cap = cv2.VideoCapture(folder+folder_i+"/gesture/"+str(i)+"/"+v)
                    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(length)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                          cap.release()
                          cv2.destroyAllWindows()
                          break
                        frame2 = cv2.flip(frame, 1)
                        results1, results2 = pos.mediapipe_detection(frame)
                        face, lh, rh = pos.data2array(results1, results2)
                        '''not_zeros_pose = pose.flatten().any()
                        if not_zeros_pose:
                            center = pose[0]
                        else:'''
                        not_zeros_face = face.flatten().any()
                        if not_zeros_face:
                            center = face[19]
                        else:
                            continue
                        if face.flatten().any():
                            face = face-center
                        if lh.flatten().any():
                            lh = lh-center
                        if rh.flatten().any():
                            rh = rh-center
                        #if pose.flatten().any():
                            #pose = pose-center'''
                        keys = np.concatenate([lh.flatten(),face.flatten(),rh.flatten()]) #pose.flatten()
                        max_value = np.amax(np.abs(keys))
                        keys = keys/max_value
                        keys = np.array(keys, dtype=np.float32)
                        with open(folder+folder_i+"/keys/"+str(i)+"/"+os.path.splitext(v)[0]+".csv", 'a', newline="") as csv_file:
                            writer = csv.writer(csv_file, delimiter=",")
                            writer.writerow(keys)
                        
                        results1, results2 = pos.mediapipe_detection(frame2)
                        face, lh, rh = pos.data2array(results1, results2)
                        '''not_zeros_pose = pose.flatten().any()
                        if not_zeros_pose:
                            center = pose[0]
                        else:'''
                        not_zeros_face = face.flatten().any()
                        if not_zeros_face:
                            center = face[19]
                        else:
                            continue
                        if face.flatten().any():
                            face = face-center
                        if lh.flatten().any():
                            lh = lh-center
                        if rh.flatten().any():
                            rh = rh-center
                        #if pose.flatten().any():
                            #pose = pose-center'''
                        keys = np.concatenate([lh.flatten(),face.flatten(),rh.flatten()])
                        max_value = np.amax(np.abs(keys))
                        keys = keys/max_value
                        keys = np.array(keys, dtype=np.float32)
                        #keys = np.concatenate([lh.flatten(),face.flatten(),pose.flatten(),rh.flatten()])
                        with open(folder+folder_i+"/keys/"+str(i)+"/"+os.path.splitext(v)[0]+"f.csv", 'a', newline="") as csv_file:
                            writer = csv.writer(csv_file, delimiter=",")
                            writer.writerow(keys)
            except Exception as e:
                print(e)
                pass
        if i==9:
            for z in range(2):
                for q in range(300):
                    with open(folder+folder_i+"/keys/"+str(i)+"/zero_"+str(z)+".csv", 'a', newline="") as csv_file:
                        writer = csv.writer(csv_file, delimiter=",")
                        writer.writerow(np.zeros(1020, dtype=np.float32))
                    with open(folder+folder_i+"/keys/"+str(i)+"/zero_"+str(z)+"f.csv", 'a', newline="") as csv_file:
                        writer = csv.writer(csv_file, delimiter=",")
                        writer.writerow(np.zeros(1020, dtype=np.float32))