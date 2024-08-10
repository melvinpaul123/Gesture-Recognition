import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque 
#from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

class EuclideanDistTracker:
    def __init__(self, min_dist=30, max_disp=50):
        # Store the center positions of the objects
        self.min_dist = min_dist
        self.max_disp = max_disp
        self.center_points = {}
        self.disappeared = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0
            
    def centroid(self, rect):
        x1, y1, x2, y2 = rect
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return np.array([cx, cy])

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []
        # Get center point of new object
        for rect in objects_rect:
            area = (rect[1]-rect[3])*(rect[0]-rect[2])
            center = self.centroid(rect)
            # Find out if that object was detected already
            same_object_detected = False
            obs = np.array(list(self.center_points.values()))
            ids = np.array(list(self.center_points.keys()))
            if len(obs)>0:
                dists = np.linalg.norm(center-obs, axis=1)
                closest_indx = np.argmin(dists)
                if dists[closest_indx] < self.min_dist:
                    id = ids[closest_indx]
                    self.center_points[id] = center
                    #print(self.center_points)
                    objects_bbs_ids.append([rect, id,area])
                    same_object_detected = True
                    self.disappeared[id] = 0

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = center
                objects_bbs_ids.append([rect, self.id_count,area])
                self.disappeared[self.id_count] = 0
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            re, object_id, ar = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        
        disap = [x for x in list(self.center_points.keys()) if x not in list(new_center_points.keys())]
        for i in disap:
            self.disappeared[i] += 1
            if self.disappeared[i]<self.max_disp:
                new_center_points[i] = self.center_points[i]
        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

class PoseDetector():
    def __init__(self, model_path, seq_len=10):
        if not model_path=="":
            classifiername, classifier_extension = os.path.splitext(model_path)
            if classifier_extension==".tflite":
                self.interpreter = tf.lite.Interpreter(model_path=model_path,num_threads=1)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.lite=True
            else:
                self.model = load_model(model_path)
                self.lite=False
        
        self.seq_len = seq_len
        self.gestures = {0: "Nodding", 1: "Stop sign", 2: "Thumbs down", 3: "Waving", 4: "Pointing",
                         5: "Calling someone", 6: "Thumbs up", 7: "Wave someone away", 8: "Shaking head", 9: "Others"}
        self.pTime = 0
        self.d = deque(maxlen=60)
        self.fps=0
        self.keys = deque(maxlen=self.seq_len)
        self.tipIds = [4, 8, 12, 16, 20]
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face = mp.solutions.face_mesh
        self.face =  self.mp_face.FaceMesh(max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5,refine_landmarks=False)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def fingersUp(self,results2):
        fingers = []
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image.flags.writeable = False
        #results2 = self.hands.process(image)
        if results2.multi_hand_landmarks:
          for hand_landmarks, handedness in zip(results2.multi_hand_landmarks,results2.multi_handedness):
              side = str(handedness.classification[0].label[0:]).lower()
              lmList = np.array([[res.x, res.y] for res in hand_landmarks.landmark])
              # Thumb
              if side=='left':
                  if lmList[self.tipIds[0]][0] > lmList[self.tipIds[0] - 1][0]:
                      fingers.append(1)
                  else:
                      fingers.append(0)
              else:
                  if lmList[self.tipIds[0]][0] < lmList[self.tipIds[0] - 1][0]:
                      fingers.append(1)
                  else:
                      fingers.append(0)
              # Fingers
              for id in range(1, 5):
                  if lmList[self.tipIds[id]][1] < lmList[self.tipIds[id] - 2][1]:
                      fingers.append(1)
                  else:
                      fingers.append(0)
        totalFingers = fingers.count(1)
        return totalFingers

    def update_keys(self, results1, results2):
        face, lh, rh = self.data2array(results1, results2)
        center= []
        not_zeros_face = face.flatten().any()
        if not_zeros_face:
            center = face[19]
        if center!=[]:
            if face.flatten().any():
                face = face-center
            if lh.flatten().any():
                lh = lh-center
            if rh.flatten().any():
                rh = rh-center
            key = np.concatenate([lh.flatten(),face.flatten(),rh.flatten()])
            max_value = np.amax(np.abs(key))
            key = key/max_value
            key = np.array(key, dtype=np.float32)
            self.keys.append(key)
        else:
            self.keys.append(np.zeros(1020, dtype=np.float32))
    
    def classify(self):
        X = np.expand_dims(self.keys, axis=0)
        X = X.reshape((X.shape[0], self.seq_len, 1, X.shape[2]))
        if self.lite:
            X = np.array(X, dtype=np.float32)
            input_details_tensor_index = self.input_details[0]['index']
            self.interpreter.set_tensor(input_details_tensor_index,X)
            self.interpreter.invoke()
            output_details_tensor_index = self.output_details[0]['index']
            pred1 = self.interpreter.get_tensor(output_details_tensor_index)
        else:
            pred1 = self.model.predict(X,verbose=0)
        prob = np.max(pred1)
        pred = np.argmax(pred1,axis=1)
        self.keys = deque(maxlen=self.seq_len)
        return self.gestures[pred[0]], prob
    
    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results1 = self.face.process(image)
        results2 = self.hands.process(image)
        return results1, results2
    
    def draw(self, image, pose):
        #pose1 = pose[0:15]#np.delete(pose, (0,21), axis = 0)
        for i in pose:
            cv2.circle(image, (int(i[0]),int(i[1])), 2, (255,255,0), -1)
        return image    
        
    def draw_landmarks(self, image, results1, results2):
        # Draw face connections
        if results1.multi_face_landmarks:
          for face_landmarks in results1.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(image=image,landmark_list=face_landmarks,
                                               connections=self.mp_face.FACEMESH_CONTOURS,landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
        # Draw hand connections
        if results2.multi_hand_landmarks:
            for hand_landmarks in results2.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image,hand_landmarks,self.mp_hands.HAND_CONNECTIONS,
                                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                        self.mp_drawing_styles.get_default_hand_connections_style())
        return image
        
    def get_fps(self):
        self.cTime = time.time()
        self.fps = 1 / (self.cTime - self.pTime)
        self.d.append(self.fps)
        self.fps = sum(self.d)/len(self.d)
        self.pTime = self.cTime
        return self.fps
    
    def data2array(self, results1, results2):
        face = np.zeros((468, 2))
        if results1.multi_face_landmarks:
          for face_landmarks in results1.multi_face_landmarks:
              #face = np.fromiter(([res.x, res.y] for res in face_landmarks.landmark), dtype=np.dtype((int, 2)))
              face = np.array([[res.x, res.y] for res in face_landmarks.landmark])
              '''face = face[:468]
              face = face-face[0]
              scaler = MinMaxScaler()
              face = scaler.fit_transform(face)'''
        lh= np.zeros((21, 2))
        rh= np.zeros((21, 2))
        if results2.multi_hand_landmarks:
          for hand_landmarks, handedness in zip(results2.multi_hand_landmarks,results2.multi_handedness):
              side = str(handedness.classification[0].label[0:]).lower()
              hand = np.array([[res.x, res.y] for res in hand_landmarks.landmark])#.flatten()
              '''hand = hand-hand[0]
              scaler = MinMaxScaler()
              hand = scaler.fit_transform(hand)'''
              if side=='left':
                  lh = hand
              else:
                  rh = hand
        return face, lh, rh
    
    def face_detect(self, image, draw=True):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_detection.process(image)
        h, w = image.shape[:2]
        face_boxes = []
        if results.detections:
          for detection in results.detections:
              bboxC = detection.location_data.relative_bounding_box
              xmin, ymin, hw, hh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
              xmax, ymax = xmin+hw, ymin+hh
              face_boxes.append([xmin, ymin, xmax, ymax])
        return face_boxes

class MultiPerDetector():
    def __init__(self,rp, num_person, model_path="", seq_len=10):
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        #print(dir_path+"/"+per_path)
        self.seq_len = seq_len
        self.num_person = num_person
        self.tracker = EuclideanDistTracker()
        for i in range(self.num_person):
            globals()["per"+str(i)] = PoseDetector(model_path=model_path, seq_len=seq_len)
        self.rp = rp
        
    def det_per(self,frame):
        bboxs = self.rp.detect_person(frame)
        objects_bbs_ids = self.tracker.update(bboxs)
        #print(bboxs,objects_bbs_ids)
        return objects_bbs_ids
    
    def multi_per_gesture(self,image,sort="id",viz=False):
        gestures = []
        objects_bbs_ids = self.det_per(image)
        #print(objects_bbs_ids)
        if sort =="id":
            sorted_objects_bbs_ids = sorted(objects_bbs_ids,key=lambda x: x[1])
        elif sort=="area":
            sorted_objects_bbs_ids = sorted(objects_bbs_ids,key=lambda x: x[2])[::-1]
        for i, bbox in enumerate(sorted_objects_bbs_ids):
            if i<self.num_person:
                person = image[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]]
                results1, results2 = globals()["per"+str(i)].mediapipe_detection(person)
                globals()["per"+str(i)].update_keys(results1, results2)
                if len(globals()["per"+str(i)].keys)==self.seq_len: #i%10==0 and 
                    gest, prob = globals()["per"+str(i)].classify()
                    finger_count = globals()["per"+str(i)].fingersUp(results2)
                    gestures.append([gest, prob, bbox[1], bbox[0], str(finger_count)]) #gest, prob, id, bbox, finger_count
                if viz:
                    per = globals()["per"+str(i)].draw_landmarks(person, results1, results2)
                    image[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]] = per
            if viz:
                cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), (255,255,255), thickness=2)
                cv2.putText(image, str(bbox[1]), (bbox[0][0], int(bbox[0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return image, gestures
    
    def multi_per_poses(self,image,sort="id",viz=False):
        poses = []
        objects_bbs_ids = self.det_per(image)
        if sort =="id":
            sorted_objects_bbs_ids = sorted(objects_bbs_ids,key=lambda x: x[1])
        elif sort=="area":
            sorted_objects_bbs_ids = sorted(objects_bbs_ids,key=lambda x: x[2])
        for i, bbox in enumerate(sorted_objects_bbs_ids):
            if i<self.num_person:
                person = image[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]]
                results1, results2 = globals()["per"+str(i)].mediapipe_detection(person)
                poses.append(results1, results2, bbox[1], bbox[0])
            if viz:
                cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), (255,255,255), thickness=2)
                cv2.putText(image, str(bbox[1]), (bbox[0][0], int(bbox[0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                per = globals()["per"+str(i)].draw_landmarks(person, results1, results2)
                image[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]] = per
        return image, poses