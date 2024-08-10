import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


folder = "datas/"
gestures = {1: "Stop sign", 2: "Thumbs down", 3: "Waving", 4: "Pointing",
            5: "Calling someone", 6: "Thumbs up", 7: "Wave someone away",9: "Others"}#, 0: "Nodding",  8: "Shaking head"}
gestures_len = 8
sequence_len = 10
size = (64,64)

def rolling_window2D(a,n,step=3):
    # a: 2D Input array 
    # n: Group/sliding window length
    return a[np.arange(a.shape[0]-n+1)[:,None] + np.arange(n)][::step, :]

def load_csv(Xd, Yd, sequence_len = 30):
    X=[]
    Y=[]
    for x1, y1 in zip(Xd, Yd):
        try:
            frames = []
            cap = cv2.VideoCapture(x1)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                  cap.release()
                  cv2.destroyAllWindows()
                  break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, size) 
                frame = frame/255
                frames.append(frame)
            frames = np.array(frames)
            x = rolling_window2D(frames,sequence_len)
            X.extend(list(x))
            y = np.ones(len(x),dtype=int)*y1
            #y = np.ones(len(x),dtype=int)*i
            Y.extend(list(y))
        except Exception as e: 
            print(e)
            pass # Exception as e
    return np.array(X),np.array(Y)

Xd=[]
Yd=[]
folder_list = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
for folder_i in folder_list:
    for i, k in enumerate(list(gestures.keys())):
        dir_list = os.listdir(folder+folder_i+"/gesture/"+str(k))
        for d in dir_list:
            if d!="desktop.ini":
                Xd.append(folder+folder_i+"/gesture/"+str(k)+"/"+d)   
                Yd.append(i)
X_train, X_test, y_train, y_test = train_test_split(Xd, Yd, test_size=0.2, random_state=42, stratify=Yd)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

X_train,y_train = load_csv(X_train,y_train,sequence_len)
X_test,y_test = load_csv(X_test,y_test,sequence_len)
X_val,y_val = load_csv(X_val,y_val,sequence_len)

y_train = to_categorical(y_train).astype(int)
y_val = to_categorical(y_val).astype(int)
print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)
print(X_test.shape,y_test.shape)


# Specify the path to your dataset directory
# dataset_path = "/media/zany/Ubuntu Secondary/Studies MAS/Sem3/DLRV/dlrv_dataset/data1/gesture"

# # Define the categories (classes) for your video classifier
# categories = ['1', '2', '5', '7']

# # Load and preprocess the video frames
# def load_frames(video_path):
#     frames = []
#     for frame_file in sorted(os.listdir(video_path)):
#         img = image.load_img(os.path.join(video_path, frame_file), target_size=(64, 64))
#         img = image.img_to_array(img)
#         frames.append(img)
#     return np.array(frames)

# # Load the dataset and preprocess the data
# def load_dataset():
#     X = []
#     y = []
#     for category in categories:
#         category_path = os.path.join(dataset_path, category)
#         for video in os.listdir(category_path):
#             video_path = os.path.join(category_path, video)
#             frames = load_frames(video_path)
#             X.append(frames)
#             y.append(categories.index(category))
#     X = np.array(X)
#     y = np.array(y)
#     y = to_categorical(y)
#     return X, y

# # Split the dataset into training and testing sets
# def split_dataset(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)     ``
#     return X_train, X_test, y_train, y_test

# # Build the video classifier model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(gestures), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# # Load and preprocess the dataset
# # X, y = load_dataset()

# # Split the dataset into training and testing sets
# # X_train, X_test, y_train, y_test = split_dataset(X, y)

# # Load frames 


# # Build the video classifier model
model = build_model()

# # Train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=10)

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)
