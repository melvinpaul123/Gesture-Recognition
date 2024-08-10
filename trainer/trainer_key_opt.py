import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
#tf.config.optimizer.set_jit(True)
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Conv1D, BatchNormalization, Flatten, TimeDistributed, Activation, MaxPooling1D, LSTM, Dense, Dropout
#from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import LambdaCallback
from keras_tuner.tuners import Hyperband
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
#import sys
'''inputs = tf.keras.Input(shape=(30, 1, 1054))
conv_2d_layer = tf.keras.layers.Conv1D(64, 3, padding='same')
outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
print(outputs.shape)'''


folder = "datas/"
gestures = {0: "Nodding", 1: "Stop sign", 2: "Thumbs down", 3: "Waving", 4: "Pointing",
            5: "Calling someone", 6: "Thumbs up", 7: "Wave someone away", 8: "Shaking head",9: "Others"}
sequence_len = 10
model_name = 'models/action10_9_np3'
project_name = 'seq10_9_np3'

def rolling_window2D(a,n,step=3):
    # a: 2D Input array 
    # n: Group/sliding window length
    return a[np.arange(a.shape[0]-n+1)[:,None] + np.arange(n)][::step, :]

#use cov 2d

def load_csv(sequence_len = 30):
    X=[]
    Y=[]
    folder_list = os.listdir(folder)
    for folder_i in folder_list:
        for i in range(len(gestures)):
            dir_list = os.listdir(folder+folder_i+"/keys/"+str(i))
            for d in dir_list:
                try:
                    df = np.array(pd.read_csv(folder+folder_i+"/keys/"+str(i)+"/"+d, sep=',', header=None))
                    x = rolling_window2D(df,sequence_len)
                    X.extend(list(x))
                    y = np.ones(len(x),dtype=int)*i
                    Y.extend(list(y))
                except: pass # Exception as e
    return np.array(X),np.array(Y)

X,Y = load_csv(sequence_len)
print(X.shape, Y.shape)
#print(X.shape[1:3])
#X = X.reshape((X.shape[0], 1,  sequence_len, int(X.shape[2]/2),2))
X = X.reshape((X.shape[0], sequence_len, 1, X.shape[2]))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
y_train = to_categorical(y_train).astype(int)
y_val = to_categorical(y_val).astype(int)
print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)
print(X_test.shape,y_test.shape)
del X, Y
#sys.exit()

class StopOnPoint(tf.keras.callbacks.Callback):
    def __init__(self, point):
        super(StopOnPoint, self).__init__()
        self.point = point

    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["val_categorical_accuracy"]
        if accuracy >= self.point:
            self.model.stop_training = True
            
class checkpoint_custom:
    def __init__(self):
        self.best_val_acc = 0
        self.best_val_loss = 1000
        
    def saveModel(self, epoch, logs):
        val_acc = logs['val_categorical_accuracy']
        val_loss = logs['val_loss']
    
        if val_acc > self.best_val_acc:
            print(f'\nModel Saved val_categorical_accuracy: {self.best_val_acc:.4f} ---> {val_acc:.4f}, val_loss: {self.best_val_loss:.4f}')
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            model.save(model_name+".h5")
        elif val_acc == self.best_val_acc:
            if val_loss < self.best_val_loss:
                print(f'\nModel Saved val_categorical_accuracy: {self.best_val_acc:.4f}, val_loss: {self.best_val_loss:.4f} ---> {val_loss:.4f}')
                self.best_val_loss = val_loss
                model.save(model_name+".h5")
        else:
            print(f'\nval_binary_accuracy or val_loss did not improve from val_binary_accuracy: {self.best_val_acc:.4f}, val_loss: {self.best_val_loss:.4f}')

savep = checkpoint_custom()

#cp_callback = ModelCheckpoint(model_name+".h5", monitor='val_categorical_accuracy', mode='max', verbose=1, save_weights_only=False, save_best_only=True)
# Callback for early stopping
#es_callback1 = EarlyStopping(monitor='val_categorical_accuracy',mode='max',verbose=1, patience=4)
#es_callback = EarlyStopping(monitor='val_categorical_accuracy',mode='max',verbose=1, patience=15)

def model_builder1(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('lstm1',min_value=32,max_value=256,step=32), return_sequences=True, activation='relu', input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(LSTM(hp.Int('lstm2',min_value=32,max_value=256,step=32), return_sequences=True, activation='relu'))
    model.add(LSTM(hp.Int('lstm3',min_value=32,max_value=256,step=32), return_sequences=False, activation='relu'))
    model.add(Dense(hp.Int('dense1',min_value=32,max_value=256,step=32), activation='relu'))
    model.add(Dense(hp.Int('dense2',min_value=32,max_value=256,step=32), activation='relu'))
    model.add(Dense(hp.Int('dense3',min_value=32,max_value=256,step=32), activation='relu'))
    #model.add(Dropout(hp.Float('dropout_1',0,0.5,step=0.05,default=0.25)))
    model.add(Dropout(0.25))
    model.add(Dense(len(gestures), activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
#X_train.shape[1]

def model_builder2(hp):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(filters=hp.Int('conv1',min_value=32,max_value=256,step=32), kernel_size=(3,3), padding='same'), input_shape=(X_train.shape[1] , X_train.shape[2], X_train.shape[3],X_train.shape[4])))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(filters=hp.Int('conv2',min_value=32,max_value=256,step=32), kernel_size=(3,3), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(hp.Float('dropout_1',0,0.5,step=0.05,default=0.25))))
    model.add(TimeDistributed(MaxPooling2D(pool_size=2, padding='same')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(hp.Int('lstm1',min_value=32,max_value=256,step=32), activation='relu')) #tanh
    model.add(Dropout(hp.Float('dropout_2',0,0.5,step=0.05,default=0.25)))
    model.add(Dense(hp.Int('dense1',min_value=32,max_value=256,step=32), activation='relu'))
    #model.add(Dense(hp.Int('dense2',min_value=32,max_value=256,step=32), activation='relu'))
    model.add(Dense(len(gestures), activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def model_builder(hp):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=hp.Int('conv1',min_value=32,max_value=256,step=32), kernel_size=3, padding='same'), input_shape=(X_train.shape[1] ,X_train.shape[2],X_train.shape[3])))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(filters=hp.Int('conv2',min_value=32,max_value=256,step=32), kernel_size=3, padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(hp.Float('dropout_1',0,0.5,step=0.05,default=0.25))))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, padding='same')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(hp.Int('lstm1',min_value=32,max_value=256,step=32), activation='relu')) #tanh
    model.add(Dropout(hp.Float('dropout_2',0,0.5,step=0.05,default=0.25)))
    model.add(Dense(hp.Int('dense1',min_value=32,max_value=256,step=32), activation='relu'))
    #model.add(Dense(hp.Int('dense2',min_value=32,max_value=256,step=32), activation='relu'))
    model.add(Dense(len(gestures), activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
batch_size = 64
tuner = Hyperband(model_builder, objective = 'val_categorical_accuracy', max_epochs=10, factor = 3, directory = 'my_dir', project_name = project_name)   
tuner.search(X_train, y_train, epochs=5, batch_size=batch_size, validation_data = (X_val, y_val))#, callbacks=[es_callback1])
model = tuner.get_best_models(1)[0]
model.summary()

plot_model(model, to_file="evaluation/"+project_name+".png", show_shapes=True, show_layer_names=True,
           rankdir="TB", expand_nested=True, dpi=300)
epochs = 50
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, callbacks=[LambdaCallback(on_epoch_end=savep.saveModel)])#cp_callback, es_callback, StopOnPoint(0.98)])
model.summary()

model = load_model(model_name+".h5")

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('evaluation/'+project_name+'_accuracy.png', dpi=300)
plt.show()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('evaluation/'+project_name+'_loss.png', dpi=300)
plt.show()

model.save(model_name+".h5")

model = load_model(model_name+".h5")

model.save(model_name+'.hdf5', include_optimizer=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_quantized_model = converter.convert()

open(model_name+'.tflite', 'wb').write(tflite_quantized_model)

pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
acc = accuracy_score(y_test,pred)

# Display the results
print(f'## {acc*100:.2f}% accuracy on the test set')

# Map the numbers into letters
y_test_letters = [gestures[x] for x in y_test]
pred_letters = [gestures[x] for x in pred]

print(classification_report(y_test_letters, pred_letters))

# Display a confusion matrix
cf_matrix = confusion_matrix(y_test_letters, pred_letters, normalize='true')
plt.figure(figsize = (20,15))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test_letters)), yticklabels = sorted(set(y_test_letters)),cbar=False)
plt.title('Normalized Confusion Matrix\n', fontsize = 23)
plt.xlabel("Predicted Classes",fontsize=15)
plt.ylabel("True Classes",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15,rotation=0)
plt.savefig('evaluation/'+project_name+'_Confusion_Matrix.png', dpi=300)
plt.show()