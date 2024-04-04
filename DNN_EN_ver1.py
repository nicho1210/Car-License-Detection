import numpy as np 
import pandas as pd 
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2 as cv


test = pd.read_csv('emnist-mnist-test.csv', dtype = np.float32)
train = pd.read_csv('emnist-mnist-train.csv', dtype = np.float32)

print(train.shape)
print(test.shape)

test.columns=train.columns

y1 = np.array(train.iloc[:,0].values)
x1 = np.array(train.iloc[:,1:].values)
y2 = np.array(test.iloc[:,0].values)
x2 = np.array(test.iloc[:,1:].values)
print(y1.shape)
print(x1.shape)

fig,axes = plt.subplots(3,5,figsize=(10,8))
for i,ax in enumerate(axes.flat):
    ax.imshow(x1[i].reshape([28,28]))
    ax.axis('off')
    ax.set_title(y1[i])
plt.show()

# Normalise and reshape data
train_dnn = x1 / 255.0
test_dnn = x2 / 255.0

train_dnn_number = train_dnn.shape[0]
train_dnn_height = 28
train_dnn_width = 28
train_dnn_size = train_dnn_height*train_dnn_width

train_dnn = train_dnn.reshape(train_dnn_number, train_dnn_height, train_dnn_width, 1)

test_dnn_number = test_dnn.shape[0]
test_dnn_height = 28
test_dnn_width = 28
test_dnn_size = test_dnn_height*test_dnn_width

test_dnn = test_dnn.reshape(test_dnn_number, test_dnn_height, test_dnn_width, 1)

# Transform labels
number_of_classes = 47

y1_dnn = tf.keras.utils.to_categorical(y1, number_of_classes)
y2_dnn = tf.keras.utils.to_categorical(y2, number_of_classes)

train_x,test_x,train_y,test_y = train_test_split(train_dnn,y1_dnn,test_size=0.2,random_state = 16)

# Define the model architecture
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(47, activation='softmax'))  # 9 classes: word_0 to word_8

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

MCP = ModelCheckpoint('Best_points.keras',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=3,mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=0.0001)

start = datetime.now()
history = model.fit(train_x,train_y,epochs=10,validation_data=(test_x,test_y),callbacks=[MCP,ES,RLP])
end = datetime.now()
time_dnn = end - start

q = len(history.history['accuracy'])

plt.figsize=(10,10)
sns.lineplot(x = range(1,1+q),y = history.history['accuracy'], label='Accuracy')
sns.lineplot(x = range(1,1+q),y = history.history['val_accuracy'], label='Val_Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Val_Accuracy')
plt.show()

model.save('car_license_model_en.h5')