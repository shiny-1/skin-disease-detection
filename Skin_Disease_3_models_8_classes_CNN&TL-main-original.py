#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In this notebook, I've used **CNN** to perform Image Classification on the skin diseases dataset.<br>
# Since this dataset is small, if we train a neural network to it, it won't really give us a good result.<br>
# Therefore, I'm going to use the concept of **Transfer Learning** and also custom CNN to train the model to compare accurate results.

# # Importing Libraries

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output
from warnings import filterwarnings
warnings.filterwarnings('ignore')
dir =r"D:\4-2\dataset (skin diseases)2"
for dirname, _, filenames in os.walk(dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Data Preperation

# In[4]:


labels = ["actinic keratosis", 
"dermatofibroma", 
"melanoma", 
"seborrheic keratosis", 
"squamous cell carcinoma",
"Acne_and_rosacea",
"Eczema",
"Tinea_Ringworm",
"healthy"
]


# In[5]:


get_ipython().system('pwd')


# We start off by appending all the images from the  directories into a Python list and then converting them into numpy arrays after resizing it.

# In[6]:


X_train = []
y_train = []
image_size = 150
for i in labels:
    folderPath = os.path.join(dir,'Train',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)
        
for i in labels:
    folderPath = os.path.join(dir,'Test',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)
        


# In[7]:


k=0
fig, ax = plt.subplots(1,9,figsize=(20,20))
fig.text(s='Sample Image From Each Class ',size=18,fontweight='bold',
             fontname='monospace',y=0.62,x=0.4,alpha=0.8)
for i in labels:
    j=0
    while True :
        if y_train[j]==i:
            ax[k].imshow(X_train[j])
            ax[k].set_title(y_train[j])
            ax[k].axis('off')
            k+=1
            break
        j+=1


# In[10]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
X_train = np.array(X_train)
y_train = np.array(y_train)
# Define the data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # Rotate the image by a random angle between -20 and 20 degrees
    width_shift_range=0.1,  # Shift the image horizontally by a random fraction of the total width
    height_shift_range=0.1,  # Shift the image vertically by a random fraction of the total height
    shear_range=0.2,  # Apply shear transformation with a random shear angle between -20 and 20 degrees
    zoom_range=0.2,  # Apply random zoom between 80% and 120% of the original image
    horizontal_flip=True,  # Flip the image horizontally
    vertical_flip=True  # Flip the image vertically
)

# Fit the data augmentation generator to the training data
datagen.fit(X_train)
# Generate augmented images and labels
augmented_images, augmented_labels = next(datagen.flow(X_train, y_train, batch_size=len(X_train)))
print(len(X_train))
print(augmented_images.shape,augmented_labels.shape,y_train.shape)
print(y_train[:10],augmented_labels[:10])
print(len(augmented_images),len(augmented_labels))


# In[11]:


X_train_new = np.concatenate((X_train,augmented_images), axis=0)
y_train_new = np.concatenate((y_train,augmented_labels), axis=0)
print(len(X_train_new),len(y_train_new))
X_train, y_train = shuffle(X_train_new,y_train_new, random_state=0)


# In[12]:


X_train.shape


# Dividing the dataset into **Training** and **Testing** sets.

# In[13]:


X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)


# Performing **One Hot Encoding** on the labels after converting it into numerical values:

# In[14]:


y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)


y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)


# In[26]:


train_labels = []
test_labels = []

img_size= 300

for i in os.listdir(dir+'/Train/'):
    for j in os.listdir(dir+"/Train/"+i):
       train_labels.append(i)
        
for i in os.listdir(dir+'/Test/'):
    for j in os.listdir(dir+"/Test/"+i):
        test_labels.append(i)

plt.figure(figsize = (17,9));
lis = ['Train', 'Test']
for i,j in enumerate([train_labels, test_labels]):
    plt.subplot(1,2, i+1);
    sns.countplot(x = j);
    plt.xlabel(lis[i])
    plt.xticks(rotation=45)


# ---

# # Custom CNN

# In[12]:


from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.models import load_model

# Start training freshly
tf.keras.backend.clear_session()

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(image_size,image_size,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(9))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()


# In[13]:


import h5py
with h5py.File(r"D:\4-2\custom-cnn.h5",'w') as f:
    pass
tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("custom-cnn.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)


# In[14]:


history = model.fit(X_train,y_train,validation_split=0.1, epochs =30, verbose=1, batch_size=16,
                   callbacks=[tensorboard,checkpoint,reduce_lr])


# # Visualize Training

# In[15]:


#Visualize Training
def plot_graphs(history, string):
    sns.set_style("whitegrid")
    plt.plot(history.history[string])
    plt.plot(history.history["val_"+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title("Skin Disease custom-cnn-Model Epochs")
    plt.legend([string,"val_"+string])
    plt.show()
plot_graphs(history,'accuracy')
plot_graphs(history,'loss')


# In[16]:


pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)
print(classification_report(y_test_new,pred))


# In[17]:


from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class_accuracy = {}
    
for true, predi in zip(y_test_new, pred):
    if true not in class_accuracy:
        class_accuracy[true] = {'correct': 0, 'total': 0}
        
    class_accuracy[true]['total'] += 1
    if true == predi:
        class_accuracy[true]['correct'] += 1
    
for cls, acc in class_accuracy.items():
    class_accuracy[cls] = (acc['correct'] / acc['total'])*100


# In[18]:


def plot_class_wise_accuracy(class_accuracy):
    class_labels = list(class_accuracy.keys())
    accuracies = list(class_accuracy.values())

    plt.figure(figsize=(8, 4))
    plt.bar(labels, accuracies, color='skyblue')
    plt.xlabel('classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-wise Accuracy')
    plt.ylim(0, 100)  # Limiting y-axis from 0% to 100% for accuracy values
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

plot_class_wise_accuracy(class_accuracy)


# In[19]:


fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True)
fig.text(s='Heatmap of the custom-cnn model Confusion Matrix',size=12,y=0.92,x=0.28,alpha=0.8)

plt.show()


# In[20]:


model.save(r"D:\4-2\Flask folder\custom-cnn.h5")


# **Callbacks** -> Callbacks can help you fix bugs more quickly, and can help you build better models. They can help you visualize how your model’s training is going, and can even help prevent overfitting by implementing early stopping or customizing the learning rate on each iteration.<br><br>
# By definition, "A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training."
# 
# In this notebook, I'll be using **TensorBoard, ModelCheckpoint and ReduceLROnPlateau** callback functions

# # Transfer Learning _ EfficientNet B0

# Deep convolutional neural network models may take days or even weeks to train on very large datasets.
# 
# A way to short-cut this process is to re-use the model weights from pre-trained models that were developed for standard computer vision benchmark datasets, such as the ImageNet image recognition tasks. Top performing models can be downloaded and used directly, or integrated into a new model for your own computer vision problems.
# 
# In this notebook, I'll be using the **EfficientNetB0** model which will use the weights from the **ImageNet** dataset.
# 
# The include_top parameter is set to *False* so that the network doesn't include the top layer/ output layer from the pre-built model which allows us to add our own output layer depending upon our use case!

# In[21]:


effnet = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))


# **GlobalAveragePooling2D** -> This layer acts similar to the Max Pooling layer in CNNs, the only difference being is that it uses the Average values instead of the Max value while *pooling*. This really helps in decreasing the computational load on the machine while training.
# <br><br>
# **Dropout** -> This layer omits some of the neurons at each step from the layer making the neurons more independent from the neibouring neurons. It helps in avoiding overfitting. Neurons to be ommitted are selected at random. The **rate** parameter is the liklihood of a neuron activation being set to 0, thus dropping out the neuron
# 
# **Dense** -> This is the output layer which classifies the image into 1 of the 4 possible classes. It uses the **softmax** function which is a generalization of the sigmoid function.

# In[22]:


tf.keras.backend.clear_session()


# In[23]:


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# In[24]:


effnet = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))


# In[25]:


x = effnet.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(9, activation='softmax')(x)
model = Model(inputs=effnet.input, outputs=predictions)


# In[26]:


model.summary()


# We finally compile our model.

# In[27]:


model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])


# In[28]:


import h5py
with h5py.File(r"D:\4-2\skin-effnet.h5",'w') as f:
    pass
tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("skin-effnet.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)


# In[29]:


#history = model.fit(X_train,y_train,validation_split=0.1, epochs =30, verbose=1, batch_size=16,
#                   callbacks=[tensorboard,checkpoint,reduce_lr])
history = model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs =28, verbose=1, batch_size=16,
                   callbacks=[tensorboard,checkpoint,reduce_lr])


# In[31]:


#Visualize Training
def plot_graphs(history, string):
    sns.set_style("whitegrid")
    plt.plot(history.history[string])
    plt.plot(history.history["val_"+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title("Skin Disease EffnetB0-Model Epochs")
    plt.legend([string,"val_"+string])
    plt.show()
plot_graphs(history,'accuracy')
plot_graphs(history,'loss')


# In[34]:


pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)
print(classification_report(y_test_new,pred))


# In[35]:


from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class_accuracy = {}
    
for true, predi in zip(y_test_new, pred):
    if true not in class_accuracy:
        class_accuracy[true] = {'correct': 0, 'total': 0}
        
    class_accuracy[true]['total'] += 1
    if true == predi:
        class_accuracy[true]['correct'] += 1
    
for cls, acc in class_accuracy.items():
    class_accuracy[cls] = (acc['correct'] / acc['total'])*100



# In[36]:


def plot_class_wise_accuracy(class_accuracy):
    class_labels = list(class_accuracy.keys())
    accuracies = list(class_accuracy.values())

    plt.figure(figsize=(8, 4))
    plt.bar(labels, accuracies, color='skyblue')
    plt.xlabel('classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-wise Accuracy')
    plt.ylim(0, 100)  # Limiting y-axis from 0% to 100% for accuracy values
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

plot_class_wise_accuracy(class_accuracy)


# In[37]:


fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True)
fig.text(s='Heatmap of the EffnetB0-model Confusion Matrix',size=12,y=0.92,x=0.28,alpha=0.8)

plt.show()


# In[33]:


model.save(r"D:\4-2\Flask folder\skin-effnet.h5")


# ### Table for Training Dataset

# In[38]:


pred = model.predict(X_train)
pred = np.argmax(pred,axis=1)
y_train_new = np.argmax(y_train,axis=1)

matrix = confusion_matrix(y_train_new,pred)
matrix.diagonal()/matrix.sum(axis=1)


# In[ ]:


import math
print(91, 89, 91, 76, 87)
print(math.ceil(0.98319328*91), math.ceil(0.96551724*89), math.ceil(0.99145299*91), math.ceil(1.0*76), math.ceil(1.0*87))
print(math.ceil(0.98319328*100), math.ceil(0.96551724*100), math.ceil(0.99145299*100), math.ceil(1.0*100), math.ceil(1.0*100))


# ### Table for Testing Dataset

# In[39]:


pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)

matrix = confusion_matrix(y_test_new,pred)
matrix.diagonal()/matrix.sum(axis=1)


# # Transfer Learning _ Inception V3

# In[40]:


import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
import pandas as pd
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3

from keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam, Nadam


# In[41]:


from keras.applications.inception_v3 import InceptionV3

tf.keras.backend.clear_session()

# We use model inceptionV3
pre_train_model = InceptionV3(
      include_top = False,
      weights = "imagenet",
      input_shape = (image_size,image_size, 3)      
)


# In[42]:


pre_train_model.summary()


# In[43]:


# just use a part of model because our task is simple so if use whole model will be overfitting

for layer in pre_train_model.layers:
    layer.trainable = False
last_layer = pre_train_model.get_layer('mixed9')  # cut begin to layer block8_9_mixed
last_output = pre_train_model.output


# In[44]:


# Add some custom layer to do our task, output will be 1 node
# x = MaxPooling2D(pool_size=(2,2))(last_output)
x = Flatten()(last_output)
# x = Dense(2048, activation='relu')(x)
# x = Dense(1024, activation='relu')(x)
# x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
# x = Dense(512, activation='relu')(x)
# x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(9, activation='softmax')(x)


# In[58]:


# Define optimizer, learning rate and loss function
model = Model(pre_train_model.input, output)
model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])


# In[59]:


import h5py
with h5py.File(r"D:\4-2\Inception-v3.h5",'w') as f:
    pass
tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("Inception-v3.h5",monitor="val_acc",save_best_only=True,mode="auto",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)


# In[60]:


#history = model.fit(X_train,y_train,validation_split=0.1, epochs =28, verbose=1, batch_size=16,
#                   callbacks=[tensorboard,checkpoint,reduce_lr])
history = model.fit(X_train, y_train, validation_split=0.1, epochs=28, verbose=1, batch_size=16,
                    callbacks=[tensorboard, checkpoint, reduce_lr],
                    validation_data=(X_test, y_test))


# In[56]:


#Visualize Training
def plot_graphs(history, string):
    sns.set_style("whitegrid")
    plt.plot(history.history[string])
    plt.plot(history.history["val_"+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title("Skin Disease Inception-v3-Model Epochs")
    plt.legend([string,"val_"+string])
    plt.show()
plot_graphs(history,'acc')
plot_graphs(history,'loss')


# In[57]:


pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)
print(classification_report(y_test_new,pred))


# In[61]:


from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class_accuracy = {}
    
for true, predi in zip(y_test_new, pred):
    if true not in class_accuracy:
        class_accuracy[true] = {'correct': 0, 'total': 0}
        
    class_accuracy[true]['total'] += 1
    if true == predi:
        class_accuracy[true]['correct'] += 1
    
for cls, acc in class_accuracy.items():
    class_accuracy[cls] = (acc['correct'] / acc['total'])*100



# In[62]:


def plot_class_wise_accuracy(class_accuracy):
    class_labels = list(class_accuracy.keys())
    accuracies = list(class_accuracy.values())

    plt.figure(figsize=(8, 4))
    plt.bar(labels, accuracies, color='skyblue')
    plt.xlabel('classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-wise Accuracy')
    plt.ylim(0, 100)  # Limiting y-axis from 0% to 100% for accuracy values
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

plot_class_wise_accuracy(class_accuracy)


# In[63]:


fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True)
fig.text(s='Heatmap of the Confusion Matrix',size=12,y=0.92,x=0.28,alpha=0.8)

plt.show()


# # VGG16 model

# In[16]:


from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

# Load the VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(9, activation='softmax')(x)  # Replace 8 with the number of classes in your dataset
model = Model(inputs=base_model.input, outputs=predictions)


# In[17]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[19]:


import h5py
with h5py.File(r"D:\4-2\VGG16-model.h5",'w') as f:
    pass
tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("VGG16-model.keras",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)


# In[20]:


# Fit the model to the training data
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=28, verbose=1, batch_size=16,
                    callbacks=[tensorboard, checkpoint, reduce_lr])


# In[22]:


#Visualize Training
def plot_graphs(history, string):
    sns.set_style("whitegrid")
    plt.plot(history.history[string])
    plt.plot(history.history["val_"+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title("Skin Disease VGG16-model Epochs")
    plt.legend([string,"val_"+string])
    plt.show()
plot_graphs(history,'accuracy')
plot_graphs(history,'loss')


# In[23]:


pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)
print(classification_report(y_test_new,pred))


# In[24]:


from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class_accuracy = {}
    
for true, predi in zip(y_test_new, pred):
    if true not in class_accuracy:
        class_accuracy[true] = {'correct': 0, 'total': 0}
        
    class_accuracy[true]['total'] += 1
    if true == predi:
        class_accuracy[true]['correct'] += 1
    
for cls, acc in class_accuracy.items():
    class_accuracy[cls] = (acc['correct'] / acc['total'])*100



# In[25]:


fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True)
fig.text(s='Heatmap of the VGG16-model Confusion Matrix',size=12,y=0.92,x=0.28,alpha=0.8)

plt.show()


# In[ ]:




