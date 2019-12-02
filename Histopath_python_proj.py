#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('../input/train_labels.csv')
print('Size of df',df.shape)
df.head()


# In[ ]:


input_dir='../input/train/'


# In[ ]:


fig = plt.figure(figsize = (15,10))
index = 1
for i in np.random.randint(low = 0, high = df.shape[0], size = 12):
    file = input_dir + df.iloc[i]['id'] + '.tif'
    img = cv2.imread(file)
    ax = fig.add_subplot(3, 4, index)
    ax.imshow(img, cmap = 'gray')
    index = index + 1
    color = ['green' if df.iloc[i].label == 1 else 'blue'][0]
    ax.set_title(df.iloc[i].label, fontsize = 20, color = color)
plt.tight_layout()
plt.show()


# In[ ]:


# removing this image because it caused a training error previously
df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']

# removing this image because it's black
df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
df.head()


# In[ ]:


print(df.label.value_counts()[0])
print(df.label.value_counts()[1])


# In[ ]:


SAMPLE_SIZE = 80000
# take a random sample of class 0 with size equal to num samples in class 1
df_0 = df[df['label'] == 0].sample(SAMPLE_SIZE, random_state = 0)
# filter out class 1
df_1 = df[df['label'] == 1].sample(SAMPLE_SIZE, random_state = 0)

# concat the dataframes
df_train = pd.concat([df_0, df_1], axis = 0).reset_index(drop = True)
# shuffle
df_train = shuffle(df_train)

df_train['label'].value_counts()


# In[ ]:


# split into train and valdiation sets
y = df_train['label']

df_train, df_val = train_test_split(df_train, test_size = 0.1, random_state = 0, stratify = y)


# In[ ]:


# Create a new directory
base_dir = 'base_dir'
os.mkdir(base_dir)


#Folder Structure

'''
    * base_dir
        |-- train_dir
            |-- notumortissue   #0
            |-- tumortissue   #1
        |-- val_dir
            |-- notumortissue
            |-- tumortissue
'''
# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

# create new folders inside train_dir
no_tumor = os.path.join(train_dir, 'notumortissue')
os.mkdir(no_tumor)
has_tumor = os.path.join(train_dir, 'tumortissue')
os.mkdir(has_tumor)


# create new folders inside val_dir
no_tumor = os.path.join(val_dir, 'notumortissue')
os.mkdir(no_tumor)
has_tumor = os.path.join(val_dir, 'tumortissue')
os.mkdir(has_tumor)


print(os.listdir('base_dir/train_dir'))
print(os.listdir('base_dir/val_dir'))


# In[ ]:


# Set the id as the index in df_data
df.set_index('id', inplace=True)

# Get a list of train and val images
train_list = list(df_train['id'])
val_list = list(df_val['id'])



# Transfer the train images

for image in train_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    file_name = image + '.tif'
    # get the label for a certain image
    target = df.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'notumortissue'
    elif target == 1:
        label = 'tumortissue'
    
    # source path to image
    src = os.path.join('../input/train', file_name)
    # destination path to image
    dest = os.path.join(train_dir, label, file_name)
    # copy the image from the source to the destination
    shutil.copyfile(src, dest)


# Transfer the val images

for image in val_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    file_name = image + '.tif'
    # get the label for a certain image
    target = df.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'notumortissue'
    elif target == 1:
        label = 'tumortissue'
    

    # source path to image
    src = os.path.join('../input/train', file_name)
    # destination path to image
    dest = os.path.join(val_dir, label, file_name)
    # copy the image from the source to the destination
    shutil.copyfile(src, dest)


# In[ ]:


print(len(os.listdir('base_dir/train_dir/notumortissue')))
print(len(os.listdir('base_dir/train_dir/tumortissue')))


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
IMAGE_SIZE = 96
train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'
test_path = '../input/test'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32 #10
val_batch_size = 32 #10


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)


datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


# In[ ]:


#Import Keras
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.core import Activation
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2


# In[ ]:


kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.5
dropout_dense = 0.5

model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
                                                                    ))
model.add(Conv2D(first_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_conv))
model.add(Dense(2, activation = "softmax"))

# Compile the model
model.compile(Adam(0.001), loss = "binary_crossentropy", metrics=["accuracy"])


# In[ ]:


from keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[ ]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1, min_lr=0.001)
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=2,
                   callbacks=[earlystopper, reducel])


# In[ ]:


predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)


# In[ ]:


df_preds = pd.DataFrame(predictions, columns=['no_tumor', 'has_tumor'])
df_preds.head()


# In[ ]:


# Get the true labels
y_true = test_gen.classes

# Get the predicted labels as probabilities
y_pred = df_preds['has_tumor']


# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve, auc
print('ROC AUC Score = ',roc_auc_score(y_true, y_pred))


# In[ ]:


fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)


# In[ ]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[ ]:


shutil.rmtree('base_dir')


# In[ ]:


#Folder Structure

'''
    * test_dir
        |-- test_images
'''

# We will be feeding test images from a folder into predict_generator().

# create test_dir
test_dir = 'test_dir'
os.mkdir(test_dir)
    
# create test_images inside test_dir
test_images = os.path.join(test_dir, 'test_images')
os.mkdir(test_images)

# check that the directory we created exists
os.listdir('test_dir')


# In[ ]:


# Transfer the test images into image_dir
test_list = os.listdir('../input/test')

for image in test_list:    
    fname = image
    # source path to image
    src = os.path.join('../input/test', fname)
    # destination path to image
    dst = os.path.join(test_images, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)
print('Total Test Images = ',len(os.listdir('test_dir/test_images')))


# In[ ]:


test_path ='test_dir'
test_gen = datagen.flow_from_directory(test_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


# In[ ]:


num_test_images = 57458 #len(os.listdir('test_dir/test_images')

predictions = model.predict_generator(test_gen, steps=num_test_images, verbose=1)


# In[ ]:


if predictions.shape[0] == num_test_images:
    print('All Predictions Done!')
else:
    print('Error!')


# In[ ]:


# Put the predictions into a dataframe
df_preds = pd.DataFrame(predictions, columns=['no_tumor', 'has_tumor'])
df_preds.head()


# In[ ]:


# This outputs the file names in the sequence in which the generator processed the test images.
test_filenames = test_gen.filenames

# add the filenames to the dataframe
df_preds['file_names'] = test_filenames

# Create an id column
# A file name now has this format: 
# images/00006537328c33e284c973d7b39d340809f7271b.tif

# This function will extract the id:
# 00006537328c33e284c973d7b39d340809f7271b
def extract_id(x):
    
    # split into a list
    a = x.split('/')
    # split into a list
    b = a[1].split('.')
    extracted_id = b[0]
    
    return extracted_id

df_preds['id'] = df_preds['file_names'].apply(extract_id)

df_preds.head()


# In[ ]:


# Get the predicted labels.
# We were asked to predict a probability that the image has tumor tissue
y_pred = df_preds['has_tumor']

# get the id column
image_id = df_preds['id']


# In[ ]:


submission = pd.DataFrame({'id':image_id, 
                           'label':y_pred, 
                          }).set_index('id')

submission.to_csv('submission.csv', columns=['label'])


# In[ ]:


shutil.rmtree('test_dir')

