from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense,Flatten,Conv2D,Activation,Dropout
from keras_preprocessing.image import ImageDataGenerator
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda, Dense, Flatten
from sklearn.metrics import classification_report
from tqdm import tqdm_notebook as tqdm
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense
from tensorflow.keras import layers
import tensorflow as tf
from keras import backend as K
import keras
from itertools import chain
from glob import glob
import glob
import pandas as pd
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import os

import os
from glob import glob
import pandas as pd

data = pd.read_csv('csv file.csv')

all_image_paths = {os.path.basename(x): x for x in
                   glob(os.path.join('paper-work-2/archive/CLAHE/*.png'))}
data['Path'] = data['Image Index'].map(all_image_paths.get)

all_image_paths2 = {os.path.basename(x): x for x in
                   glob(os.path.join('paper-work-2/archive/DWT/*.png'))}
data['Path2'] = data['Image Index'].map(all_image_paths2.get)

all_image_paths3 = {os.path.basename(x): x for x in
                   glob(os.path.join('paper-work-2/archive/GC/*.png'))}
data['Path3'] = data['Image Index'].map(all_image_paths3.get)

all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels]

for label in all_labels:
    if len(label)>1:
        data[label] = data['Finding Labels'].map(lambda finding: 1 if label in finding else 0)

data = data.groupby('Finding Labels').filter(lambda x : len(x)>11)

train_and_valid_df, test_df = train_test_split(data,
                                               test_size = 0.20,
                                               random_state = 2023,
                                              )

train_df, valid_df = train_test_split(train_and_valid_df,
                                      test_size=0.20,
                                      random_state=2023,
                                     )

base_generator = ImageDataGenerator(rescale=1./255)

IMG_SIZE = (224, 224)
def flow_from_dataframe(image_generator, dataframe, batch_size):

    df_gen = image_generator.flow_from_dataframe(dataframe,
                                                 x_col='Path',
                                                 y_col=all_labels,
                                                 target_size=IMG_SIZE,
                                                 classes=all_labels,
                                                 color_mode='rgb',
                                                 class_mode='raw',
                                                 shuffle=False,
                                                 batch_size=batch_size)

    return df_gen

total_samples = len(train_df)

train_gen = flow_from_dataframe(image_generator=base_generator,
                                dataframe= train_df,
                                batch_size = 512)

valid_gen = flow_from_dataframe(image_generator=base_generator,
                                dataframe=valid_df,
                                batch_size = 512)

test_gen = flow_from_dataframe(image_generator=base_generator,
                               dataframe=test_df,
                               batch_size = 512)

train_x, train_y = next(train_gen)

valid_x, valid_y = next(valid_gen)

test_x, test_y = next(test_gen)

IMG_SIZE = (224, 224)
def flow_from_dataframe(image_generator1, dataframe, batch_size):

    df_gen1= image_generator1.flow_from_dataframe(dataframe,
                                                 x_col='Path2',
                                                 y_col=all_labels,
                                                 target_size=IMG_SIZE,
                                                 classes=all_labels,
                                                 color_mode='rgb',
                                                 class_mode='raw',
                                                 shuffle=False,
                                                 batch_size=batch_size)

    return df_gen1

train_gen2 = flow_from_dataframe(image_generator1=base_generator,
                                dataframe= train_df,
                                batch_size = len(train_x))

valid_gen2 = flow_from_dataframe(image_generator1=base_generator,
                                dataframe=valid_df,
                                batch_size = len(valid_x))

test_gen2 = flow_from_dataframe(image_generator1=base_generator,
                               dataframe=test_df,
                               batch_size = len(test_x))

train_x2, train_y2 = next(train_gen2)
valid_x2, valid_y2 = next(valid_gen2)
test_x2, test_y2 = next(test_gen2)

def flow_from_dataframe(image_generator2, dataframe, batch_size):

    df_gen2= image_generator2.flow_from_dataframe(dataframe,
                                                 x_col='Path3',
                                                 y_col=all_labels,
                                                 target_size=IMG_SIZE,
                                                 classes=all_labels,
                                                 color_mode='rgb',
                                                 class_mode='raw',
                                                 shuffle=False,
                                                 batch_size=batch_size)

    return df_gen2

train_gen3 = flow_from_dataframe(image_generator2=base_generator,
                                dataframe= train_df,
                                batch_size = len(train_x))

valid_gen3 = flow_from_dataframe(image_generator2=base_generator,
                                dataframe=valid_df,
                                batch_size = len(valid_x))

test_gen3 = flow_from_dataframe(image_generator2=base_generator,
                               dataframe=test_df,
                               batch_size = len(test_x))

train_x3, train_y3 = next(train_gen3)
valid_x3, valid_y3 = next(valid_gen3)
test_x3, test_y3 = next(test_gen3)

csv_logger = tf.keras.callbacks.CSVLogger('/working/train.csv', append=True)

input_shape=(224, 224, 3)
img_input = Input(shape=input_shape, name = 'm1')


a= Conv2D(32, (3,3), activation="relu", padding= 'same', name = 'Input_layer1')(img_input)
a = Model(img_input, outputs=a)
img_input2 = Input(shape=input_shape, name = 'm2')
b= Conv2D(32, (3,3), activation="relu", padding= 'same', name = 'Input_layer2')(img_input2)
b = Model(img_input2, outputs=b)
img_input3 = Input(shape=input_shape, name = 'm3')
c= Conv2D(32, (3,3), activation="relu", padding= 'same', name = 'Input_layer3')(img_input3)
c = Model(img_input3, outputs=c)
combined = keras.layers.concatenate([a.output, b.output, c.output], name = 'Combined_layer')
X2 = Conv2D(filters=96, kernel_size=(3,3), padding='valid', activation='relu', name='Layer4')(combined)
y = tf.keras.layers.MaxPool2D(pool_size=(4,4),strides=(4,4))(X2)
p = Flatten()(y)
p = Dense(500, name='Layer5')(p)
p = Dense(3, name='Layer6')(p)
model = Model(inputs=[a.input,b.input,c.input], outputs=p)

initial_learning_rate=1e-3
optimizer = Adam(lr=initial_learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
epochs=50
fit_history = model.fit([train_x, train_x2, train_x3], train_y,
      steps_per_epoch=100,
      epochs=epochs,
      validation_data=([valid_x, valid_x2, valid_x3], valid_y),
      validation_steps=50
)

feature_extractor = keras.Model(
   inputs=model.inputs,
   outputs=model.get_layer(name="Layer5").output,
)

# Assuming you have 'train_x', 'train_x2', and 'train_x3' as your input data
feature_extractor2 = feature_extractor.predict([train_x, train_x2, train_x3])

# Reshape the features and prepare a DataFrame
features = feature_extractor2.reshape(feature_extractor2.shape[0], -1)
df_features = pd.DataFrame(features)

# Save the features DataFrame to a CSV file
csv_features_path = '/working/500_features.csv'
df_features.to_csv(csv_features_path, index=False)


csv_labels_path = '/working/500_labels.csv'
train_labels = train_df.iloc[:512, :][all_labels].idxmax(axis=1)  # Convert one-hot encoded labels back to single column
train_labels_df = pd.DataFrame({"Finding Labels": train_labels})
train_labels_df.to_csv(csv_labels_path, index=False)

loaded_features = pd.read_csv(csv_features_path)
loaded_labels = pd.read_csv(csv_labels_path)


df_concatenated = pd.concat([loaded_labels, loaded_features], axis=1)


csv_concatenated_path = '/working/500_features_with_labels_concatenated.csv'
df_concatenated.to_csv(csv_concatenated_path, index=False)

