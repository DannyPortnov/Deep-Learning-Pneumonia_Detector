import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras import layers
from keras import applications
from keras import callbacks
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import cv2
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy.typing import NDArray
from typing import Optional, Literal, Type, TypedDict
import glob


labels = ['PNEUMONIA', 'NORMAL']
# IMG_SIZE = 150
IMG_DEPTH = 3
IMG_SIZE = 224
BATCH = 32
SEED = 42


normal_dataset = glob.glob('/content/drive/MyDrive/deep learning/chest_xray/NORMAL/*.jpeg')
pneumonia_dataset = glob.glob('/content/drive/MyDrive/deep learning/chest_xray/PNEUMONIA/*.jpeg')
virus_dataset = list(filter(lambda x: 'virus' in x, pneumonia_dataset))
bacterial_dataset = list(filter(lambda x: 'bacteria' in x, pneumonia_dataset))

def split_data(dataSet, testSize, valSize):
    train, test= train_test_split(dataSet, test_size=testSize, random_state=42, shuffle=True)
    train, val= train_test_split(train, test_size=valSize, random_state=42, shuffle=True)
    return train, test, val


train_normal, test_normal, val_normal = split_data(normal_dataset, 0.15, 0.038)
train_bacterial, test_bacterial, val_bacterial = split_data(bacterial_dataset, 0.05, 0.0095)
train_virus, test_virus, val_virus = split_data(virus_dataset, 0.019, 0.019)


# train_normal, test_normal= train_test_split(normal_dataset, test_size=0.15, random_state=42, shuffle=True)
# train_bacterial, test_bacterial = train_test_split(bacterial_dataset, test_size=0.05, random_state=42, shuffle=True)
# train_virus, test_virus = train_test_split(virus_dataset, test_size=0.075, random_state=42, shuffle=True)

# # Split the training sets into training and validation sets for each class
# # TODO: check why we need to split the validation set from the training set in advance instead of using validation split
# train_normal, val_normal= train_test_split(train_normal, test_size=0.038, random_state=42, shuffle=True)
# train_bacterial, val_bacterial = train_test_split(train_bacterial, test_size=0.0095, random_state=42, shuffle=True)
# train_virus, val_virus = train_test_split(train_virus, test_size=0.019, random_state=42, shuffle=True)

# Concatenate the three classes
train = [x for x in train_normal]
train.extend([x for x in train_bacterial])
train.extend([x for x in train_virus])

test = [x for x in test_normal]
test.extend([x for x in test_bacterial])
test.extend([x for x in test_virus])

val = [x for x in val_normal]
val.extend([x for x in val_bacterial])
val.extend([x for x in val_virus])

df_train = pd.DataFrame(np.concatenate([['Normal']*(len(train_normal)) , ['Pneumonia']*(len(train_bacterial) + len(train_virus))]), columns = ['class'])
df_train['image'] = [x for x in train]

df_val = pd.DataFrame(np.concatenate([['Normal']*(len(val_normal)) , ['Pneumonia']*(len(val_bacterial) + len(val_virus))]), columns = ['class'])
df_val['image'] = [x for x in val]

df_test = pd.DataFrame(np.concatenate([['Normal']*len(test_normal) , ['Pneumonia']*(len(test_bacterial) + len(test_virus))]), columns = ['class'])
df_test['image'] = [x for x in test]


# With data augmentation to prevent overfitting and handling the imbalance in dataset
# Because the dataset is small we "increase" the dataset by change of images parameters.
# In this way we increase our dataset and prevent overfitting.

train_datagen = ImageDataGenerator(rescale=1/255.,
                                  zoom_range = 0.1,
                                  #rotation_range = 0.1,
                                  width_shift_range = 0.1,
                                  height_shift_range = 0.1)

test_datagen = ImageDataGenerator(rescale=1/255.)

ds_train = train_datagen.flow_from_dataframe(df_train,
                                             #directory=train_path, #dataframe contains the full paths
                                             x_col = 'image',
                                             y_col = 'class',
                                             target_size = (IMG_SIZE, IMG_SIZE),
                                             class_mode = 'binary',
                                             batch_size = BATCH,
                                             seed = SEED)

ds_val = test_datagen.flow_from_dataframe(df_val,
                                            #directory=train_path,
                                            x_col = 'image',
                                            y_col = 'class',
                                            target_size = (IMG_SIZE, IMG_SIZE),
                                            class_mode = 'binary',
                                            batch_size = BATCH,
                                            seed = SEED)

ds_test = test_datagen.flow_from_dataframe(df_test,
                                            #directory=test_path,
                                            x_col = 'image',
                                            y_col = 'class',
                                            target_size = (IMG_SIZE, IMG_SIZE),
                                            class_mode = 'binary',
                                            batch_size = 1,
                                            shuffle = False)








# --------- create base model ----------#
base_model = keras.applications.ResNet152V2(
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, IMG_DEPTH),
    include_top=False)

# --------- FREEZE base model ----------#
base_model.trainable = False

# --------- get pretrained model ----------#
keras.backend.clear_session()

#Input shape = [width, height, color channels]
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model(inputs)

# Head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
# x = layers.Dense(64, activation='relu')(x)
# x = layers.Dropout(0.2)(x)

#Final Layer (Output)
output = layers.Dense(1, activation='sigmoid')(x)

model_pretrained = keras.Model(inputs=[inputs], outputs=output)

model_pretrained.compile(loss='binary_crossentropy',
               optimizer = keras.optimizers.Adam(learning_rate=0.001), metrics='accuracy')

model_pretrained.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-7,
    restore_best_weights=True,
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    factor = 0.2,
    patience = 2,
    min_delt = 1e-7,
    cooldown = 0,
    verbose = 1
)

history = model_pretrained.fit(ds_train,
          batch_size = BATCH, epochs = 10,
          validation_data=ds_val,
          callbacks=[early_stop, learning_rate_reduction],
          steps_per_epoch=(len(df_train)/BATCH),
          validation_steps=(len(df_val)/BATCH))
# history = model_pretrained.fit(ds_train,
#           batch_size = BATCH, epochs = 10,
#           validation_data=ds_val,
#           callbacks=[early_stop, learning_rate_reduction])

test_loss, test_accuracy = model_pretrained.evaluate(ds_test, verbose=0)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


fig, ax = plt.subplots(1, 2)

train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

fig.set_size_inches(20, 10)

ax[0].plot(epochs, train_acc, 'bo-', label='Training Accuracy')
ax[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, 'b-o', label='Training Loss')
ax[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
ax[1].set_title('Training & Validation Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()