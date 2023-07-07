import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy.typing import NDArray
from typing import Optional, Literal, Type, TypedDict

labels = ['PNEUMONIA', 'NORMAL']
IMG_SIZE = 150


def get_training_data(data_dir) -> NDArray:
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # Reshaping images to preferred size
                resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


# Includes both normal and pneumonia cases.
lungs_dataset = get_training_data('chest_xray/')  # list of [image, label]

x_dataset = []
y_dataset = []
for x, y in lungs_dataset:
    x_dataset.append(x)
    y_dataset.append(y)

x_dataset = np.array(x_dataset)
y_dataset = np.array(y_dataset)


X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, random_state=42,
                                                    test_size=0.2, shuffle=True)

# When the Random_state is not defined in the code for every run train data will
#  change and accuracy might change for every run. When the
#  Random_state = " constant integer" is defined then train data will be constant
#  For every run so that it will make easy to debug.

# On a serious note, random_state simply sets a seed to the random generator,
#  so that your train-test splits are always deterministic.
#  If you don't set a seed, it is different each time.

# Normalize data
X_train = np.array(X_train) / 255
X_test = np.array(X_test) / 255

# When using the image as it is and passing through a Deep Neural Network,
#  the computation of high numeric values may become more complex.
# To reduce this we can normalize the values to range from 0 to 1.

# In this way, the numbers will be small and the computation becomes
#  easier and faster.
# As the pixel values range from 0 to 256, apart from 0 the range is 255.
#  So dividing all the values by 255 will convert it to range from 0 to 1.

# It is not needed anymore. The reason for normalizing the images
#  is to avoid the possibility of exploding gradients because
#  of the high range of the pixels [0, 255], and improve the
#  convergence speed. Therefore, you either standardize the each image,
#  so that the range is [-1, 1] or you just divide the with the
#  maximum pixel value as you are doing, so that the range of the
#  pixels is in the [0, 1] range.

# Another reason why you might want to normalize the image data is
#  if you are using transfer learning. For example, if you are using
#  a pre-trained model that has been trained with images with pixels
#  in the [0, 1] range, you should make sure that the inputs you are
#  providing the model are in the same range. Otherwise, your results
#  will be messed up.


y_train = np.array(y_train)
y_test = np.array(y_test)
# resize data for deep learning
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# With data augmentation to prevent overfitting and handling the imbalance in dataset
# Because the dataset is small we "increase" the dataset by change of images parameters.
# In this way we increase our dataset and prevent overfitting.

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


# --------- Building CNN network ----------#

model = Sequential()
model.add(Conv2D(32, (3, 3), strides=1, padding='same',
          activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# binary_crossentropy is chosen because we have a binary ouput, either sick or not.

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss', patience=5, verbose=1, factor=0.2, min_lr=0.00000001, min_delta=0.001)
# min_delta: threshold for measuring the new optimum, to only focus on
#   significant changes.

history = model.fit(X_train, y_train, batch_size=32,
                    epochs=30, validation_split=0.025, callbacks=[learning_rate_reduction])
score = model.evaluate(X_test, y_test, verbose=0)

print("val Loss:", score[0])
print("val Accuracy:", score[1])
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

epochs = [i for i in range(30)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20, 10)

ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
ax[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
ax[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()
