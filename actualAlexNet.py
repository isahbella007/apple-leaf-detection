# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 16:53:13 2023

@author: Bella
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras import   callbacks
from keras.callbacks import ModelCheckpoint
import os

import time
import matplotlib.pyplot as plt

# Initialize the model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(200,200,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(200,200,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))

# 3rd Fully Connected Layer
#Output Layer
model.add(Dense(4))
model.add(BatchNormalization())
model.add(Activation('softmax'))

# # Output Layer
# model.add(Dense(4))
# model.add(BatchNormalization())
# model.add(Activation('softmax'))

# from keras.utils import plot_model
# plot_model(model, to_file ='model.png', show_shapes=True, show_layer_names=True)
model.summary()


#Compile the model
from keras import optimizers
model.compile(optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9, decay=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()

valid_datagen = ImageDataGenerator()

batch_size = 32

base_dir = "C:\\Users\\Bella\\Desktop\\Masters\\ThirdSemester\\GraduationProject\\alexNet-plant-leaf-disease-classification\\data"

training_set = train_datagen.flow_from_directory(base_dir+'/training',
                                                  target_size=(200, 200),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

valid_set = valid_datagen.flow_from_directory(base_dir+'/validation',
                                            target_size=(200, 200),
                                            batch_size=batch_size,
                                            class_mode='categorical')

class_dict = training_set.class_indices
print(class_dict)

li = list(class_dict.keys())
print(li)

train_num = training_set.samples
valid_num = valid_set.samples

print("valid set numbers is:", valid_num)

print("test set numbers is:", train_num)
start_time = time.time()


weightpath = "best_weights_2_1.hdf5"

earlyStopping = callbacks.EarlyStopping(monitor = 'val_accuracy', mode = "max", patience = 10)
checkpoint = ModelCheckpoint(weightpath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [earlyStopping, checkpoint]

#fitting images to CNN
history = model.fit(training_set,
                          validation_data=valid_set,
                          epochs=50,
                          validation_steps=valid_num//batch_size,
                          callbacks=callbacks_list)
#saving model
filepath="AlexNetModel_2_1.hdf5"
model.save(filepath)

end_time = time.time()

total_time = end_time - start_time
avergae_time = total_time/50

print("Totoal time take:", total_time)
print("Average time per epoch: ", avergae_time)
# Plot the test accuracy vs epoch graph
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Test'], loc='upper left')
plt.show()