# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:10:49 2023

@author: Bella
"""

from keras.preprocessing.image import ImageDataGenerator
import os
import cv2

# create an instance of ImageDataGenerator
train_datagen = ImageDataGenerator(
        rotation_range=10,
        shear_range= 0.15,
        rescale =1./255,
        width_shift_range= 0.05, 
        height_shift_range= 0.02,
        zoom_range= 0.2
        )

classes = [ "apple_scab"]
base_dir = "C:\\Users\\Bella\\Desktop\\Masters\\ThirdSemester\\GraduationProject\\alexNet-plant-leaf-disease-classification\\data_train_aug\\"

for class_name in classes: 
    print(base_dir)
    class_dir = os.path.join('C:\\Users\\Bella\\Desktop\\Masters\\ThirdSemester\\GraduationProject\\alexNet-plant-leaf-disease-classification\\dataset\\', class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        
    i = 0
    
    for x_batch, y_batch in train_datagen.flow_from_directory(base_dir ,target_size=(200, 200), batch_size=1, class_mode='categorical'):
        for x in x_batch:
            i += 1
            save_path = os.path.join(class_dir, 'new_'+str(i)+'.jpg')
            cv2.imwrite(save_path, x*255)
        if len(os.listdir(class_dir)) > 3155:
            break


