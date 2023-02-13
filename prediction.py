# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 00:08:15 2023

@author: Bella
"""

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

model = load_model("C:\\Users\\Bella\\Desktop\\Masters\\ThirdSemester\\GraduationProject\\alexNet-plant-leaf-disease-classification\\AlexNetModel_2.hdf5")


base_dir = "C:\\Users\\Bella\\Desktop\\Masters\\ThirdSemester\\GraduationProject\\alexNet-plant-leaf-disease-classification\\data"

test_datagen = ImageDataGenerator()
test_set = test_datagen.flow_from_directory(base_dir+'/testing',
                                            target_size=(200, 200),
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=False
                                            )

# Get predictions on test set
predictions = model.predict(test_set)
pred_classes = np.argmax(predictions, axis=1)

true_labels = test_set.classes

confusion_matrix = confusion_matrix(true_labels, pred_classes)
confusion_matrix = np.round(confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100, 2)

import seaborn as sns
sns.heatmap(confusion_matrix, annot = True, fmt='.2f')
plt.show()

test_loss, test_acc = model.evaluate(test_set)
print('Test accuracy:', test_acc)
print('Test Loss:', test_loss)

from sklearn.metrics import recall_score
recall = recall_score(true_labels, pred_classes, average='weighted')

from sklearn.metrics import precision_score
precision = precision_score(true_labels, pred_classes, average='weighted')

from sklearn.metrics import f1_score
f1 = f1_score(true_labels, pred_classes, average='weighted')

print('Recall:', recall)
print('Precision:', precision)
print('F1-score:', f1)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_true_binary = encoder.fit_transform(true_labels.reshape(-1, 1)).toarray()
y_pred_binary = encoder.fit_transform(pred_classes.reshape(-1, 1)).toarray()

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true_binary.ravel(), y_pred_binary.ravel())

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_true_binary, y_pred_binary, average='weighted')

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()