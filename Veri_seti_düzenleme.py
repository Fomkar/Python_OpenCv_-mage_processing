Kütüphane versiyonları;
tensorflow == 2.6.0 (pip install tensorflow-gpu==2.6.0)
keras == 2.6.0 (pip install keras==2.6.0)
"""

import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import cv2
#from tensorflow.keras.optimizers import SGD
from datetime import datetime
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


save_path = r"C:\Users\AKTS\Desktop\spain\deneme"
       
train_path = r"C:\Users\AKTS\Desktop\spain\English" 


def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

#train_or_test_folder_name = test ya da train ya da validation klasörünün sadece ismi
#dataset_type_name = open ya da close olan klasörün ismi
def dataset_load(path, start_index=None, end_index=None):
    dataset_load = []
    images_ds = []
    string_labels = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_valid = []
    y_valid = []
    i = 0
    for folder in tqdm(os.listdir(path)): 
        images_ds = []
        string_labels = []
        
        for image in tqdm(os.listdir(path+"/"+folder)):
            path_image = os.path.join(path+'/'+folder, image)
            image_data = cv2.imdecode(np.fromfile(path_image, np.uint8), -1)
            # image_data = cv2.resize(image_data, (image_width, image_height))
            if len(image_data.shape) == 2:
                print("path : {}".format(path_image))
            images_ds.append(np.array(image_data))        
            string_labels.append(folder)
            # dataset_load.append([np.array(image_data), folder])
            i = i + 1
            
        # label_train = one_hot(string_labels)
        X_train2, X_rem2, y_train2, y_rem2 = train_test_split(images_ds, string_labels, train_size=0.7, shuffle=True)
        X_valid2, X_test2, y_valid2, y_test2 = train_test_split(X_rem2,y_rem2, test_size=0.5)
        
        for element in range(0, len(X_train2)):
            if not os.path.exists(os.path.join(save_path+'/Train', folder)):
                os.makedirs(os.path.join(save_path+'/Train', folder))
            save_path_image = os.path.join(save_path+'/Train/'+folder, folder+'_'+str(element)+'.bmp')
            cv2.imwrite(save_path_image, X_train2[element])
            # X_train.append(np.array(X_train2[element]))
            # y_train.append(y_train2[element])
            
            
        for element in range(0, len(X_test2)):
            if not os.path.exists(os.path.join(save_path+'/Test', folder)):
                os.makedirs(os.path.join(save_path+'/Test', folder))
            save_path_image = os.path.join(save_path+'/Test/'+folder, folder+'_'+str(element)+'.bmp')
            cv2.imwrite(save_path_image, X_train2[element])
            # X_test.append(np.array(X_test2[element]))
            # y_test.append(y_test2[element])

        for element in range(0, len(X_valid2)):
            if not os.path.exists(os.path.join(save_path+'/Validate', folder)):
                os.makedirs(os.path.join(save_path+'/Validate', folder))
            save_path_image = os.path.join(save_path+'/Validate/'+folder, folder+'_'+str(element)+'.bmp')
            cv2.imwrite(save_path_image, X_train2[element])
            # X_valid.append(np.array(X_valid2[element]))
            # y_valid.append(y_valid2[element])             

    
    return X_train, y_train, X_test, y_test, X_valid, y_valid
    
x_train, y_train, x_test, y_test, x_valid, y_valid = dataset_load(train_path)


y_train = one_hot(y_train)
y_test = one_hot(y_test)
y_valid = one_hot(y_valid)


x_train = np.array(x_train,  dtype='uint8')
x_test = np.array(x_test)
x_valid = np.array(x_valid)
