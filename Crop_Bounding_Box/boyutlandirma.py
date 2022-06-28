# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 10:28:04 2021

@author: Gitek_Micro
"""
import os

import cv2 as cv
from glob import glob
import numpy as np


images = []
#Bu fonksiyon da ise yine daha önce oluşturduğumuz images isimli diziye resimlerimizi ekledik.
def get_images():
    path = os.getcwd()
    print(path)
    images_path=glob(path+"\*.png")
    for count, i in enumerate(images_path,start=1):
        images.append(i)
        img  = cv.imread(i)
        # img = cv.resize(img, (400,400))
        # # os.rename(rimages[i])
        # cv.imwrite(str(count)+".jpg", img)
       
        
        # cv.imwrite(str+".jpg", img)
        
        if i.endswith(".png"):
        #     print(i)
            os.remove(i)
    # for a in range(1,224):
    #      # cv.imwrite(str(a)+".jpg", img)
      
    return images

get_images()

