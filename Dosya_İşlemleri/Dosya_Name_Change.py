# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 08:15:30 2021

@author: Gitek_Micro
"""
import glob
import os
for dirname in os.listdir("."):
    if os.path.isdir(dirname):
        for i, filename in enumerate(os.listdir(dirname)):
            os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".bmp")
a = []    
currentDir = os.getcwd()
print(currentDir)
os.chdir(currentDir)
files = os.listdir()
for i,f in enumerate(files):
    if f.endswith(".bmp"):
        #os.rename(f,str(i) + ".bmp")
        print(f,"{}".format(i+1))
        a.append(f)
        if ((i)%2 == 1):
            os.remove(str(a[i]))
images = []    
for k in files:
    if k.endswith(".bmp"):
        images.append(k)
        print(images)

b = []
for a in images:
    a = a.split(".")
    print(a[0])
    b.append(int(a[0]))
    b.sort()
    print(b)
    
for i,s in enumerate(range(len(b))):
    if s.endswith(".jpeg"):
        os.rename(s,"{]}".format(b[i]) + ".bmp")
        images.append(s)
print(images)
a = []    
images.sort()
for i in range(0,len(images),1):
    a.append(sorted(images[i].split(".")))
    print(i)
b.append(a[0:].sort())
