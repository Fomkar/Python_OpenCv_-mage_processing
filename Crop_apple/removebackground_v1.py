# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 08:53:25 2022

@author: Gitek_Micro
"""

# Requires "requests" to be installed (see python-requests.org)
import requests
import os



currentDir = 'D:/Bakground'
print(currentDir)
os.chdir(currentDir)
files = os.listdir()
idx=0
for i,f in enumerate(files):
    if f.endswith(".jpg"):
        response = requests.post(
    'https://api.remove.bg/v1.0/removebg',
    files={'image_file': open(f, 'rb')},
    data={'size': 'auto'},
    headers={'X-Api-Key': '4VrVjwkBsbW8cKE5sr3LkoPr'},)
    if response.status_code == requests.codes.ok:
        idx +=1
        with open('D:/Bakground/out'+'_'+str(idx)+'.png', 'wb') as out:
            out.write(response.content)
    else:
            print("Error:", response.status_code, response.text)