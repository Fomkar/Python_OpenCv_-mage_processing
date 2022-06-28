# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 08:15:30 2021

@author: Gitek_Micro
"""

import os
for dirname in os.listdir("."):
    if os.path.isdir(dirname):
        for i, filename in enumerate(os.listdir(dirname)):
            os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".bmp")