# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:44:37 2022

@author: Gitek_Micro
"""

# Burada kullanıcıdan bir girdi alıyoruz ve o alınan sayının kare
#olup olmadığını kontrol ediyoruz.
import math

while True:
    t=0
    print("Girdiğiniz sayı tam kare olmalıdır !!!")
    print("Bir sayı girer misiniz : ")
    
    
    a=int(input("a : "))
    
    #sonuc = a*a
     
    #print("{} sayısının karesi {} sayısıdır.".format(a,sonuc))
    
    for i in range(0, a // 2 + 2): # (X//2+2) kere dönen döngünün oluşturulması
        if (a == i * i): # Koşulun kontrol edilmesi
            t = 1
            break # Fazladan işlem yaptırmamak için karekökü bulunduğunda döngüden çıkarız
    # Koşulun sağlanıp sağlanmadığını kontrol etmek için “t”yi kullandık, çünkü koşul sağlandığında “t” değişecek
    # Sonucun yazdırılması
    if (t != 0):
        print("tam kare")
        print(math.sqrt(a))
        break
    else:
        print("tam kare degil")



for l in range(1,int(math.sqrt(a) + 1),1):
 print("l :",l)

 
 