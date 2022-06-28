import cv2 # Open cv kütüphanesini ileride görüntü işlemlerinde kullanmak üzere çağırdık.
import os  # Dosya işlemleri için os modülünü kodumuza ekliyoruz.
import numpy as np # Matematiksel işlemler için numy kütüphanesini kodumuza ekledik.
from glob import glob # Klasörlerden resim okuma işlemlerinde kullanmak üzere glob kütüphanesini çağırdık.
from statistics import median

camera=cv2.VideoCapture(0) # Bilgisayarımızın kamerasını kullanacağımızı belirttik.
images = [] # Daha sonra resimlerimizi tutacağımız bir dizi oluşturduk.


#Bu fonksiyonda "images" isimli bir klasör oluşturduk.
def create_directory():
    path=os.getcwd()
    print(path)
    os.mkdir("images")
    #print(os.listdir())
    os.chdir(path+"/images")
    new_path = os.getcwd()
    print(new_path)

#save_image fonksiyonunda daha önce oluşturduğumuz "images" isimli klasöre kameradan aldığımız fotoğrafları kaydettik.
def save_image():
    path = os.getcwd()
    counter=0
    while True:
        ret,frame=camera.read()
        counter+=1
        cv2.imwrite(path+"/" + str(counter) + '.jpg', frame)
        cv2.imshow("Frame",frame)
        if counter>=100:
            break
#Bu fonksiyon da ise yine daha önce oluşturduğumuz images isimli diziye resimlerimizi ekledik.
def get_images():
    path = os.getcwd()
    print(path)
    images_path=glob(path+"\*.jpg")
    for i in images_path:
        images.append(i)
    return images

#Bu fonksiyonda images klasöründen aldığımız görüntüleri gri formata çevirerek yeni oluşturduğumuz gray_images klasörüne kaydettik.
def convert_gray_image():
    path = os.getcwd()
    os.mkdir("gray_images")
    os.chdir(path+"/gray_images")
    new_path = path+"/gray_images"
    images_path = glob(path + "\images\*.jpg")
    counter=0
    for i in images_path:
        image = cv2.imread(i)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        counter += 1
        cv2.imwrite(new_path + "/gray_images" + str(counter) + '.jpg', gray_image)

# Aşağıda ki iki fonksiyonda ise farklı yollarla images klasörünün ortasında bulunan fotoğrafı aldık.
def get_median():
    median_value=len(images)/2
    median_value=int(median_value)
    median_img=cv2.imread(images[median_value])
    cv2.imshow("Median",median_img)
    cv2.waitKey(0)
     

def get_median2():
    path = os.getcwd()
    array=[]
    numbers=[]
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            print(filename)
            array.append(filename)
            for i in array:
                name=i.split(".")
                print(name[0])
            name[0]=int(name[0])
            numbers.append(name[0])
            continue
        else:
            continue
    
    print(numbers)
    median_value=median(numbers)
    print("Veri Setinin ortasında ki değer: {}".format(median_value))
    median_value=int(median_value)
    median_img=cv2.imread(images[median_value])
    cv2.imshow("Median",median_img)
    cv2.waitKey(0)
    
# Fonksiyonları sırası ile çağırdık.
create_directory()
save_image()
get_images()
convert_gray_image()
get_median()
get_median2()

#print("Veri setinin ortasında ki % s"%median(images))

camera.release()
cv2.destroyAllWindows()