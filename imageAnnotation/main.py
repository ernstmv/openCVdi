import os
import  cv2
import matplotlib
import numpy as np
import  matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve
matplotlib.rcParams['figure.figsize']=(9.0, 9.0)

def download_and_unzip(url, save_path):
    print(f'Downloading files...')
    urlretrieve(url, save_path)
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
            print ("Done")
    except Exception as e:
        print("Invalid file",e)
            

def main():
    URL = r"https://www.dropbox.com/s/48hboi1m4crv1tl/opencv_bootcamp_assets_NB3.zip?dl=1"
    asset_zip_path=os.path.join(os.getcwd(),"openvc_bootcamp_assets_NB3.zip")

    if not os.path.exists(asset_zip_path):
        download_and_unzip(URL, asset_zip_path)
    image=cv2.imread("Apollo_11_Launch.jpg",cv2.IMREAD_COLOR)
    plt.imshow(image[:,:,::-1])
    plt.show()

    imageLine=image.copy()
    cv2.line(imageLine,(200, 100),(400, 100),(0, 255, 255), thickness=5, lineType=cv2.LINE_AA)
    plt.imshow(imageLine[:,:,::-1])
    plt.show()

    imageCircle=image.copy()
    cv2.circle(imageCircle, (900, 500), 100, (0,0,255), thickness=5, lineType=cv2.LINE_AA)
    plt.imshow(imageCircle[:,:,::-1])
    plt.show()

    imageRectangle=image.copy()
    cv2.rectangle(imageRectangle, (500, 100),(700, 600), (255,0,255),thickness=-5, lineType=cv2.LINE_AA)
    plt.imshow(imageRectangle[:,:,::-1])
    plt.show()

    imageText=image.copy()
    text="Apollo 11 Saturn V Launch ,July 16, 1969"
    fontScale=-2.3
    fontFace=cv2.FONT_HERSHEY_PLAIN
    fontColor=(0,255,0)
    fontThickness=2
    cv2.putText(imageText, text, (200,700), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA)
    plt.imshow(imageText[:,:,::-1])
    plt.show()

if __name__=="__main__":
    main()