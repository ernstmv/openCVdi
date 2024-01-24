import os #import the library for interact with the OS (operative system)
import cv2 #import the cv2 module from opencv
import numpy as np #importh numpy, it's used for operations with matrix
import matplotlib.pyplot as plt #import matplot lib wich is used for making 2d graphics
from zipfile import ZipFile #imports the library for working with zip files
from urllib.request import urlretrieve #urlretrieve downloads and save the file in local
from IPython.display import Image

def download_and_unzip(url, save_path):
    print(f'Downloading and extracting assests...')

    urlretrieve(url, save_path) #downloads and save the file in the current directory

    try:
        with ZipFile(save_path) as z:#opens the zip file specified in save_path in read-only mode
            z.extractall(os.path.split(save_path)[0])#extracts the z files in the specified path between braces
        print ("Done")
    except Exception as e: #any returned exception is saved as e
        print("\n Invalid file.",e) #prints invalid file, and the error

def main():
    #-------------------------------------DOWNLOAD-ZONE-------------------------------------
    URL = r"https://www.dropbox.com/s/qhhlqcica1nvtaw/opencv_bootcamp_assets_NB1.zip?dl=1"
    asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB1.zip")#gets the current local path and joins it with the second path
    if not os.path.exists(asset_zip_path):#if asset_zip_path doesnt exist
        download_and_unzip(URL, asset_zip_path)#download and unzip

    #-------------------------------------READING-FILES-DISPLAYING-ZONE----------------------
    retval=cv2.imread("checkerboard_18x18.png",0)#convert the image to a matrix
    print(retval)#shows the matrix of retval
    print("Image size (H,W) is: ", retval.shape)#print dimensions
    print("Data type os image is: ", retval.dtype)#print bits
    plt.imshow(retval,cmap="gray")#print image in B&W scale
    #plt.show()  # shows the plot FUCKKK!!!

    cb_img_fuzzy=cv2.imread("checkerboard_fuzzy_18x18.jpg",0)
    print(cb_img_fuzzy)
    plt.imshow(cb_img_fuzzy, cmap="gray")
    #plt.show() #shows the plot FUCKKK!!!

    #---------------------------MERGING-COLORS-ZONE------------------------------------------
    img_NZ_bgr=cv2.imread("New_Zealand_Lake.jpg",1)
    b,g,r=cv2.split(img_NZ_bgr)

    plt.figure(figsize=[20,5])

    plt.subplot(141);plt.imshow(r,cmap="gray");plt.title("RED")
    plt.subplot(142);plt.imshow(r,cmap="gray");plt.title("GREEN")
    plt.subplot(143);plt.imshow(r,cmap="gray");plt.title("BLUE")

    imgMerged=cv2.merge((b,g,r))

    plt.subplot(144)
    plt.imshow(imgMerged[:,:,::-1])
    plt.title("Merged")
    plt.show()
    #-----------------CONVERTING-TO-DIFFERENT-COLOR-SPACES-----------------------------------
    img_hsv = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2HSV)

    # Split the image into the B,G,R components
    h, s, v = cv2.split(img_hsv)

    # Show the channels
    plt.figure(figsize=[20, 5])
    plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H Channel");
    plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S Channel");
    plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V Channel");
    plt.subplot(144);plt.imshow(img_NZ_rgb);plt.title("Original");
    #------------------------------SAVING-IMAGES---------------------------------------------
    # save the image
    cv2.imwrite("New_Zealand_Lake_SAVED.png", img_NZ_bgr)

    Image(filename='New_Zealand_Lake_SAVED.png')
    # read the image as Color
    img_NZ_bgr = cv2.imread("New_Zealand_Lake_SAVED.png", cv2.IMREAD_COLOR)
    print("img_NZ_bgr shape (H, W, C) is:", img_NZ_bgr.shape)

    # read the image as Grayscaled
    img_NZ_gry = cv2.imread("New_Zealand_Lake_SAVED.png", cv2.IMREAD_GRAYSCALE)
    print("img_NZ_gry shape (H, W) is:", img_NZ_gry.shape)

if __name__ == '__main__':
    main()