import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
from zipfile import ZipFile

def downloadAndUnzip(url, savePath):
    urlretrieve(url, savePath)    
    
    try:
        with ZipFile(savePath) as z:
            z.extractall(os.path.split(savePath)[0])
        print("Done...")
    except Exception as e:
        print("Error downloading",e)

def readImagesAndTimes():
    filenames=["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]

    times=np.array([1/30.0, 0.25, 2.5, 15.0], dtype=np.float32)

    images=[]

    for filename in filenames:
        im=cv2.imread(filename)
        im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images.append(im)
    return images, times


def main():
    URL = "https://www.dropbox.com/s/qa1hsyxt66pvj02/opencv_bootcamp_assets_NB10.zip?dl=1"
    assestPath=os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB10.zip")
    if not os.path.exists(assestPath):
        downloadAndUnzip(URL, assestPath)

    images, times = readImagesAndTimes()

    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    calibrateDebevec=cv2.createCalibrateDebevec()
    responseDebevec=calibrateDebevec.process(images, times)

    x=np.arange(256, dtype=np.uint8)
    y=np.squeeze(responseDebevec)

    ax=plt.figure(figsize=(30,10))
    plt.title("Debevec Inverse Camera Response Function", fontsize=24)
    plt.xlabel("Measured Pixel Value", fontsize=22)
    plt.ylabel("Calibrated Intensity", fontsize=22)
    plt.xlim([0,260])
    plt.grid()
    plt.plot(x,y[:,0],"r",x,y[:,1],"g",x,y[:,2],"b")
    # plt.show()

    mergeDebevec=cv2.createMergeDebevec()
    hdrDebevec=mergeDebevec.process(images, times, responseDebevec)

    tonemapDrago=cv2.createTonemapDrago(1.0,0.7)
    ldrDrago=tonemapDrago.process(hdrDebevec)
    ldrDrago=3*ldrDrago

    plt.figure(figsize=(20,10)); plt.imshow(np.clip(ldrDrago,0,1)); plt.axis("off")

    cv2.imwrite("ldr-Drago.jpg", ldrDrago*255)
    print("saved ldr-Drago.jpg")

    print("Tonemap using Reinhard's method ...")
    tonmeapReinhard=cv2.createTonemapReinhard()
    ldrReinhard=tonmeapReinhard.process(hdrDebevec)

    plt.figure(figsize=(20,10)); plt.imshow(np.clip(ldrReinhard, 0, 1)); plt.axis("off")

    cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard*255)
    print("Saved ldr-Reinhard.jpg")

    print("Tonemaping using Mantiuk's method...")
    tonemapMantiuk=cv2.createTonemapMantiuk(2.2,0.85,1.2)
    ldrMantiuk=tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk=3*ldrMantiuk

    plt.figure(figsize=(20,10)); plt.imshow(np.clip(ldrMantiuk, 0,1)); plt.axis("off")

    cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk*255)
    print("saved ldr-Mantiuk.jpg")

    plt.show()


    
    

if __name__ == "__main__":
    main()