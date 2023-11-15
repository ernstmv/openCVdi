# Steps for creating panoramas

#     Find keypoints in all images
#     Find pairwise correspondences
#     Estimate pairwise Homographies
#     Refine Homographies
#     Stitch with Blending


import os
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
from zipfile import ZipFile

def downloadAndUnzip(url, savePath):
    print (f"Downloading...\n")

    urlretrieve(url, savePath)

    try:
        with ZipFile(savePath) as z:
            z.extractall(os.path.split(savePath)[0])
        print("Done...\n")
    except Exception as e:
        print("Error",e)


def main():
    
    URL = r"https://www.dropbox.com/s/0o5yqql1ynx31bi/opencv_bootcamp_assets_NB9.zip?dl=1"
    asset_path=os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB9.zip")
    if not os.path.exists(asset_path):
        downloadAndUnzip(URL, asset_path)

    imageFiles=glob.glob(f"boat{os.sep}*")
    imageFiles.sort()

    images=[]
    for filename in imageFiles:
        img = cv2.imread(filename)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    num_images=len(images)
    #display images

    plt.figure(figsize=[30,10])
    num_cols=3
    num_rows=math.ceil(num_images/ num_cols)
    for i in range(0, num_images):
        plt.subplot(num_rows, num_cols, i+1)
        plt.axis("off")
        plt.imshow(images[i])

    #user the switcher class
    stitcher=cv2.Stitcher_create()
    status, result=stitcher.stitch(images)

    if status ==0:
        plt.figure(figsize=[30, 10])
        plt.imshow(result)
    plt.show()


if __name__ == '__main__':
    main()