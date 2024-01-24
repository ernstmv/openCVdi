import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import YouTubeVideo, display, Image

def download_and_unzip(url, save_path):
    print("Downloading...")
    urlretrieve(url, save_path)

    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("Invalid file {e}")

def main():
    URL = r"https://www.dropbox.com/s/089r2yg6aao858l/opencv_bootcamp_assets_NB14.zip?dl=1"

    asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB14.zip")

    if not os.path.exists(asset_zip_path):
        download_and_unzip(URL, asset_zip_path)

    video=YouTubeVideo("RyCsSc_2ZEI", width=1024, height=640)
    display(video)

    #a typical caffe model has two files; architecture (.prototxt) and weights (.caffemodel)
    protoFile=("pose_deploy_linevec_faster_4_stages.prototxt")
    weightsFile=os.path.join("model", "pose_iter_160000.caffemodel")

    nPoints=15
    POSE_PAIRS=[
            [0,1],
            [1,2],
            [2,3],
            [3,4],
            [1,5],
            [5,6],
            [6,7],
            [1,14],
            [14,8],
            [8,9],
            [9,10],
            [14,11],
            [11,12],
            [12,13],
    ]

    net=cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    im=cv2.imread("Tiger_Woods_crop.png")
    im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    imWidth=im.shape[1]
    imHeight=im.shape[0]

    Image(filename="Tiger_Woods.png")

    netInputSize=(368,368)
    inpBlob=cv2.dnn.blobFromImage(im, 1.0/255, netInputSize, (0,0,0), swapRB=True, crop=False)
    net.setInput(inpBlob)

    output=net.forward()
    plt.figure(figsize=(20,5))
    for i in range(nPoints):
        probMap=output[0,i,:,:]
        displayMap=cv2.resize(probMap, (imWidth, imHeight), cv2.INTER_LINEAR)

        plt.subplot(2,8,i+1)
        plt.axis("off")
        plt.imshow(displayMap, cmap="jet")

    scaleX=imWidth/output.shape[3]
    scaleY=imHeight/output.shape[2]

    points=[]

    threshold=0.1

    for i in range(nPoints):
        probMap=output[0,i,:,:]
        minVal, prob, minLoc, point=cv2.minMaxLoc(probMap)

        x=scaleX*point[0]
        y=scaleY*point[1]

        if prob>threshold: points.append((int(x), int(y)))
        else: points.append(None)

    imPoints=im.copy()
    imSkeleton=im.copy()

    for i, p in enumerate(points):
        cv2.circle(imPoints, p, 8, (255, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(imPoints, "{}".format(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2, lineType=cv2.LINE_AA)

    for pair in POSE_PAIRS:
        partA=pair[0]
        partB=pair[1]

        if points[partA] and points[partB]:
            cv2.line(imSkeleton, points[partA], points[partB], (255,255,0),2)
            cv2.circle(imSkeleton, points[partA], 8, (255,0,0), thickness=-1, lineType=cv2.FILLED)

    plt.figure(figsize=(50,50))
    plt.subplot(121)
    plt.axis("off")
    plt.imshow(imPoints)

    plt.subplot(122)
    plt.axis("off")
    plt.imshow(imSkeleton)

    plt.show()

if __name__=="__main__":
    main()
