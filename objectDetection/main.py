import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve


FONTFACE=cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE=0.7
THICKNESS=1

def downloadAndUnzip(url,savepath):
    print("Downloading...")

    urlretrieve(url, savepath)

    try:
        with ZipFile(savepath) as z:
            z.extractall(os.path.split(savepath)[0])
        print("Done")
    except Exception as e:
        print(f"Invalid file {e}")


def detectObject(net, im, dim=300):
    blob=cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    objects=net.forward()
    return objects


def display_text(im, text, x, y):
    textSize=cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim=textSize[0]
    baseline=textSize[1]
    cv2.rectangle(
            im, 
            (x,y-dim[1]-baseline),
            (x + dim[0],y+baseline),
            (0,0,0),
            cv2.FILLED,
     )
    cv2.putText(
            im, 
            text, 
            (x,y-5),
            FONTFACE,
            FONT_SCALE,
            (0,255,255),
            THICKNESS,
            cv2.LINE_AA,
    )

def display_objects(im, simple_labels, objects, threshold=0.25):
    rows=im.shape[0]
    cols=im.shape[1]

    for i in range(objects.shape[2]):
        classID=int(objects[0,0,i,1])
        score=float(objects[0,0,i,2])
        
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        # Check if the detection is of good quality
        if score > threshold:
            display_text(im, "{}".format(simple_labels[classID]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Convert Image to RGB since we are using Matplotlib for displaying image
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(30, 10))
    plt.imshow(mp_img)
    plt.show()

def main():
    URL = r"https://www.dropbox.com/s/xoomeq2ids9551y/opencv_bootcamp_assets_NB13.zip?dl=1"

    asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB13.zip")

    if not os.path.exists(asset_zip_path):
       downloadAndUnzip(URL, asset_zip_path)

    classFile="coco_class_labels.txt"

    with open(classFile) as fp:
        simple_labels=fp.read().split("\n")

    modelFile=os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29","frozen_inference_graph.pb")
    configFile=os.path.join("models","ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

    net=cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    

    im=cv2.imread(os.path.join("images", "baseball.jpg"))
    objects=detectObject(net, im)
    display_objects(im,simple_labels, objects, 0.2)
    
    im=cv2.imread(os.path.join("images", "street.jpg"))
    objects=detectObject(net, im)
    display_objects(im,simple_labels, objects, 0.2)

    im=cv2.imread(os.path.join("images", "soccer.jpg"))
    objects=detectObject(net, im)
    display_objects(im,simple_labels, objects, 0.2)
if __name__=="__main__":
    main()
