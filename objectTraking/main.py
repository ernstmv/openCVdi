import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from IPython.display import YouTubeVideo, display, HTML
from base64 import b64encode

def downloadAndUnzip(url, savePath):
    print("Downloading\n")
    urlretrieve(url, savePath)

    try:
        with ZipFile(savePath) as z:
            z.extractall(os.path.split(savePath)[0])
        print("Done\n")
    except Exception as e:
        print("Error",e)

def drawRectangle(frame, bbox):
    p1=(int(bbox[0]), int(bbox[1]))
    p2=(int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
    cv2.rectangle(frame, p1, p2, (255,0,0),2,1)

def displayRectangle(frame, bbox):
    plt.figure(figsize=(20,10))
    frameCopy=frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy=cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis("off")

def drawText(frame, txt, location, color=(50,170,50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX,1,color, 3)




def main():
    URL = r"https://www.dropbox.com/s/ld535c8e0vueq6x/opencv_bootcamp_assets_NB11.zip?dl=1"
    assestZipPath=os.path.join(os.getcwd(), "opencv_bootcamp_assetsNB11.zip")

    if not os.path.exists(assestZipPath):
        downloadAndUnzip(URL, assestZipPath)

    video=YouTubeVideo("XkJCvtCRdVM",width=1024, height=640)
    display(video)

    video_input_file_name="race_car.mp4"

    #set up tracker
    tracker_types=[
        "BOOSTING",
        "MIL",
        "KCF",
        "CSRT",
        "TLD",
        "MEDIANFLOW",
        "GOTURN",
        "MOSSE"
    ]
     #change the index to change the tracker type
    tracker_type=tracker_types[2]
    
    if tracker_type == "BOOSTING":
        tracker = cv2.legacy.TrackerBoosting.create()
    elif tracker_type == "MIL":
        tracker = cv2.legacy.TrackerMIL.create()
    elif tracker_type == "KCF":
        tracker = cv2.TrackerKCF.create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT.create()
    elif tracker_type == "TLD":
        tracker = cv2.legacy.TrackerTLD.create()
    elif tracker_type == "MEDIANFLOW":
        tracker = cv2.legacy.TrackerMedianFlow.create()
    elif tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN.create()
    else:
        tracker = cv2.legacy.TrackerMOSSE.create()
    
    #read video
    video=cv2.VideoCapture(video_input_file_name)
    ok, frame=video.read()

    #exit if video not opened
    if not video.isOpened():
        print("Could not open the video\n")
        sys.exit()
    else:
        width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videoOutputFileName="race-car-"+tracker_type+".mp4"
    videoOut = cv2.VideoWriter(videoOutputFileName, cv2.VideoWriter_fourcc(*"XVID"), 10, (width, height))

    #define a bounding box
    bbox=(1300, 405, 160, 120)
    #bbox=cv2.selectROI(frame, False)
    #print(bbox)
    displayRectangle(frame, bbox)

    #initialize tracker with first frame and bounding box
    ok=tracker.init(frame, bbox)

    #read frame and track object
    while True:
        ok, frame= video.read()

        if not ok:
            break
            
        timer = cv2.getTickCount()
        ok, bbox=tracker.update(frame)
        fps=cv2.getTickFrequency()/(cv2.getTickCount()-timer)

        if ok:
            drawRectangle(frame, bbox)
        else:
            drawText(frame, "FPS : "+str(int(fps)),(80,100))

        videoOut.write(frame)
    video.release()
    videoOut.release()

    # # Installing ffmpeg
    # !apt-get -qq install ffmpeg 

    # # Change video encoding of mp4 file from XVID to h264 
    # !ffmpeg -y -i {video_output_file_name} -c:v libx264 $"race_car_track_x264.mp4"  -hide_banner -loglevel error

    mp4=open("race_car_track_x264.mp4","rb").read()
    dataUrl="data:video/mp4lbase64,"+b64encode(mp4).decode()

    HTML(f"""<video width=1024 controls><source src="{dataUrl}" type="video/mp4"></video>""")
    video = YouTubeVideo("pk3tmdRX4ww", width=1024, height=640)
    display(video)

    video = YouTubeVideo("6gGDf-7ypBE", width=1024, height=640)
    display(video)

    video = YouTubeVideo("0bnWxc4zMvY", width=1024, height=640)
    display(video)

if __name__ == "__main__":
    main()