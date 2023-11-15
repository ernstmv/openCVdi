import cv2
import os
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
from zipfile import ZipFile

from IPython.display import YouTubeVideo, display, HTML
from base64 import b64encode

def downloadAndUnzip(url, savePath):
    print(f"Downloading...\n")
    urlretrieve(url, savePath)

    try:
        with ZipFile(savePath) as z:
            z.extractall(os.path.split(savePath)[0])
            print("Done...\n")
    except Exception as e:
        print("Invalid file", e)


def main():
    URL = r"https://www.dropbox.com/s/p8h7ckeo2dn1jtz/opencv_bootcamp_assets_NB6.zip?dl=1"
    assestZipPath=os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB6.zip")

    if not os.path.exists(assestZipPath):
        downloadAndUnzip(URL, assestZipPath)

    source='race_car.mp4' #source =0 for webcam
    cap=cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error opening video stream or file")

    ret, frame = cap.read()
    plt.imshow(frame[...,::-1])

    video = YouTubeVideo("RwxVEjv78LQ", width=700, height=438)
    display(video)

    frame_width = int (cap.get(3))
    frame_height = int (cap.get(4))

    out_avi=cv2.VideoWriter("race_car.avi", cv2.VideoWriter_fourcc("M","J","P","G"),10,(frame_width, frame_height))
    # out_mp4=cv2.VideoWriter("race_car.mp4", cv2.VideoWriter_fourcc(*"XVID"),10,(frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
    
        if ret:
            out_avi.write(frame)
            # out_mp4.write(frame)

        else:
            break
    
    cap.release()
    out_avi.release()
    # out_mp4.release()





if __name__ == '__main__':
    main()