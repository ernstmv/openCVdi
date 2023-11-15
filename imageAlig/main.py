import os
import numpy as np
import cv2

from matplotlib import pyplot as plt
from zipfile import ZipFile
from urllib.request import urlretrieve

def download_and_unzip(url, savePath):
    print(f"Downloading...\n")

    urlretrieve(url, savePath)
    try:
        with ZipFile(savePath) as z:
            z.extractall(os.path.split(savePath)[0])
        print("Done\n")
    except Exception as e:
        print("Invalid file\n")


def main():
    URL = r"https://www.dropbox.com/s/zuwnn6rqe0f4zgh/opencv_bootcamp_assets_NB8.zip?dl=1"

    assestZipPath=os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB8.zip")

    if not os.path.exists(assestZipPath):
        download_and_unzip(URL, assestZipPath)
    
    refFilename="form.jpg"
    print("Reading reference image: ", refFilename)
    im1=cv2.imread(refFilename, cv2.IMREAD_COLOR)
    im1=cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    imFilename="scanned-form.jpg"
    print("Reading image to align", imFilename)

    im2=cv2.imread(imFilename, cv2.IMREAD_COLOR)
    im2=cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=[20,10])
    plt.subplot(121); plt.axis('off');plt.imshow(im1); plt.title("Original form")
    plt.subplot(122); plt.axis('off');plt.imshow(im2); plt.title("Scanned form")

    im1Gray=cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2Gray=cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)   

    MAX_NUM_FEATURES=500
    orb=cv2.ORB_create(MAX_NUM_FEATURES)
    keypoints1, descriptors1=orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2=orb.detectAndCompute(im2Gray, None)

    im1_display=cv2.drawKeypoints(im1, keypoints1, outImage=np.array([]), color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im2_display=cv2.drawKeypoints(im2, keypoints2, outImage=np.array([]), color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize=[20,10])
    plt.subplot(121); plt.axis('off'); plt.imshow(im1_display); plt.title("Original Form...")
    plt.subplot(122); plt.axis('off'); plt.imshow(im2_display); plt.title("Scanned Form...")

    matcher=cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches=list(matcher.match(descriptors1, descriptors2, None))
    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches)*0.1)
    matches=matches[:numGoodMatches]

    im_matches=cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    plt.figure(figsize=[40, 10])
    plt.imshow(im_matches); plt.axis('off');plt.title("ORIGINAL FORM")

    points1=np.zeros((len(matches),2), dtype=np.float32)
    points2=np.zeros((len(matches),2), dtype=np.float32)

    for i, match in enumerate (matches):
        points1[i,:]=keypoints1[match.queryIdx].pt
        points2[i,:]=keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    height, width, channels = im1.shape
    im2_reg=cv2.warpPerspective(im2, h, (width, height))

    plt.figure(figsize=[20,10])
    plt.subplot(121);plt.imshow(im1); plt.axis('off'); plt.title("Original form")
    plt.subplot(122);plt.imshow(im2_reg); plt.axis('off'); plt.title("Scanned form")
    plt.show()

if __name__ == '__main__':
    main()