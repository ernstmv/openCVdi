import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import Image

def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets... ", end=" ")
    urlretrieve(url, save_path)

    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
            print(f"Successfully downloaded and extracted assets")
    except Exception as e:
        print(f"\nFailed to download and extract assets",e)

def main():
    URL = r"https://www.dropbox.com/s/rys6f1vprily2bg/opencv_bootcamp_assets_NB2.zip?dl=1"

    asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB2.zip")

    # Download if assest ZIP does not exists. 
    if not os.path.exists(asset_zip_path):
        download_and_unzip(URL, asset_zip_path)

    retval=cv2.imread("checkerboard_18x18.png",0)
    plt.imshow(retval, cmap="gray")
    print(retval)
    plt.show()

    print(retval[0,0])
    print(retval[0,6])

    retvalCopy=retval.copy()
    retvalCopy[2,2]=200
    retvalCopy[2,3]=200
    retvalCopy[3,2]=200
    retvalCopy[3,3]=200
    plt.imshow(retvalCopy,cmap="gray")
    plt.show()

    img_NZ_bgr=cv2.imread("New_Zealand_Boat.jpg",cv2.IMREAD_COLOR)
    print(img_NZ_bgr)
    img_NZ_rgb=img_NZ_bgr[:,:,::-1]
    plt.imshow(img_NZ_rgb,cmap="gray")
    plt.show()

    cropped_region=img_NZ_rgb[200:400,300:600]
    plt.imshow(cropped_region)
    plt.show()

    resized_cropped_region_2x=cv2.resize(cropped_region, None, fx=20,fy=1)
    plt.imshow(resized_cropped_region_2x)
    plt.show()

    desired_width = 100
    desired_height = 200
    dim = (desired_width, desired_height)
    resized_cropped_region = cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)
    plt.imshow(resized_cropped_region)
    plt.show()

    desired_width=100
    aspect_ratio=desired_width/cropped_region.shape[1]
    desired_height=int (cropped_region.shape[0]*aspect_ratio)
    dim=(desired_width, desired_height)

    resized_cropped_region=cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)
    plt.imshow(resized_cropped_region)
    plt.show()

    resized_cropped_region_2x=resized_cropped_region_2x[:,:,::-1]
    cv2.imwrite("resized_cropped_region_2x.png",resized_cropped_region_2x)
    Image(filename="resized_cropped_region_2x.png")

    cropped_region=cropped_region[:,:,::-1]
    cv2.imwrite("cropped_region.png",cropped_region)
    Image(filename="cropped_region.png")

    img_NZ_rgb_flipped_horz = cv2.flip(img_NZ_rgb, 1)
    img_NZ_rgb_flipped_vert = cv2.flip(img_NZ_rgb, 0)
    img_NZ_rgb_flipped_both = cv2.flip(img_NZ_rgb, -1)

    # Show the images
    plt.figure(figsize=(18, 5))
    plt.subplot(141);plt.imshow(img_NZ_rgb_flipped_horz);plt.title("Horizontal Flip");
    plt.subplot(142);plt.imshow(img_NZ_rgb_flipped_vert);plt.title("Vertical Flip");
    plt.subplot(143);plt.imshow(img_NZ_rgb_flipped_both);    plt.title("Both Flipped");
    plt.subplot(144);plt.imshow(img_NZ_rgb);plt.title("Original");

    plt.show()

if __name__ == "__main__":
    main()