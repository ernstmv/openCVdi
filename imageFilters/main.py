import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

PREVIEW =0
BLUR    =1
FEATURES=2
CANNY   =3

def main():
    feature_params=dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)
    s=0

    if len(sys.argv)>1:
        s=sys.argv[1]
    imageFilter=PREVIEW
    alive=True

    winName="Camera Filters"
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    result=None

    source=cv2.VideoCapture(s)

    while alive:
        hasFrame, frame = source.read()

        if not hasFrame: break

        frame = cv2.flip(frame, 1)

        if imageFilter==PREVIEW: result=frame
        elif imageFilter==CANNY: result=cv2.Canny(frame, 80, 150)
        elif imageFilter==BLUR: result=cv2.blur(frame, (13, 13))
        elif imageFilter==FEATURES: 
            result = frame
            frameGray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners=cv2.goodFeaturesToTrack(frameGray, **feature_params)
            if corners is not None:
                for x, y in np.float32(corners).reshape(-1,2):
                    cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)

        cv2.imshow(winName, result)
        plt.show()
        key=cv2.waitKey(1)
        if key==ord("Q") or key ==ord("q") or key==27: alive=False
        elif key == ord("C") or key == ord("c"): imageFilter = CANNY
        elif key == ord("B") or key == ord("b"): imageFilter = BLUR
        elif key == ord("F") or key == ord("f"): imageFilter = FEATURES
        elif key == ord("P") or key == ord("p"): imageFilter = PREVIEW

    source.release()
    cv2.destroyWindow(winName)

if __name__ == "__main__":
    main()

