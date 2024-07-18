import cv2

cam = cv2.VideoCapture('/dev/video2')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

i = 0
while True:
    has, frame = cam.read()
    if not has:
        break

    key  = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f'./imagenes/img{i}.jpg', frame)
        i += 1

    cv2.imshow('img', frame)

cv2.destroyAllWindows()
cam.release()
