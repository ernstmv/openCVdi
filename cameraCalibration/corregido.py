import cv2
import numpy as np

# Cargar los parámetros de calibración
with np.load('calibracion.npz') as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# Inicializar la cámara
cap = cv2.VideoCapture('/dev/video2')  # Ajusta este número si es necesario
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Obtener las dimensiones del frame
ret, frame = cap.read()
if not ret:
    print("No se pudo obtener el frame")
    exit()

height, width = frame.shape[:2]

# Calcular la nueva matriz de la cámara
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo obtener el frame")
        break

    # Aplicar la corrección de distorsión
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Recortar la imagen (opcional)
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    # Mostrar la imagen original y la corregida
    cv2.imshow('Original', frame)
    cv2.imshow('Corregida', undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
