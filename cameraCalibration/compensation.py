import cv2
import numpy as np
import glob

# Definir el tamaño del tablero de ajedrez
chess_board_size = (7, 7)  # Número de esquinas internas

# Preparar los puntos del objeto (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chess_board_size[0] * chess_board_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chess_board_size[0], 0:chess_board_size[1]].T.reshape(-1,2)

# Arrays para almacenar puntos de objeto y puntos de imagen de todas las imágenes
objpoints = [] # puntos 3d en el espacio del mundo real
imgpoints = [] # puntos 2d en el plano de la imagen

# Obtener la lista de imágenes
images = glob.glob('./imagenes/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontrar las esquinas del tablero
    ret, corners = cv2.findChessboardCorners(gray, chess_board_size, None)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Dibujar y mostrar las esquinas
        cv2.drawChessboardCorners(img, chess_board_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibración
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Imprimir los resultados
print("Matriz de la cámara:")
print(camera_matrix)
print("\nCoeficientes de distorsión:")
print(dist_coeffs)

# Guardar los resultados
np.savez('calibracion.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
