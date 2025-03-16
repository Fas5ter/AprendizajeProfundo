import cv2
import numpy as np

# Cargar imagen en escala de grises
imagen = cv2.imread("imagen.jpg", cv2.IMREAD_GRAYSCALE)

# Crear desbordamiento sumando un valor alto
brillo_alto = imagen + 50  # Posible overflow
brillo_alto = np.clip(brillo_alto, 0, 255)  # Corrección con saturación

# Crear subdesbordamiento restando un valor alto
brillo_bajo = imagen - 200  # Posible underflow
brillo_bajo = np.clip(brillo_bajo, 0, 255)  # Corrección con saturación

# Mostrar imágenes
cv2.imshow("Original", imagen)
cv2.imshow("Overflow Corregido", brillo_alto)
cv2.imshow("Underflow Corregido", brillo_bajo)

cv2.waitKey(0)
cv2.destroyAllWindows()