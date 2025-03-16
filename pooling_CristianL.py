
""""
Ingeniería en Computación Inteligente
Aprendizaje Profundo
Pooling tipo suma
Docente: 
8°D
By: Cristian Armando Larios Bravo
"""

import cv2
import numpy as np

# Cargar la imagen en escala de grises
img = cv2.imread("number-6_2.png", cv2.IMREAD_GRAYSCALE)

# Aplicar umbral binario
_, img_binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Convertir a matriz de 0s y 1s (0 = negro, 1 = blanco)
matriz = (img_binaria / 255).astype(np.uint8)

print(len(matriz))

pool = 3 # 3x3
# stride = 

""" for row in range(len(matriz)):
    # vector = []
    for col in range(len(matriz[row])):
        print(f"{matriz[row][col]},",end="")
        # vector.append(matriz[row][col])#!
    # matrizNueva.append(vector)#!
    print() """
    
newMatrix = np.zeros((int((len(matriz)/3)),int((len(matriz)/3))))

# print(type(int((len(matriz)/3))))

for i in range((int((len(matriz)/3))**2)):
    newMatrix.append(sum)

for row in range(len(matriz)):
    for col in range(len(matriz[row])):
        for n in range(pool**2):
            
            pass

# print(len(newMatrix[0]))
# print(len(newMatrix))
# print(newMatrix)

