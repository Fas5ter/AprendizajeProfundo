import cv2
import numpy as np
import matplotlib.pyplot as plt

I = np.array([
     [10,20,30,40,50,0,0],
     [20,30,40,50,60,0,0],
     [30,40,50,60,70,0,0],
     [40,50,60,70,80,0,0],
     [50,60,70,80,90,0,0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0,1,]])

I2 = np.zeros(5,5)
window_size = 3

for i in range(len(I2)):
    for j in range(len(I2)):
        for k in range(3):
            window = I[
            i * window_size : (i + 1) * window_size,
            j * window_size : (j + 1) * window_size
        ]
        newMatrixP3[i, j] = np.sum(window) # suma
        pass