
# Desarrollar una red neuronal que clasifique dos tipos de imagenes de numeros
# Red neuronal MLP

"""
Dataset: MNIST
* 75 % para entrenamiento
* 15 % para validación
* 10 % para prueba
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset: MNIST
from keras.datasets import mnist

# seed 0
np.random.seed(0)

# Importar el dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar los datos
X_train = X_train / 255

# Redimensionar los datos
X_train = X_train.reshape(X_train.shape[0], -1)

# Dividir el dataset
# 75 % para entrenamiento
# 15 % para validación
# 10 % para prueba
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.4, random_state=0)

# Normalizar los datos
escalador = StandardScaler()
X_train = escalador.fit_transform(X_train)
X_val = escalador.transform(X_val)
X_test = escalador.transform(X_test)

# Modelo de la red neuronal MLP con librerias
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la red neuronal
modelo = Sequential()

# Añadir capas
modelo.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],))) # Capa de entrada
modelo.add(Dense(units=64, activation='relu')) # Capa oculta
modelo.add(Dense(units=10, activation='softmax')) # Capa de salida

# Compilar el modelo
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluar el modelo
test_loss, test_acc = modelo.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {test_loss}")
print(f"Exactitud en el conjunto de prueba: {test_acc}")

# Predicciones
y_pred = modelo.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Visualizar las predicciones
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title(f"Real: {y_test[i]}\nPredicción: {y_pred[i]}")
plt.tight_layout()
plt.show()

# Guardar el modelo
modelo.save('modelo_mnist.h5')