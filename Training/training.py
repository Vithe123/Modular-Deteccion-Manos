import os
import numpy as np
#import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Ruta a la carpeta que contiene las imágenes de manos
data_dir = 'Redimencionadas'

# Obtener la lista de nombres de clases (nombres de las subcarpetas)
class_names = os.listdir(data_dir)

# Obtener el número de clases
num_classes = len(class_names)

# Dimensiones de las imágenes de entrada
input_shape = (320, 240, 3) # Cambia el tamaño según tus necesidades

# Crear generadores de datos de entrenamiento y validación
data_generator = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = data_generator.flow_from_directory(
    data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=32,
    class_mode='categorical',
    subset='training')

validation_generator = data_generator.flow_from_directory(
    data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Crear el modelo CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_generator, epochs=30, validation_data=validation_generator)

# Guardar el modelo entrenado
model.save('modelo.h5')