import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Directorio que contiene las imágenes y los archivos XML
data_dir = 'Dataset'

# Tamaño de las imágenes
image_size = (320, 240)

# Leer los archivos XML y extraer las anotaciones
annotations = []
for file in os.listdir(data_dir):
    if file.endswith('.xml'):
        xml_path = os.path.join(data_dir, file)
        image_path = os.path.join(data_dir, file.replace('.xml', '.jpg'))
        if not os.path.exists(image_path):
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        annotations.append({'image_path': image_path, 'boxes': boxes, 'labels': labels})

# Verificar si se encontraron archivos XML válidos
if len(annotations) == 0:
    print("No se encontraron archivos XML válidos en el directorio.")
    exit()

# Dividir los datos en conjuntos de entrenamiento y validación
train_data, val_data = train_test_split(annotations, test_size=0.2, random_state=42)

# Obtener las rutas de las imágenes para los generadores de datos
train_image_paths = [data['image_path'] for data in train_data]
val_image_paths = [data['image_path'] for data in val_data]

# Definir generadores de datos de entrenamiento y validación
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    classes=train_image_paths,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    data_dir,
    classes=val_image_paths,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical')

# Crear el modelo CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))  # 4 clases: Adelante, Atras, Derecha, Izquierda

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Guardar el modelo entrenado
model.save('modelo.h5')