import cv2
from keras.models import load_model
import numpy as np

# Ruta del archivo .h5 del modelo
ruta_modelo = r"H:\Usuarios\ArturoVithe\Escritorio\Archivos\CUCEI\Modular\Modular-Deteccion-Manos\Training\modelo_final.h5"

# Cargar el modelo
modelo = load_model(ruta_modelo)

# Dimensiones de las imágenes de entrada
input_shape = (320, 240, 3)  # Misma dimensión que se utilizó para entrenar el modelo

# Clases del modelo (en el mismo orden que se utilizó para entrenar el modelo)
class_names = ["Adelante", "Atras", "Derecha", "Izquierda"]  # Reemplaza con las clases reales del modelo

# Inicializar la captura de video desde la cámara web
cap = cv2.VideoCapture(1)  # Índice "0" para la cámara web predeterminada

while True:
    # Leer un fotograma de la cámara web
    ret, frame = cap.read()

    # Redimensionar el fotograma a las dimensiones de entrada del modelo
    frame = cv2.resize(frame, (input_shape[1], input_shape[0]))

    # Preprocesar el fotograma (si es necesario)
    frame = frame / 255.0  # Normalizar los valores de píxeles

    # Agregar una dimensión adicional para representar el lote de imágenes
    frame = np.expand_dims(frame, axis=0)

    # Realizar la predicción utilizando el modelo
    predicciones = modelo.predict(frame)

    # Obtener la clase con la mayor probabilidad
    clase_predicha = np.argmax(predicciones)

    # Obtener el nombre de la clase predicha
    nombre_clase_predicha = class_names[clase_predicha]

    # Mostrar el fotograma y la clase predicha en la ventana de visualización
    #cv2.imshow("Prediccion", frame)
    #cv2.putText(frame, nombre_clase_predicha, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    print(nombre_clase_predicha)
    #cv2.imshow("Prediccion", frame)

    # Detener el bucle cuando se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
