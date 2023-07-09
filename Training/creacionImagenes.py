import cv2
import time

fold = "Redimencionadas"

FolderAdelante = fold + "/Adelante"
FolderAtras = fold + "/Atras"
FolderDerecha = fold + "/Derecha"
FolderIzquierda = fold + "/Izquierda"

# Función para capturar una imagen desde la cámara web
def capture_image():
    # Inicializar la captura de video desde la cámara web
    cap = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Número de veces para capturar imágenes
    num_captures = 100

    # Capturar imágenes el número de veces especificado
    capture_count = 0
    while capture_count < num_captures:
        # Leer un fotograma de la cámara web
        ret, frame = cap.read()

        # Verificar si se pudo capturar el fotograma correctamente
        if ret:
            # Generar un nombre de archivo único basado en la hora actual
            timestamp = int(time.time())
            filename = f"captura_{timestamp}.jpg"

            # Guardar la imagen en disco
            cv2.imwrite(FolderAdelante + "/" + filename, frame)
            print(f"Imagen guardada como {filename}")
        else:
            mensaje_error = "Se produjo un error debido a que hubo un problema con la camara"
            raise Exception(mensaje_error)
            cap.release()
            exit()
        capture_count += 1
        time.sleep(1)
    cap.release()

capture_image()
