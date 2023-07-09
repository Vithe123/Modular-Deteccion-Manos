import os
from PIL import Image

#FolderAtras = "Imagenes/Atras"
#FolderDerecha = "Imagenes/Derecha"
#FolderIzquierda = "Imagenes/Izquierda"

# Carpeta que contiene las imágenes originales
carpeta_originales = "Imagenes/Izquierda"

# Carpeta donde se guardarán las imágenes redimensionadas
carpeta_redimensionadas = "Redimencionadas/Izquierda"

# Obtener la lista de archivos en la carpeta de originales
archivos_originales = os.listdir(carpeta_originales)

# Iterar sobre los archivos originales
for archivo in archivos_originales:
    # Comprobar si el archivo es una imagen (por extensión)
    if archivo.endswith(".jpg") or archivo.endswith(".png"):
        # Ruta completa del archivo original
        ruta_original = os.path.join(carpeta_originales, archivo)

        # Abrir la imagen original
        imagen_original = Image.open(ruta_original)

        # Redimensionar la imagen
        imagen_redimensionada = imagen_original.resize((320, 240))

        # Generar el nuevo nombre de archivo
        nuevo_nombre = archivo.split(".")[0] + "_mod.jpg"

        # Ruta completa del archivo redimensionado
        ruta_redimensionada = os.path.join(carpeta_redimensionadas, nuevo_nombre)

        # Guardar la imagen redimensionada con el nuevo nombre
        imagen_redimensionada.save(ruta_redimensionada)