#Equipo 
#Angel Miguel Sánchez Pérez
#Guillermo Carreto Sánchez
#Miguel Alejandro Flores Sotelo 

import cv2
import numpy as np

#Función para capturar imagen con la cámara de la laptop
def capturar_imagen():
    cap = cv2.VideoCapture(0)  # La cámara 0 es la cámara principal
    if not cap.isOpened():
        print("No se pudo acceder a la cámara")
        return None
    print("Presiona 'Espacio' para capturar la imagen o 'Esc' para salir")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar la imagen")
            break
        cv2.imshow("Presiona espacio para capturar", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Espacio para capturar
            imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            break
        elif key == 27:  # Esc para salir
            cap.release()
            cv2.destroyAllWindows()
            return None
    cap.release()   
    cv2.destroyAllWindows()
    return imagen

#Función para redimensionar la imagen manualmente
def redimensionar_manual(img, factor):
    filas, columnas = img.shape[:2]
    new_filas, new_columnas = filas // factor, columnas // factor
    
    # Cambiar a interpolación por área (mejor para reducción)
    img_reducida = cv2.resize(
        img, 
        (new_columnas, new_filas), 
        interpolation=cv2.INTER_AREA  # Más efectivo para reducción
    )
    
    return img_reducida

#Función para obtener las escalas necesarias
def generar_escalas(imagen):
    escalas = [imagen]
    factores = [2,4,8]
    for factor in factores:
        escalas.append(redimensionar_manual(imagen, factor))
    return escalas

#Función para mostrar las escalas en pantalla
def mostrar_escalas(escalas):
    for i, escala in enumerate(escalas[1:], start=1):  # Omitir la primera escala (imagen original)
        # Convertir la imagen escalada a BGR para mostrarla con OpenCV
        escala_bgr = cv2.cvtColor(escala, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Escala {i + 1}", escala_bgr)  # Mostrar desde Escala 2

#Función para calcular el mapa de intensidad
def calcular_mapa_intensidad(escalas):
    mapas = []
    
    # Tomamos el tamaño de la imagen original
    altura, anchura = escalas[0].shape[:2]  # Imagen original
    
    for escala in escalas:
        # Convertir a flotante para evitar problemas en la división
        escala = escala.astype(np.float32)
        
        # Calcular la intensidad como la media de los canales R, G y B
        intensidad = np.mean(escala, axis=2)
        
        intensidad = (intensidad - np.min(intensidad)) / (np.max(intensidad) - np.min(intensidad))
        
        # Verificar si el tamaño es lo suficientemente grande para redimensionar
        if intensidad.shape[0] <= 1 or intensidad.shape[1] <= 1:
            print("Las dimensiones del mapa de intensidad son demasiado pequeñas para redimensionar")
            intensidad = np.zeros_like(intensidad)  # Asignar cero si no se puede redimensionar
        else:
            # Evitar que el paso del slice sea cero
            step_fila = max(intensidad.shape[0] // altura, 1)
            step_columna = max(intensidad.shape[1] // anchura, 1)
            
            # Redimensionar la intensidad al tamaño de la imagen original (aqui usamos una interpolación simple)
            intensidad = intensidad[::step_fila, ::step_columna]
        
        # Evitar divisiones por cero en la normalización
        min_val = np.min(intensidad)
        max_val = np.max(intensidad)
        
        if max_val > min_val:  # Solo normalizar si hay diferencia
            intensidad = (intensidad - min_val) / (max_val - min_val)
        else:
            intensidad = np.power(intensidad, 0.8)  # Si la imagen es uniforme, asignar ceros
        
        # Redimensionar intensidad a las dimensiones originales para que todas tengan el mismo tamaño
        intensidad = cv2.resize(intensidad, (anchura, altura), interpolation=cv2.INTER_LINEAR)
        
        mapas.append(intensidad)
    
    # Promediar los mapas de intensidad de todas las escalas
    mapa_final = sum(mapas) / len(mapas)
    
    return mapa_final

#Función para calcular el mapa de color
def calcular_mapa_color(escalas):
    mapas_color = []
    
    # Obtener el tamaño de la imagen original (la primera escala)
    altura, anchura = escalas[0].shape[:2]
    
    for imagen in escalas:
        # Convertir la imagen a flotante y normalizar entre 0 y 1
        imagen = imagen.astype(float) / 255.0
        
        # Separar los canales de color RGB
        r = imagen[:, :, 0]
        g = imagen[:, :, 1]
        b = imagen[:, :, 2]
        
        # Calcular R, G, B, Y usando las fórmulas proporcionadas
        R = r - 0.7*g - 0.7*b  # Mayor peso a rojo
        G = g - 0.7*r - 0.7*b  # Mayor peso a verde
        B = b - 0.7*r - 0.7*g  # Mayor peso a azul
        Y = 0.5*(r + g) - 0.5*np.abs(r - g) - 0.7*b 
        
        # Normalizar los mapas R, G, B, Y al rango [0, 1]
        R = (R - np.min(R)) / (np.max(R) - np.min(R)) if np.max(R) != np.min(R) else np.zeros_like(R)
        G = (G - np.min(G)) / (np.max(G) - np.min(G)) if np.max(G) != np.min(G) else np.zeros_like(G)
        B = (B - np.min(B)) / (np.max(B) - np.min(B)) if np.max(B) != np.min(B) else np.zeros_like(B)
        Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y)) if np.max(Y) != np.min(Y) else np.zeros_like(Y)
        
        # Combinar los mapas R, G, B, Y linealmente (promedio)
        mapa_color = np.maximum(np.maximum(R, G), np.maximum(B, Y))
        
        # Redimensionar el mapa de color al tamaño de la imagen original
        mapa_color = cv2.resize(mapa_color, (anchura, altura), interpolation=cv2.INTER_LINEAR)
        
        mapas_color.append(mapa_color)
    
    # Combinar linealmente los mapas de color de todas las escalas
    mapa_completo = np.zeros_like(mapas_color[0])
    for mapa in mapas_color:
        mapa_completo += mapa
    mapa_completo /= len(mapas_color)  # Promediar los mapas
    
    # Normalizar el mapa completo al rango [0, 1]
    mapa_completo = (mapa_completo - np.min(mapa_completo)) / (np.max(mapa_completo) - np.min(mapa_completo))
    
    # Convertir el mapa completo a uint8 para visualización
    mapa_completo_uint8 = np.uint8(mapa_completo * 255)
    
    return mapa_completo_uint8

#Función para calcular el mapa de orientación
def mapa_orientacion(escalas, tamañokernel=45, sigma=9, lambd=15, gamma=0.25):
  
    def convolucion_manual(img, kernel):
        
        filas, columnas = img.shape  # Dimensiones de la imagen (2D en escala de grises)
        k_filas, k_columnas = kernel.shape  # Dimensiones del kernel
        pad_filas, pad_columnas = k_filas // 2, k_columnas // 2  # Tamaño del padding
        
        # Se rellena la imagen con ceros en los bordes para permitir la convolución
        img_padded = np.pad(img, ((pad_filas, pad_filas), (pad_columnas, pad_columnas)), mode='constant')
        resultado = np.zeros_like(img, dtype=np.float32)
        
        # Aplicación de la convolución recorriendo la imagen
        for i in range(filas):
            for j in range(columnas):
                region = img_padded[i:i+k_filas, j:j+k_columnas]  # Se extrae la región de interés
                resultado[i, j] = np.sum(region * kernel)  # Se realiza la multiplicación y suma
        
        return resultado

    def normalizar_manual(img):
        
        min_val = np.min(img)
        max_val = np.max(img)
        
        if max_val - min_val == 0:
            return np.zeros_like(img, dtype=np.uint8)  # Evitar división por cero
        
        img_norm = (img - min_val) * (255.0 / (max_val - min_val))  # Normalización lineal
        return img_norm.astype(np.uint8)

    # Definimos las direcciones (0°, 45°, 90° y 135°)
    direcciones = [0, 45, 90, 135]
    respuestas_por_escala = []

    for escala in escalas:
        # Convertir la imagen a escala de grises
        escala_gris = cv2.cvtColor(escala, cv2.COLOR_RGB2GRAY)
        respuestas = []
        respuestas_x = []
        respuestas_y = []
        
        # Aplicamos los filtros de Gabor en las direcciones definidas
        for theta in direcciones:
            theta_rad = np.deg2rad(theta)  # Convertimos grados a radianes
            kernel = cv2.getGaborKernel((tamañokernel, tamañokernel), sigma, theta_rad, lambd, gamma, 0, ktype=cv2.CV_32F)
            respuesta = convolucion_manual(escala_gris, kernel)  # Aplicamos convolución
            respuestas.append(respuesta)

            # Agrupamos respuestas en direcciones perpendiculares (0° y 90° / 45° y 135°)
            if theta in [0, 90]:
                respuestas_x.append(respuesta)
            else:
                respuestas_y.append(respuesta)
        
        # Calculamos la magnitud combinada de las respuestas en X y Y
        magnitud_respuesta = np.sqrt(np.square(respuestas_x[0]) + np.square(respuestas_x[1])) + \
                             np.sqrt(np.square(respuestas_y[0]) + np.square(respuestas_y[1]))

        respuestas_por_escala.append(magnitud_respuesta)  

    # Inicializamos el mapa de orientación con ceros
    mapa_orientacion = np.zeros_like(escalas[0][:, :, 0], dtype=np.float32)

    for idx, respuestas in enumerate(respuestas_por_escala):
        suma_respuestas = respuestas  # Ya está calculada la magnitud combinada
        escala_ampliada = np.zeros_like(escalas[0][:, :, 0], dtype=np.float32)
        factor = 2 ** idx  # Factor de escalado según la escala actual
        
        # Expandimos la escala para alinearla con la imagen original
        for i in range(suma_respuestas.shape[0]):
            for j in range(suma_respuestas.shape[1]):
                escala_ampliada[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor] = suma_respuestas[i, j]

        # Sumamos la escala ampliada al mapa de orientación final
        mapa_orientacion += escala_ampliada


    mapa_orientacion = normalizar_manual(mapa_orientacion)

    return mapa_orientacion

def calcular_mapa_saliencia(mapa_intensidad, mapa_color, mapa_orientacion):
    #Normalizar los tres mapas al rango [0, 1]
    mapa_intensidad = (mapa_intensidad - np.min(mapa_intensidad)) / (np.max(mapa_intensidad) - np.min(mapa_intensidad))
    mapa_color = (mapa_color - np.min(mapa_color)) / (np.max(mapa_color) - np.min(mapa_color))
    mapa_orientacion = (mapa_orientacion - np.min(mapa_orientacion)) / (np.max(mapa_orientacion) - np.min(mapa_orientacion))
    
    #Sumar los mapas ponderando cada característica
    mapa_saliencia = (0.25*mapa_intensidad + 0.35*mapa_color + 0.4*mapa_orientacion)
    
    # Convertir a float32 para el filtro bilateral
    mapa_saliencia = mapa_saliencia.astype(np.float32)
    
    # Aplicar suavizado adaptativo
    mapa_saliencia = cv2.bilateralFilter(mapa_saliencia, 9, 75, 75)
    
    # Normalizar (no es necesario pero se obtienen mejores resultados) y convertir a uint8
    #mapa_saliencia = (mapa_saliencia - np.min(mapa_saliencia)) / (np.max(mapa_saliencia) - np.min(mapa_saliencia))
    mapa_saliencia_uint8 = np.uint8(mapa_saliencia * 255)
    
    return mapa_saliencia_uint8

def main():
    #Función para capturar imagen con la cámara de la laptop
    imagen = capturar_imagen()
    
    """""
    # Por si se quiere cargar una imagen y trabajar con ella
    nombre_imagen = "perro.jpg" 
    
    imagen = cv2.imread(nombre_imagen)
    
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen {nombre_imagen}.")
        return
    
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    """
    
    escalas = generar_escalas(imagen)
    
    # Mostrar las imágenes escaladas sin filtros
    mostrar_escalas(escalas)
    
    # Calcular los mapas de intensidad
    mapa_intensidad = calcular_mapa_intensidad(escalas)
    if mapa_intensidad is None:
        print("Error al calcular el mapa de intensidad.")
        return
    
    # Calcular el mapa de color
    mapa_color_resultado = calcular_mapa_color(escalas)
    if mapa_color_resultado is None:
        print("Error al calcular el mapa de color.")
        return
    mapa_color = mapa_color_resultado
    
    # Calcular el mapa de orientación
    mapa_orientacion_resultado = mapa_orientacion(escalas)
    if mapa_orientacion_resultado is None:
        print("Error al calcular el mapa de orientación.")
        return
    
    # Calcular el mapa de saliencia
    mapa_saliencia = calcular_mapa_saliencia(mapa_intensidad, mapa_color, mapa_orientacion_resultado)
    
    # Mostrar todas las imágenes en ventanas separadas
    cv2.imshow("Imagen Original", cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR))
    cv2.imshow("Mapa de Intensidad", mapa_intensidad)
    cv2.imshow("Mapa de Color", mapa_color)
    cv2.imshow("Mapa de Orientacion", mapa_orientacion_resultado)
    cv2.imshow("Mapa de Saliencia", mapa_saliencia)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()