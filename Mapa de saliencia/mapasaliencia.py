import cv2
import numpy as np
import matplotlib.pyplot as plt

def capturar_imagen():
    cap = cv2.VideoCapture(0)  # La cámara 0 es la cámara principal
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
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

def redimensionar_manual(img, factor):
    filas, columnas = img.shape[:2]  # Obtener las dimensiones originales
    new_filas, new_columnas = filas // factor, columnas // factor  # Calcular las nuevas dimensiones
    
    # Crear una nueva imagen de las dimensiones reducidas
    img_reducida = np.zeros((new_filas, new_columnas, img.shape[2]), dtype=img.dtype)
    
    # Realizar el escalamiento tomando píxeles con el factor de submuestreo
    for i in range(new_filas):
        for j in range(new_columnas):
            # Tomar los píxeles correspondientes con el factor de escala
            img_reducida[i, j] = img[i * factor, j * factor]
    
    return img_reducida

def generar_escalas(imagen):
    escalas = [imagen]
    factores = [2, 4, 8]
    for factor in factores:
        escalas.append(redimensionar_manual(imagen, factor))
    return escalas

def mostrar_escalas(escalas):
    for i, escala in enumerate(escalas[1:], start=1):  # Omitir la primera escala (imagen original)
        # Convertir la imagen escalada a BGR para mostrarla con OpenCV
        escala_bgr = cv2.cvtColor(escala, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Escala {i + 1}", escala_bgr)  # Mostrar desde Escala 2

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
            print("Advertencia: Las dimensiones del mapa de intensidad son demasiado pequeñas para redimensionar.")
            intensidad = np.zeros_like(intensidad)  # Asignar cero si no se puede redimensionar
        else:
            # Evitar que el paso del slice sea cero
            step_fila = max(intensidad.shape[0] // altura, 1)
            step_columna = max(intensidad.shape[1] // anchura, 1)
            
            # Redimensionar la intensidad al tamaño de la imagen original (Interpolación simple)
            intensidad = intensidad[::step_fila, ::step_columna]
        
        # Evitar divisiones por cero en la normalización
        min_val = np.min(intensidad)
        max_val = np.max(intensidad)
        
        if max_val > min_val:  # Solo normalizar si hay diferencia
            intensidad = (intensidad - min_val) / (max_val - min_val)
        else:
            intensidad = np.zeros_like(intensidad)  # Si la imagen es uniforme, asignar ceros
        
        # Redimensionar intensidad a las dimensiones originales para que todas tengan el mismo tamaño
        intensidad = cv2.resize(intensidad, (anchura, altura), interpolation=cv2.INTER_LINEAR)
        
        mapas.append(intensidad)
    
    # Asegurarse de que haya exactamente 4 escalas
    if len(mapas) != 4:
        raise ValueError("Debe haber exactamente 4 escalas para calcular el mapa de intensidad.")
    
    # Promediar los mapas de intensidad de todas las escalas
    mapa_final = sum(mapas) / len(mapas)
    
    return mapa_final

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
        R = r - ((g + b) / 2)
        G = g - ((r + b) / 2)
        B = b - ((r + g) / 2)
        Y = ((r + g) / 2) - ((np.abs(r - g)) / 2) - b
        
        # Normalizar los mapas R, G, B, Y al rango [0, 1]
        R = (R - np.min(R)) / (np.max(R) - np.min(R)) if np.max(R) != np.min(R) else np.zeros_like(R)
        G = (G - np.min(G)) / (np.max(G) - np.min(G)) if np.max(G) != np.min(G) else np.zeros_like(G)
        B = (B - np.min(B)) / (np.max(B) - np.min(B)) if np.max(B) != np.min(B) else np.zeros_like(B)
        Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y)) if np.max(Y) != np.min(Y) else np.zeros_like(Y)
        
        # Combinar los mapas R, G, B, Y linealmente (promedio)
        mapa_color = (R + G + B + Y) / 4.0
        
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

def mapa_orientacion(img, escalas, ksize=31, sigma=5, lambd=10, gamma=0.5):
    def convolucion_manual(img, kernel):
        filas, columnas = img.shape  # Ahora img es 2D (escala de grises)
        k_filas, k_columnas = kernel.shape
        pad_filas, pad_columnas = k_filas // 2, k_columnas // 2
        
        img_padded = np.pad(img, ((pad_filas, pad_filas), (pad_columnas, pad_columnas)), mode='constant')
        resultado = np.zeros_like(img, dtype=np.float32)
        
        for i in range(filas):
            for j in range(columnas):
                region = img_padded[i:i+k_filas, j:j+k_columnas]
                resultado[i, j] = np.sum(region * kernel)
        
        return resultado

    def normalizar_manual(img):
        min_val = np.min(img)
        max_val = np.max(img)
        
        if max_val - min_val == 0:
            return np.zeros_like(img, dtype=np.uint8)
        
        img_norm = (img - min_val) * (255.0 / (max_val - min_val))
        return img_norm.astype(np.uint8)

    direcciones = [0, 45, 90, 135]
    respuestas_por_escala = []
    
    for escala in escalas:
        # Convertir la imagen a escala de grises
        escala_gris = cv2.cvtColor(escala, cv2.COLOR_RGB2GRAY)
        respuestas = []
        for theta in direcciones:
            theta_rad = np.deg2rad(theta)
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta_rad, lambd, gamma, 0, ktype=cv2.CV_32F)
            respuesta = convolucion_manual(escala_gris, kernel)
            respuestas.append(np.abs(respuesta))  # Calcular la magnitud de la respuesta
        respuestas_por_escala.append(respuestas)
    
    mapa_orientacion = np.zeros_like(escalas[0][:, :, 0], dtype=np.float32)  # Usar solo una capa para el mapa final
    for idx, respuestas in enumerate(respuestas_por_escala):
        suma_respuestas = sum(respuestas)  # Sumar las respuestas de los filtros
        escala_ampliada = np.zeros_like(escalas[0][:, :, 0], dtype=np.float32)
        factor = 2 ** idx
        for i in range(suma_respuestas.shape[0]):
            for j in range(suma_respuestas.shape[1]):
                escala_ampliada[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor] = suma_respuestas[i, j]
        
        mapa_orientacion += escala_ampliada
    
    # Normalizar el mapa de orientación
    mapa_orientacion = normalizar_manual(mapa_orientacion)
    
    return mapa_orientacion

def calcular_mapa_saliencia(mapa_intensidad, mapa_color, mapa_orientacion):
    # 1. Normalizar los tres mapas al rango [0, 1]
    mapa_intensidad = (mapa_intensidad - np.min(mapa_intensidad)) / (np.max(mapa_intensidad) - np.min(mapa_intensidad))
    mapa_color = (mapa_color - np.min(mapa_color)) / (np.max(mapa_color) - np.min(mapa_color))
    mapa_orientacion = (mapa_orientacion - np.min(mapa_orientacion)) / (np.max(mapa_orientacion) - np.min(mapa_orientacion))
    
    # 2. Sumar los mapas ponderando cada característica de manera equitativa
    mapa_saliencia = (0.8*mapa_intensidad + 0.1*mapa_color + 0.8*mapa_orientacion) / 3.0
    
    # 3. Normalizar el mapa de saliencia final al rango [0, 1]
    #mapa_saliencia = (mapa_saliencia - np.min(mapa_saliencia)) / (np.max(mapa_saliencia) - np.min(mapa_saliencia))
    
    # Convertir el mapa de saliencia a uint8 para visualización
    mapa_saliencia_uint8 = np.uint8(mapa_saliencia * 255)
    
    return mapa_saliencia_uint8

def main():
    #imagen = capturar_imagen()
    
    # Especificar el nombre de la imagen en la misma carpeta
    nombre_imagen = "perro1.jpg"  # Cambia "tu_imagen.jpg" por el nombre de tu archivo
    
    # Cargar la imagen desde el archivo
    imagen = cv2.imread(nombre_imagen)
    
    # Verificar si la imagen se cargó correctamente
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen {nombre_imagen}.")
        return
    
    # Convertir la imagen de BGR a RGB (opcional, dependiendo de tu flujo de trabajo)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    
    escalas = generar_escalas(imagen)
    
    # Mostrar las imágenes escaladas sin filtros (omitir la primera escala)
    #mostrar_escalas(escalas)
    
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
    mapa_orientacion_resultado = mapa_orientacion(imagen, escalas)
    if mapa_orientacion_resultado is None:
        print("Error al calcular el mapa de orientación.")
        return
    
    # Calcular el mapa de saliencia
    mapa_saliencia = calcular_mapa_saliencia(mapa_intensidad, mapa_color, mapa_orientacion_resultado)
    
    # Mostrar todas las imágenes en ventanas separadas
    #cv2.imshow("Imagen Original", cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR))
    #cv2.imshow("Mapa de Intensidad", mapa_intensidad)
    #cv2.imshow("Mapa de Color", mapa_color)
    #cv2.imshow("Mapa de Orientacion", mapa_orientacion_resultado)
    cv2.imshow("Mapa de Saliencia", mapa_saliencia)
    
    # Esperar a que el usuario presione una tecla para cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()