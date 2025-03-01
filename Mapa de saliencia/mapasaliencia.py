import cv2
import numpy as np
import matplotlib.pyplot as plt

def capturar_imagen():
    cap=cv2.VideoCapture(0) #La camara 0 es la camra principal
    if not cap.isOpened():
        print("Error: No se pudo acceder a la camara")
    print("Presiona 'Espacio' para capturas la imagen o 'Esc' para salir ")
    while True:
        ret,frame=cap.read()
        if not ret:
            print("Error al capturar la imagen")
            break
        cv2.imshow("Presiona espacio para capturar",frame)
        
        key=cv2.waitKey(1) & 0xFF
        if key==32: #Espacio para capturar
            imagen=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            break
        elif key==27: #Esc para salir
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


def calcular_mapa_intensidad(escalas):
    mapas = []
    
    # Redimensionar todos los mapas de intensidad al tamaño de la imagen original
    altura, anchura = escalas[0].shape[:2]  # Tomamos el tamaño de la imagen original
    
    for escala in escalas:
        # Convertir a flotante para evitar problemas en la división
        escala = escala.astype(np.float32)
        
        # Calcular la intensidad como la media de los canales R, G y B
        intensidad = np.mean(escala, axis=2)
        
        # Redimensionar la intensidad al tamaño de la imagen original
        intensidad = cv2.resize(intensidad, (anchura, altura), interpolation=cv2.INTER_LINEAR)
        
        # Evitar divisiones por cero en la normalización
        min_val = np.min(intensidad)
        max_val = np.max(intensidad)
        
        if max_val > min_val:  # Solo normalizar si hay diferencia
            intensidad = (intensidad - min_val) / (max_val - min_val)
        else:
            intensidad = np.zeros_like(intensidad)  # Si la imagen es uniforme, asignar ceros
        
        mapas.append(intensidad)
    
    # Promediar los mapas de intensidad de todas las escalas
    mapa_final = sum(mapas) / len(mapas)
    
    return mapa_final



def calcular_mapa_color(escalas):
    mapas_color = []
    for imagen in escalas:
        imagen = imagen.astype(float) / 255.0  # Se normaliza entre 0 y 1

        # Separamos los canales de color RGB
        r, g, b = cv2.split(imagen)

        # Calcular R, G, B, Y con las fórmulas
        R = r - ((g + b) / 2)
        G = g - ((r + b) / 2)
        B = b - ((r + g) / 2)
        Y = ((r + g) / 2) - ((abs(r - g)) / 2) - (b)

        # Normalizamos los valores a rango [0, 1]
        R = (R - np.min(R)) / (np.max(R) - np.min(R))
        G = (G - np.min(G)) / (np.max(G) - np.min(G))
        B = (B - np.min(B)) / (np.max(B) - np.min(B))
        Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

        # Combinar los mapas en una imagen
        mapa_color = np.stack((R, G, B), axis=-1)  # Concatenamos en el eje de canales
        mapas_color.append(mapa_color)  # Guardamos el mapa de color

    # Superponer las escalas (across-scale)
    # Inicializamos la imagen combinada con la primera escala
    mapa_completo = mapas_color[0].copy()

    # Sumamos las demás escalas
    for i in range(1, len(mapas_color)):
        mapa_completo += mapas_color[i]  # Sumamos directamente

    # Normalizamos el resultado para que los valores estén en [0, 1]
    mapa_completo = (mapa_completo - np.min(mapa_completo)) / (np.max(mapa_completo) - np.min(mapa_completo))
    mapa_completo_uint8 = np.uint8(mapa_completo * 255)  # Convertir a uint8
    gris = cv2.cvtColor(mapa_completo_uint8, cv2.COLOR_RGB2GRAY)  # Convertir a grises

    # Mostrar la imagen combinada en color y en escala de grises
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(mapa_completo)
    plt.title("Escalas combinadas (Color)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gris, cmap='gray')
    plt.title("Escalas combinadas (Grises)")
    plt.axis('off')

    plt.show()

    return mapas_color, mapa_completo, gris




def calcular_mapa_orientacion(escalas):
    kernel_size = 31
    sigmas = [3, 5, 7]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    mapas = []
    
    for escala in escalas:
        respuestas = np.zeros_like(escala[:, :, 0], dtype=np.float32)
        for theta in thetas:
            kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigmas[0], theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            respuesta = cv2.filter2D(escala[:, :, 0], cv2.CV_32F, kernel)
            respuestas += np.abs(respuesta)
        mapas.append(respuestas / np.max(respuestas))
    
    mapa_final = sum(mapas) / len(mapas)
    return mapa_final



def calcular_mapa_saliencia(mapa_intensidad, mapa_color, mapa_orientacion):
    mapa_final = (mapa_intensidad + mapa_color + mapa_orientacion) 
    return mapa_final


def visualizar_mapas(mapas, titulos):
    plt.figure(figsize=(10, 5))
    for i, (mapa, titulo) in enumerate(zip(mapas, titulos)):
        plt.subplot(1, len(mapas), i+1)
        plt.imshow(mapa, cmap='jet')
        plt.title(titulo)
        plt.axis('off')
    plt.show()

"""""
if __name__ == "__main__":
    imagen=capturar_imagen()
    if imagen is not None:
        escalas=redimensionar_imagen(imagen)
        visualizar_escalas(escalas)
        mapas_color=calcular_mapa_color(escalas)
"""