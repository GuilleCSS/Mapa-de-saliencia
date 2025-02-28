import cv2
import numpy as np
import matplotlib.pyplot as plt

def cargar_imagen(ruta):
    imagen = cv2.imread(ruta)
    if imagen is None:
        raise FileNotFoundError(f"Error: No se pudo cargar la imagen. Verifica la ruta: {ruta}")
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

def redimensionar_imagen(imagen):
    escalas = [imagen,
               cv2.resize(imagen, (imagen.shape[1] // 2, imagen.shape[0] // 2)),
               cv2.resize(imagen, (imagen.shape[1] // 4, imagen.shape[0] // 4)),
               cv2.resize(imagen, (imagen.shape[1] // 8, imagen.shape[0] // 8))]
    # Ajustamos todas las escalas al tamaño de la menor imagen
    min_height = min([escala.shape[0] for escala in escalas])
    min_width = min([escala.shape[1] for escala in escalas])
    escalas = [cv2.resize(escala, (min_width, min_height)) for escala in escalas]
    return escalas

def calcular_mapa_intensidad(escalas):
    mapas = [np.mean(escala, axis=2) / 255.0 for escala in escalas]
    mapa_final = sum(mapas) / len(mapas)
    return mapa_final

def calcular_mapa_color(escalas):
    mapas_rg = [(escala[:, :, 0] - escala[:, :, 1]) / 255.0 for escala in escalas]
    mapas_by = [((escala[:, :, 2] - (escala[:, :, 0] + escala[:, :, 1]) / 2)) / 255.0 for escala in escalas]
    mapa_final = (sum(mapas_rg) + sum(mapas_by)) / (2 * len(escalas))
    return mapa_final

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

if __name__ == "__main__":
    ruta_imagen = "C:/Users/angel/OneDrive/Documents/Inteligencia Artificial/QUINTO SEMESTRE/VISION ARTIFICIAL/PRIMER PARCIAL/Mapa de saliencia/perro.jpg"

    imagen = cargar_imagen(ruta_imagen)
    escalas = redimensionar_imagen(imagen)
    
    mapa_intensidad = calcular_mapa_intensidad(escalas)
    mapa_color = calcular_mapa_color(escalas)
    mapa_orientacion = calcular_mapa_orientacion(escalas)
    mapa_saliencia = calcular_mapa_saliencia(mapa_intensidad, mapa_color, mapa_orientacion)
    
    visualizar_mapas([mapa_intensidad, mapa_color, mapa_orientacion, mapa_saliencia], 
                     ["Mapa de Intensidad", "Mapa de Color", "Mapa de Orientación", "Mapa de Saliencia"])
