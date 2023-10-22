import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Configurar la página
st.set_page_config(page_title="Aplicación de Dibujo Fourier", layout="wide")

# Título de la aplicación
st.title("Aplicación de Dibujo Fourier")

# Crear el canvas para dibujar
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Color de relleno
    stroke_width=3,                      # Ancho del pincel
    stroke_color="#000000",              # Color del pincel
    background_color="#FFFFFF",          # Color de fondo
    height=400,
    width=700,
    drawing_mode="freedraw",
    key="canvas",
)

def extraer_puntos(datos_imagen):
    filas, columnas, _ = datos_imagen.shape
    puntos = []
    for i in range(filas):
        for j in range(columnas):
            # Verificar si el píxel está dibujado (no es blanco)
            if np.any(datos_imagen[i, j, :3] != [255, 255, 255]):
                puntos.append([j, filas - i])  # Añadir el punto y corregir la coordenada y
    puntos = np.array(puntos, dtype=float)  # Convertir a float para realizar operaciones
    
    if len(puntos) == 0:
        return np.array([])  # Devolver un array vacío si no hay puntos
    
    # Normalizar los puntos para que estén centrados en 0 y en un rango de -1 a 1
    puntos -= np.mean(puntos, axis=0)
    max_range = np.max(np.abs(puntos))
    if max_range > 0:
        puntos /= max_range
    return puntos

# Función para recrear la imagen a partir de la Transformada de Fourier
def recrear_imagen(fourier_transform):
    if len(fourier_transform) == 0:
        return np.array([])
    
    # Aplicar la Transformada Inversa de Fourier
    puntos_recreados = np.fft.ifft(fourier_transform, axis=0)
    
    return puntos_recreados.real

# Función para aplicar la Transformada de Fourier
def aplicar_transformada_fourier(puntos):
    if len(puntos) == 0:
        return np.array([]), np.array([])
    
    # Aplicar la Transformada de Fourier
    fourier_transform = np.fft.fft(puntos, axis=0)
    
    # Calcular las frecuencias
    frecuencias = np.fft.fftfreq(len(puntos))
    
    return fourier_transform, frecuencias

# Mostrar el resultado del canvas
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
    puntos = extraer_puntos(canvas_result.image_data)
    
    if puntos.size > 0:  # Chequear si hay puntos dibujados
        st.write("Puntos extraídos:", puntos.size)
        df_puntos = pd.DataFrame(puntos, columns=['x', 'y'])
        st.write("Puntos dibujados:", df_puntos)
        
        # Aplicar la Transformada de Fourier
        fourier_transform, frecuencias = aplicar_transformada_fourier(puntos)
        if fourier_transform.size > 0:  # Chequear si la Transformada de Fourier se calculó correctamente
            df_fourier_real = pd.DataFrame(fourier_transform.real, columns=['x', 'y'])
            df_fourier_imag = pd.DataFrame(fourier_transform.imag, columns=['x', 'y'])
            st.write("Transformada de Fourier (Parte Real):", df_fourier_real)
            st.write("Transformada de Fourier (Parte Imaginaria):", df_fourier_imag)
            st.write("Frecuencias:", frecuencias)
        
            # Recrear la imagen
            puntos_recreados = recrear_imagen(fourier_transform)
            if puntos_recreados.size > 0:  # Chequear si los puntos fueron recreados correctamente
                df_puntos_recreados = pd.DataFrame(puntos_recreados, columns=['x', 'y'])
                st.write("Puntos recreados:", df_puntos_recreados)
                
                # Mostrar la imagen recreada
                fig, ax = plt.subplots()
                ax.plot(puntos_recreados[:, 0], -puntos_recreados[:, 1], 'b-')
                ax.axis('equal')
                st.pyplot(fig)
    else:
        st.write("Por favor, dibuja algo en el canvas.")
