import streamlit as st
import os
import base64

def run_download_competition():
    st.header("Descargar Competición")

    # Obtener la lista de archivos en el directorio
    files = [f for f in os.listdir("data/partidos/") if f.endswith(".xlsx")]

    if not files:
        st.error("No hay archivos disponibles en el directorio 'data/partidos/'.")
        st.stop()

    # Crear un selector para elegir un archivo
    selected_file = st.selectbox("Selecciona el archivo a descargar:", files)

    # Ruta completa del archivo seleccionado
    file_path = os.path.join("data/partidos/", selected_file)

    # Leer el archivo en modo binario
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Agregar el botón de descarga
    st.download_button(
        label="Descargar archivo",
        data=file_bytes,
        file_name=selected_file,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
