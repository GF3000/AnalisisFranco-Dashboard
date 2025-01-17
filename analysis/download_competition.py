import streamlit as st
import os
import base64
from db_manager import DataBase

def run_download_competition():
    st.header("Descargar Competición")

    try:
        db = DataBase.load('db.pkl') 
    except FileNotFoundError:
        st.error("No se ha encontrado la base de datos 'db.pkl'.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {e}")
        st.stop()


    nombres = [competicion.nombre for competicion in db.competiciones]

    if not nombres:
        st.error("No hay archivos disponibles en el directorio 'data/equipos/'.")
        st.stop()


    # Crear un selector para elegir un archivo
    name = st.selectbox("Selecciona la competición a descargar:", nombres)

    competicion = db.get_competicion(name)

    if not competicion:
        st.error("No se ha encontrado la competición seleccionada.")
        st.stop()



    selected_file = competicion.archivo
    temporada = competicion.temporada

    st.info(f"Temporada: {temporada}")



    

    # Ruta completa del archivo seleccionado
    file_path = os.path.join("data/partidos/", selected_file)

    # Leer el archivo en modo binario
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Agregar el botón de descarga
    st.download_button(
        label=f"Descargar {selected_file}",
        data=file_bytes,
        file_name=selected_file,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
