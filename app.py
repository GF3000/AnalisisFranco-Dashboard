import streamlit as st
from analysis import competition_analysis, teams_analysis, team_comparison

# Título de la aplicación
st.set_page_config(
    page_title="Dashboard de Análisis Franco",
    page_icon=":bar_chart:",
    layout="wide",
)

# Creación de pestañas
# tabs = ["Análisis de Competición", "Análisis de Equipos", "Comparativa de Equipos"]
tabs = ["Análisis de Competición", "Análisis de Equipos"]
selected_tab = st.sidebar.radio("Selecciona una pestaña:", tabs)

# Mostrar el contenido dependiendo de la pestaña seleccionada
if selected_tab == "Análisis de Competición":
    competition_analysis.run_analysis()  # Función que ejecuta el análisis de competencia
elif selected_tab == "Análisis de Equipos":
    teams_analysis.run_analysis()  # Función que ejecuta el análisis de equipos
elif selected_tab == "Comparativa de Equipos":
    team_comparison.run_comparison()  # Función que ejecuta la comparativa de equipos
