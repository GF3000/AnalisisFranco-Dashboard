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
tabs = ["Presentación", "Análisis de Competición", "Análisis de Equipos"]
selected_tab = st.sidebar.radio("Selecciona una pestaña:", tabs)


def show_pagina_presentacion():
    # Título principal
    st.title("Bienvenido al Dashboard del CESA 2023 y 2024")

    # Introducción
    st.markdown("""
    ### ¿Qué encontrarás aquí?
    Esta aplicación está diseñada para visualizar y comparar datos y estadísticas de las dos últimas ediciones del **CESA** (Campeonato de España de Selecciones Autonómicas) correspondientes a los años **2023** y **2024**.

    Podrás analizar:
    - Desempeño y estadísticas de las selecciones participantes.
    - Comparativas entre equipos.
    - Visualización detallada de datos para descubrir tendencias y análisis clave.

    ### Sobre el desarrollador
    La aplicación ha sido desarrollada por **Guille Franco** como parte de un proyecto enfocado en la exploración interactiva de datos de balonmano.

    Para más información, puedes visitar mi página personal haciendo clic en el siguiente enlace:
    [Visitar página personal](https://guillermofranco.notion.site/Guille-Franco-Datos-Balonmano-b6e68f2b46ba461886b311e0cba46dbe)

    """)

    # Información adicional o enlaces
    # Información de contacto
    st.markdown("""
    ---
    **Nota**: Si tienes algún comentario o sugerencia sobre esta aplicación, no dudes en ponerte en contacto conmigo a través de los siguientes canales:
    
    - **Correo electrónico**: [guillermofrancogimeno@gmail.com](mailto:guillermofrancogimeno@gmail.com)
    - **Twitter/X**:  [@GuilleFrancoBM](https://twitter.com/GuilleFrancoBM)
    """)
    

# Mostrar el contenido dependiendo de la pestaña seleccionada
if selected_tab == "Presentación":
    show_pagina_presentacion()  # Función que ejecuta la presentación
elif selected_tab == "Análisis de Competición":
    competition_analysis.run_analysis()  # Función que ejecuta el análisis de competencia
elif selected_tab == "Análisis de Equipos":
    teams_analysis.run_analysis()  # Función que ejecuta el análisis de equipos
elif selected_tab == "Comparativa de Equipos":
    team_comparison.run_comparison()  # Función que ejecuta la comparativa de equipos
