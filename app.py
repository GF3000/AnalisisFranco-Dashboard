import streamlit as st
from analysis import competition_analysis, teams_analysis, team_comparison, download_competition

# Título de la aplicación
st.set_page_config(
    page_title="Guille Franco - Datos & Balonmano",
    page_icon=":bar_chart:",
    layout="wide",
)

# Creación de pestañas
# tabs = ["Análisis de Competición", "Análisis de Equipos", "Comparativa de Equipos"]
tabs = ["Presentación", "Análisis de Competición", "Análisis de Equipos", "Descargar Competición"]
selected_tab = st.sidebar.radio("Selecciona una pestaña:", tabs)


def show_pagina_presentacion():
    # Título principal
    st.title("Bienvenido al Dashboard Guille Franco")

    # Introducción
    st.markdown("""
    ### ¿Qué encontrarás aquí?
    Con esta aplicación podrás explorar y analizar datos de balonmano de una manera interactiva y visual. Los datos son extraídos de forma automática de las actas oficiales publicadas por la RFEBM.
                
    La página se divide en 3 diferentes pestañas:
    - **Análisis de una competición**: Podrás ver un listado de los partidos, el flujo de los resultados, información sobre las exclusiones e información sobre los jugadores.
    - **Análisis de equipos**: Podrás ver un resumen del equipo, cómo se ven influenciados según el lugar de juego, información sobre las exclusiones y tiempos muertos, la evolución por parciales y datos sobre sus jugadores.
    - **Descargar competición**: Podrás descargar los datos de la competición en formato Excel.
                
    Más funcionalidades serán añadidas en futuras actualizaciones.
                

    ### Sobre el desarrollador
    La aplicación ha sido desarrollada por **Guille Franco** como parte de un proyecto enfocado en la exploración interactiva de datos de balonmano.

    Para más información, puedes visitar mi [página personal](https://guillermofranco.notion.site/Guille-Franco-Datos-Balonmano-b6e68f2b46ba461886b311e0cba46dbe)  

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
elif selected_tab == "Descargar Competición":
    download_competition.run_download_competition()
elif selected_tab == "Comparativa de Equipos":
    team_comparison.run_comparison()  # Función que ejecuta la comparativa de equipos
