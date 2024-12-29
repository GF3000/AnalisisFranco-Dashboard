import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go


def run_comparison():
    
    # subheader
    st.subheader("Comparativa de Equipos")
    # Obtener los archivos disponibles
    files = [f for f in os.listdir("data/equipos/") if f.endswith(".xlsx")]
    if not files:
        st.error("No hay archivos disponibles en el directorio 'data/equipos/'.")
        st.stop()

    # Seleccionar archivos a procesar
    selected_files = st.multiselect("Selecciona los archivos:", files)

    if not selected_files:
        st.warning("Selecciona al menos un archivo para continuar.")
        st.stop()


    # Procesar archivos seleccionados
    equipos_in_files = []
    for f in selected_files:
        df = pd.read_excel(f"data/equipos/{f}")
        equipos_in_files.extend([f"{name} ({f})" for name in df["nombre"].unique()])
    
    # Opciones de equipos
    equipos_in_files = list(set(equipos_in_files))
    equipos_in_files.sort()

    equipos_in_files.insert(0, "Seleccionar todos")

    # Seleccionar equipos con multiselect
    selected_equipos = st.multiselect(
        "Selecciona equipos:", 
        options=equipos_in_files,
        default=[]  # Opcional: puedes definir equipos seleccionados por defecto
    )

    if "Seleccionar todos" in selected_equipos:
        selected_equipos = equipos_in_files[1:]

    # Cargar datos de equipos seleccionados
    equipos_selected = []
    for selected_equipo in selected_equipos:
        file = selected_equipo.split(" (")[1][:-1]
        equipo = selected_equipo.split(" (")[0]
        df = pd.read_excel(f"data/equipos/{file}")
        equipo_df = df[df["nombre"] == equipo]
        equipo_df["nombre"] = equipo + " (" + file.split("_")[1].split(".")[0] + ")"
        equipos_selected.append(equipo_df)

    # Unir los dataframes
    if equipos_selected:  # Verificar si hay datos seleccionados
        equipo_df_selected = pd.concat(equipos_selected)
    else:
        st.warning("Por favor, selecciona al menos un equipo.")
        st.stop()

    st.divider()

    # Mostrar tabla de datos
    st.write("Datos seleccionados:", equipo_df_selected)

    # Reescalar tamaño de los puntos
    factor_escala = 15
    equipo_df_selected["puntos_escalados"] = np.interp(
        equipo_df_selected["puntos"],
        (equipo_df_selected["puntos"].min(), equipo_df_selected["puntos"].max()),
        (1, factor_escala)
    )

    # Gráfico de Goles Totales vs Diferencia de Goles
    st.subheader("Goles Totales vs Diferencia de Goles")


    # Crear scatter plot
    fig = px.scatter(
        equipo_df_selected,
        x="diferencia_goles",
        y="goles_totales",
        size="puntos_escalados",
        color="tasa de rendimiento",
        hover_data={"nombre": True, "puntos": True, "puntos_escalados": False},
        size_max=50
    )

    # Agregar correlación
    correlation = equipo_df_selected["diferencia_goles"].corr(equipo_df_selected["goles_totales"])
    fig.update_layout(
        title=f"Goles Totales vs. Diferencia de Goles<br><sup>Correlación: {correlation:.2f}</sup>",
    )

    # modify x-axis and y-axis labels
    fig.update_xaxes(title_text="Diferencia de goles")
    fig.update_yaxes(title_text="Goles totales")


    # Mostrar gráfico
    st.plotly_chart(fig)

    # Gráfico de Tasa de rendimiento vs Puntos
    st.subheader("Tasa de rendimiento vs Puntos")

    # Crear scatter plot
    fig2 = px.scatter(
        equipo_df_selected,
        x="tasa de rendimiento",
        y="puntos",
        hover_data={"nombre": True, "puntos": True, "tasa de rendimiento": True, "puntos_escalados": False},
        size="puntos_escalados",
        color="tasa de rendimiento",
        size_max=50
    )

    # Calcular la correlación
    correlation = equipo_df_selected["tasa de rendimiento"].corr(equipo_df_selected["puntos"])

    # Actualizar el diseño del gráfico
    fig2.update_layout(
        title=f"Tasa de rendimiento vs. Puntos<br><sup>Correlación: {correlation:.2f}</sup>",
    )

    fig2.update_xaxes(title_text="Tasa de rendimiento")
    fig2.update_yaxes(title_text="Puntos")

    # Mostrar el gráfico en la dashboard
    st.plotly_chart(fig2)

        # Gráfico de Diferencia de Goles vs Puntos
    st.subheader("Diferencia de goles vs Puntos")

    # Crear scatter plot
    fig3 = px.scatter(
        equipo_df_selected,
        x="diferencia_goles",
        y="puntos",
        hover_data={"nombre": True, "puntos": True, "diferencia_goles": True, "puntos_escalados": False},
        size="puntos_escalados",
        color="tasa de rendimiento",
        size_max=50
    )

    # Calcular la correlación
    correlation = equipo_df_selected["diferencia_goles"].corr(equipo_df_selected["puntos"])

    # Actualizar el diseño del gráfico
    fig3.update_layout(
        title=f"Diferencia de goles vs. Puntos<br><sup>Correlación: {correlation:.2f}</sup>",
    )

    fig3.update_xaxes(title_text="Diferencia de goles")
    fig3.update_yaxes(title_text="Puntos")


    # Mostrar el gráfico en la dashboard
    st.plotly_chart(fig3)

    # Gráfico de Goles Totales por Parcial
    st.subheader("Goles Totales por Parcial")

    # Crear figura
    fig4 = go.Figure()

    # Obtener equipos únicos
    equipos = equipo_df_selected["nombre"].tolist()

    # Agregar trazas para cada equipo
    for equipo in equipos:
        goles_totales_equipo = []
        for i in range(1, 13):
            goles_totales_equipo.append(equipo_df_selected[equipo_df_selected["nombre"] == equipo][f"goles_totales_parcial_{i}"].values[0])
        
        fig4.add_trace(go.Scatter(
            x=[i for i in range(1, 13)],
            y=goles_totales_equipo,
            mode='lines+markers',
            name=equipo,
            hoverinfo="name+y",
        ))

    # Configurar el diseño
    fig4.update_layout(
        title_text="Goles totales por parcial",
        xaxis_title="Parcial",
        yaxis_title="Goles totales",
        yaxis=dict(tickmode='linear', tick0=0, dtick=1),
        showlegend=True,
    )

    # Agregar ticks al eje x
    fig4.update_xaxes(tickvals=[i for i in range(1, 13)])

    # Calcular el máximo de goles en todos los parciales
    goles_max = equipo_df_selected[[f"goles_totales_parcial_{i}" for i in range(1, 13)]].max().max()

    # Agregar ticks al eje Y para cada unidad desde 0 hasta el máximo de goles (redondeado hacia arriba)
    fig4.update_yaxes(
        tickvals=[i for i in range(0, int(goles_max) + 2)],  # Desde 0 hasta goles_max + 1
        tickmode='array'  # Especificar que usamos valores explícitos para los ticks
    )

    # Mostrar el gráfico en la dashboard
    st.plotly_chart(fig4)

    # Gráfico de Diferencia de Goles por Parcial
    st.subheader("Diferencia de Goles por Parcial")

    # Crear figura
    fig5 = go.Figure()

    # Agregar trazas para cada equipo
    for equipo in equipo_df_selected["nombre"]:
        diferencia_equipo = []
        for i in range(1, 13):
            diferencia_equipo.append(equipo_df_selected[equipo_df_selected["nombre"] == equipo][f"diferencia_parcial_{i}"].values[0])
        
        fig5.add_trace(go.Scatter(
            x=[i for i in range(1, 13)],
            y=diferencia_equipo,
            mode='lines+markers',
            name=equipo,
            hoverinfo="name+y",
        ))

    # Configurar el diseño
    fig5.update_layout(
        title_text="Diferencia de goles por parcial",
        xaxis_title="Parcial",
        yaxis_title="Diferencia de goles",
        yaxis=dict(tickmode='linear', tick0=-5, dtick=1),
        showlegend=True,
    )

    # Agregar ticks al eje x
    fig5.update_xaxes(tickvals=[i for i in range(1, 13)])

    # Mostrar el gráfico en la dashboard
    st.plotly_chart(fig5)

    # Gráfico de Variación en la Diferencia de Goles por Parcial
    st.subheader("Variación en la Diferencia de Goles por Parcial")

    # Crear etiquetas para el eje x indicando los parciales involucrados
    etiquetas_x = [f"{i}-{i+1}" for i in range(0, 12)]

    # Crear figura
    fig6 = go.Figure()

    # Agregar trazas para cada equipo
    for equipo in equipo_df_selected["nombre"]:
        # Obtener las diferencias de goles por parcial
        diferencia_equipo = [
            equipo_df_selected[equipo_df_selected["nombre"] == equipo][f"diferencia_parcial_{i}"].values[0] 
            for i in range(1, 13)
        ]
        # Calcular la variación entre parciales (derivada discreta)
        variacion_diferencia = np.diff(diferencia_equipo)
        # Añadir el primer valor de la diferencia de goles
        variacion_diferencia = np.insert(variacion_diferencia, 0, diferencia_equipo[0])

        # Agregar la traza para cada equipo
        fig6.add_trace(go.Scatter(
            x=list(range(0, 12)),  # Índices de las variaciones (1 a 11)
            y=variacion_diferencia,
            mode='lines+markers',
            name=equipo,
            hoverinfo="name+y",
        ))

    # Actualizar el diseño del gráfico
    fig6.update_layout(
        title_text="Variación en la diferencia de goles por parcial", 
        xaxis_title="Parcial",
        yaxis_title="Variación en la diferencia de goles",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 12)),  # Posiciones de las marcas
            ticktext=etiquetas_x,         # Etiquetas descriptivas
            title="Parciales consecutivos",
        ),
        yaxis=dict(tickmode='linear', tick0=-2, dtick=1),
        title_x=0.5
    )

    # Mostrar el gráfico en la dashboard
    st.plotly_chart(fig6)






