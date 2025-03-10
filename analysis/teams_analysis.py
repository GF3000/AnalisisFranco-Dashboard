import streamlit as st
import pandas as pd
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2_contingency
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp, pearsonr, spearmanr, linregress
from db_manager import DataBase

num_parciales = 12

def run_analysis():
    st.header("Análisis de los Partidos")


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

    st.write("Selecciona una competición a analizar:")

    selected_competition = st.selectbox("Competición:", nombres)

    if not selected_competition:
        st.warning("Selecciona una competición para continuar.")
        st.stop()

    competicion = db.get_competicion(selected_competition)
    if competicion.infantil:
        st.write("Competición Infantil")
        global num_parciales
        num_parciales = 10
    else:
        num_parciales = 12

    temporada = competicion.temporada

    st.info(f"Temporada: {temporada}")

    




    # Cargamos el df de partidos
    df = pd.read_excel(f"data/partidos/{competicion.archivo}")

    # Seleccionamos equipo
    todos_equipos = [df["nombre_local"].unique(), df["nombre_visitante"].unique()]
    todos_equipos = set(np.concatenate(todos_equipos))
    selected_team = st.selectbox("Equipo:", todos_equipos)

    if not selected_team:
        st.warning("Selecciona un equipo para continuar.")
        st.stop()

    numero_partidos = len(df[(df['nombre_local'] == selected_team) | (df['nombre_visitante'] == selected_team)])
    st.write("Número de partidos jugados: ", numero_partidos)

    st.divider()

    tabs = st.tabs(["Resumen", "Influencia del lugar de juego", "Exclusiones", "Evolución", "Rachas", "Tiempo muertos", "Jugadores"])

    with tabs[0]:
        st.subheader("Resumen del Equipo")
        st.write(f"Resumen de los partidos del equipo {selected_team}:")
        st.write(f"Partidos jugados: {len(df[(df['nombre_local'] == selected_team) | (df['nombre_visitante'] == selected_team)])}")

        filtered_matches = df[(df["nombre_local"] == selected_team) | (df["nombre_visitante"] == selected_team)]


        all_matches = crear_columnas(df, selected_team)

        # Restultados en casa y fuera de casa
        # Create a new column to classify the match result
        all_matches['resultado'] = all_matches['diferencia_goles'].apply(lambda x: 'Victoria' if x > 0 else ('Derrota' if x < 0 else 'Empate'))

        # Create a pivot table to summarize the results
        summary_table = all_matches.pivot_table(index='my_team_is', columns='resultado', aggfunc='size', fill_value=0)

        # Ensure that 'Victoria', 'Empate', and 'Derrota' columns exist
        for col in ['Victoria', 'Empate', 'Derrota']:
            if col not in summary_table.columns:
                summary_table[col] = 0

        # Reorder the columns 
        summary_table = summary_table[['Victoria', 'Empate', 'Derrota']]

        # Rename the index to the chosen team
        summary_table.index = [f"Local" if idx == 'Local' else f'Visitante' for idx in summary_table.index]

        # Display specific columns
        st.write("Partidos:")
        st.write(all_matches[['actaid', 'fecha', 'partido', 'goles_my_team', 'goles_rival']])
        
        # Display the summary table
        st.write("Resumen de los Resultados:")
        st.write(summary_table)

        # Displat gt vs dif
        st.write("Goles totales vs Diferencia de goles:")
        fig_gt_dif = get_gt_vs_dif(all_matches, selected_team)
        st.plotly_chart(fig_gt_dif)


        # Mejores 3 partidos (mayor diferencia de goles)
        st.write("Mejores 3 partidos:")
        best_matches = all_matches.nlargest(3, 'diferencia_goles')
        st.write(best_matches[['actaid', 'fecha', 'partido', 'goles_my_team', 'goles_rival', 'diferencia_goles']])

        # Peores 3 partidos (menor diferencia de goles)
        st.write("Peores 3 partidos:")
        worst_matches = all_matches.nsmallest(3, 'diferencia_goles')
        st.write(worst_matches[['actaid', 'fecha', 'partido', 'goles_my_team', 'goles_rival', 'diferencia_goles']])


    with tabs[1]:
        st.divider()
        st.subheader("Influencia del lugar de juego en el partido")


        st.info("Los test de Kolmogorov-Smirnov y Mann-Whitney U se utilizan para comparar la distribución de la diferencia de goles y goles totales entre los partidos jugados en casa y fuera de casa. Un p-valor menor a 0.05 indica que las distribuciones son diferentes. Si el p-valor es mayor a 0.05, no se puede rechazar la hipótesis nula de que las distribuciones son iguales.", icon="ℹ️")


        # Distribución de la diferencia de goles
        fig = get_fig_dist_dif(all_matches, selected_team)
        st.plotly_chart(fig)

        # Distribución de goles totales
        fig2 = get_dist_goles_totales(all_matches, selected_team)
        st.plotly_chart(fig2)

    with tabs[2]:
        st.divider()
        st.subheader("Exclusiones")
        st.info("Los test de Kolmogorov-Smirnov y Mann-Whitney U se utilizan para comparar la distribución de exclusiones entre los partidos jugados en casa y fuera de casa. Un p-valor menor a 0.05 indica que las distribuciones son diferentes. Si el p-valor es mayor a 0.05, no se puede rechazar la hipótesis nula de que las distribuciones son iguales.", icon="ℹ️")

        # Distribución de exclusiones
        fig3 = get_fig_dist_exclusionses_lugar(all_matches, selected_team)
        st.plotly_chart(fig3)

        # Distribución de exclusiones del equipo seleccionado y del rival
        fig4 = get_fig_dist_exclusiones_rival(all_matches, selected_team)
        st.plotly_chart(fig4)




    with tabs[3]:

        st.subheader("Resultados Parciales vs Resultados Finales")
        st.info("En estos mapas de calor se puede ver la relación entre los resultados parciales y el resultado final del partido.", icon="ℹ️")

        # HeatMap de resultados en los primeros 5 minutos vs Resultados finales
        fig5 = get_fig_resultados_5mins(all_matches, selected_team)
        st.plotly_chart(fig5)

        # HeatMap de resultados en la mitad del partido vs Resultados finales
        fig6 = get_fig_resultados_mitad(all_matches, selected_team)
        st.plotly_chart(fig6)

        # HeatMap de resultados al minuto 50 vs Resultados finales
        fig7 = get_fig_resultados_min50(all_matches, selected_team)
        st.plotly_chart(fig7)


        st.divider()
        st.subheader("Análisis de los resultados parciales")

        st.info("La linea discontinua horizontal roja indica la mediana de victorias en parciales. La linea discontinua horizontal azul indica la mediana de derrotas en parciales desde arriba hacia abajo.", icon="ℹ️")

        fig8 = get_fig_resultados_parciales(all_matches, selected_team)
        st.plotly_chart(fig8)

        fig9 = get_fig_resultados_parciales_local(all_matches, selected_team)
        st.plotly_chart(fig9)

        fig_resultados_parciales_visitante = get_fig_resultados_parciales_visitante(all_matches, selected_team)
        st.plotly_chart(fig_resultados_parciales_visitante)

        fig_rendimiento = get_fig_rendimiento_parciales(all_matches, selected_team)
        st.plotly_chart(fig_rendimiento)

        st.divider()
        st.subheader("Análisis de las diferencias parciales")
        st.info("Se muestra en cada box-plot la distribución de la diferencia de goles de cada parcial. Se muestra una linea con la mediana.", icon="ℹ️")

        fig10 = get_fig_diferencias_parciales(all_matches, selected_team)
        st.plotly_chart(fig10)

        fig_diferencias_parciales_local = get_fig_diferencias_parciales_local(all_matches, selected_team)
        st.plotly_chart(fig_diferencias_parciales_local)

        fig_diferencias_parciales_visitante = get_fig_diferencias_parciales_visitante(all_matches, selected_team)
        st.plotly_chart(fig_diferencias_parciales_visitante)

    with tabs[4]:
        
        st.subheader("Transiciones entre rachas")

        st.info("En este gráfico se muestra la racha de victorias. El valor '0' indica un empate. Un valor positivo indica una racha de victorias y un valor negativo indica una racha de derrotas.", icon="ℹ️")

        fig_rachas = get_fig_rachas(all_matches, selected_team)

        
        # show as fig_rachas as table
        st.write(fig_rachas)

        st.write("Tabla de probabilidades de rachas:")
        st.info("En esta tabla se muestra la probabilidad de que el equipo gane, empate o pierda en función de la racha de victorias o derrotas. Cada racha es una fila y cada columna es la probabilidad de ganar, empatar o perder.", icon="ℹ️")
        tabla_probalbilidades_rachas = get_tabla_probabilidades_rachas(all_matches, selected_team)
        st.write(tabla_probalbilidades_rachas)
        st.divider()

        fig_rachas_puntos = get_fig_rachas_puntos(all_matches, selected_team)
        st.header("Influencia de los últimos partidos en el resultado:")
        st.info("En el eje horizontal se muentra la diferencia de goles de los partidos mientras que en el eje vertical los puntos obtenidos en los últimos 5 partidos."
                " Se muestra una regresión lineal junto a su Coeficiente de Determinación (R²), que indica la opacidad de la linea de la regresión ", icon="ℹ️")
        
        st.write(fig_rachas_puntos)

    with tabs[5]:

        st.divider()
        st.subheader("Análisis de los tiempo muertos")
        st.info("Distribución de los tiempos muertos por parciales.", icon="ℹ️")

        
        fig_histograma_tiempos_muertos = get_analisis_tiempos_muertos(all_matches, selected_team)
        st.plotly_chart(fig_histograma_tiempos_muertos)

    with tabs[6]:


        st.divider()
        st.subheader("Máximos goleadores")
        st.info("Se muestra en un gráfico de barras horizontal los goles totales de cada jugador y su meddiana de goles. El tamaño de la barra de la mediana está escalado", icon="ℹ️")


        fig_maximos_goleadores = get_maximos_goleadores(all_matches, selected_team)
        st.plotly_chart(fig_maximos_goleadores)


        st.divider()
        st.subheader("Máximos infractores")
        st.info("Se muestra en un gráfico de barras horizontal las exclusiones y tarjetas rojas de cada jugador", icon="ℹ️") 
        fig_maximos_expulsados = get_maximos_infractores(all_matches, selected_team)
        st.plotly_chart(fig_maximos_expulsados)











def crear_columnas(df, chosen_team):
    # Replace column names containing 'local' with 'my_team' and 'visitante' with 'rival'

    # Filter matches where the chosen team is the local team
    local_matches = df[df['nombre_local'] == chosen_team]
    visiting_matches = df[df['nombre_visitante'] == chosen_team]
    
    # Replace column names containing 'local' with 'my_team' and 'visitante' with 'rival'
    if not local_matches.empty:
        local_matches.columns = local_matches.columns.str.replace('local', 'my_team').str.replace('visitante', 'rival')
        local_matches.loc[:, 'my_team_is'] = "Local" 

    if not visiting_matches.empty:
        # Filter matches where the chosen team is the visiting team
        # Replace column names containing 'visitante' with 'my_team' and 'local' with 'rival'
        visiting_matches.columns = visiting_matches.columns.str.replace('visitante', 'my_team').str.replace('local', 'rival')
        visiting_matches.loc[:, 'my_team_is'] = "Visitante"

    # Concatenate the filtered matches
    if not local_matches.empty and not visiting_matches.empty:
        all_matches = pd.concat([local_matches, visiting_matches])
    if local_matches.empty:
        all_matches = visiting_matches
    if visiting_matches.empty:
        all_matches = local_matches


    print(all_matches.columns)
    # Sort by actaid and reset the index
    all_matches = all_matches.sort_values(by='actaid').reset_index(drop=True)

    # Columns "my_team_is" ins the 5th column
    all_matches = all_matches[all_matches.columns.tolist()[:4] + ['my_team_is'] + all_matches.columns.tolist()[4:-1]]

    # Diferencia de goles
    all_matches['diferencia_goles'] = all_matches['goles_my_team'] - all_matches['goles_rival']

    all_matches['partido'] = all_matches.apply(lambda row: f"{row['nombre_my_team']} vs {row['nombre_rival']}" if row['my_team_is'] == 'Local' else f"{row['nombre_rival']} vs {row['nombre_my_team']}", axis=1)

    # Columna goles_totales
    all_matches['goles_totales'] = all_matches['goles_my_team'] + all_matches['goles_rival']

    return all_matches

        
def get_fig_dist_dif(df, chosen_team):
    """Devuelve un gráfico de caja con la distribución de la diferencia de goles del equipo seleccionado."""

    # Diferencia de goles siendo local y visitante
    local_team_matches = df[df['my_team_is'] == 'Local']
    visiting_team_matches = df[df['my_team_is'] == 'Visitante']
    # Create subplots with 1 row and 2 columns, sharing the y-axis
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Diferencia de Goles siendo Local', 'Diferencia de Goles siendo Visitante'),
                        shared_yaxes=True)  # Share y-axis
    
    # Add the box plot for 'diferencia_goles' where my_team_is 'Local'
    fig.add_trace(
        px.box(local_team_matches,
                y='diferencia_goles',
                color_discrete_sequence=['#FFA500'],
                hover_data= {'partido': True}
                ).data[0],
        row=1, col=1
    )

    # Add the box plot for 'diferencia_goles' where my_team_is 'Visiting'
    fig.add_trace(
        px.box(visiting_team_matches,
                y='diferencia_goles',
                color_discrete_sequence=['#00FFFF'],
                hover_data= {'partido': True}
                ).data[0],
        row=1, col=2
    )


    # Update layout for the entire figure
    fig.update_layout(
    title=dict(
        text=f'Distribución de Diferencia de Goles de {chosen_team}',
        x=0.5,  # Position the title in the center
        xanchor='center',  # Anchor the title to the center
        y=0.9,  # Position the title at the top
        yanchor='top'  # Anchor the title to the top
    ),
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
    plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
    font=dict(color='#ececec')  # Font color
    )

    # Update y-axis title (only needed once because of shared y-axis)
    fig.update_yaxes(title_text='Diferencia de Goles', row=1, col=1)

    mannwhitneyu_result = mannwhitneyu(local_team_matches['diferencia_goles'], visiting_team_matches['diferencia_goles'])
    ks_result = ks_2samp(local_team_matches['diferencia_goles'], visiting_team_matches['diferencia_goles'])

    # Add the Mann-Whitney U test result to the figure
    fig.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.2,  # Position the annotation in the center
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Mann-Whitney U test: p-value={mannwhitneyu_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    # Add the Kolmogorov-Smirnov test result to the figure
    fig.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.13,  # Position the annotation below the Mann-Whitney U test result
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Kolmogorov-Smirnov test: p-value={ks_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    return fig


def get_dist_goles_totales(df, chosen_team):
    """Devuelve un gráfico de caja con la distribución de goles totales del equipo seleccionado."""

        # Distribution of 'goles_totales' for the chosen team

    # Filter the matches where my_team_is 'Local' and 'Visiting'
    local_team_matches = df[df['my_team_is'] == 'Local']
    visiting_team_matches = df[df['my_team_is'] == 'Visitante']

    # Create subplots with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Goles Totales siendo Local', 'Goles Totales siendo Visitante'), 
                        shared_yaxes=True)  # Share y-axis

    # Add the box plot for 'goles_totales' where my_team_is 'Local'
    fig.add_trace(
        px.box(local_team_matches, y='goles_totales', 
            color_discrete_sequence=['#FFA500'],
            hover_data= {'partido': True}
            ).data[0],
        row=1, col=1
    )

    # Add the box plot for 'goles_totales' where my_team_is 'Visitante'
    fig.add_trace(
        px.box(visiting_team_matches, y='goles_totales', 
            color_discrete_sequence=['#00FFFF'],
            hover_data= {'partido': True}
            ).data[0],
        row=1, col=2
    )

    # Update layout for the entire figure
    fig.update_layout(
        title=dict(
            text=f'Distribución de Goles Totales de {chosen_team}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec'),  # Font color
        yaxis=dict(range=[df['goles_totales'].min(), df['goles_totales'].max()])  # Set the same scale for both plots
    )

    # Update y-axis titles
    fig.update_yaxes(title_text='Goles Totales', row=1, col=1)

    mannwhitneyu_result = mannwhitneyu(local_team_matches['goles_totales'], visiting_team_matches['goles_totales'])
    ks_result = ks_2samp(local_team_matches['goles_totales'], visiting_team_matches['goles_totales'])

    # Add the Mann-Whitney U test result to the figure
    fig.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.2,  # Position the annotation in the center
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Mann-Whitney U test: p-value={mannwhitneyu_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    # Add the Kolmogorov-Smirnov test result to the figure
    fig.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.13,  # Position the annotation below the Mann-Whitney U test result
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Kolmogorov-Smirnov test: p-value={ks_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    return fig

def get_fig_dist_exclusionses_lugar(df, chosen_team):
    """Devuelve un gráfico de caja con la distribución de exclusiones del equipo seleccionado como local y como visitante."""

    # Filter the matches where my_team_is 'Local' and 'Visitante'
    local_team_matches = df[df['my_team_is'] == 'Local']
    visiting_team_matches = df[df['my_team_is'] == 'Visitante']

    # Create subplots with 1 row and 2 columns, sharing the y-axis
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Exclusiones siendo Local', 'Exclusiones siendo Visitante'),
                        shared_yaxes=True)  # Share y-axis

    # Add the box plot for 'exclusiones_my_team' where my_team_is 'Local'
    fig.add_trace(
        px.box(local_team_matches, y='exclusiones_my_team', color_discrete_sequence=['#FFA500'],
            hover_data= {'partido': True}
            ).data[0],
        row=1, col=1
    )

    # Add the box plot for 'exclusiones_my_team' where my_team_is 'Visitante'
    fig.add_trace(
        px.box(visiting_team_matches, y='exclusiones_my_team', color_discrete_sequence=['#00FFFF'],
            hover_data= {"partido": True}).data[0],
        row=1, col=2
    )

    # Update layout for the entire figure
    fig.update_layout(
        title=dict(
            text=f'Distribución de Exclusiones de {chosen_team}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec')  # Font color
    )

    # Update y-axis title (only needed once because of shared y-axis)
    fig.update_yaxes(title_text='Exclusiones', row=1, col=1)

    # Perform the Mann-Whitney U test

    mannwhitneyu_result = mannwhitneyu(local_team_matches['exclusiones_my_team'], visiting_team_matches['exclusiones_my_team'])
    ks_result = ks_2samp(local_team_matches['exclusiones_my_team'], visiting_team_matches['exclusiones_my_team'])

    # Add the Mann-Whitney U test result to the figure
    fig.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.2,  # Position the annotation in the center
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Mann-Whitney U test: p-value={mannwhitneyu_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    # Add the Kolmogorov-Smirnov test result to the figure
    fig.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.13,  # Position the annotation below the Mann-Whitney U test result
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Kolmogorov-Smirnov test: p-value={ks_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    return fig

def get_fig_dist_exclusiones_rival(df, chosen_team):
    """Devuelve un gráfico de caja con la distribución de exclusiones del equipo seleccionado como local y como visitante."""
    

    # Create subplots with 1 row and 2 columns, sharing the y-axis
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Exclusiones {chosen_team}', 'Exclusiones Rival'),
                        shared_yaxes=True)  # Share y-axis

    # Add the box plot for 'exclusiones_my_team' 
    fig.add_trace(
        px.box(df, y='exclusiones_my_team', color_discrete_sequence=['#FFA500'],
            hover_data= {'partido': True}
            ).data[0],
        row=1, col=1
    )

    # Add the box plot for 'exclusiones_rival' 
    fig.add_trace(
        px.box(df, y='exclusiones_rival', color_discrete_sequence=['#00FFFF'],
            hover_data= {'partido': True}
            ).data[0],
        row=1, col=2
    )


    # Update layout for the entire figure
    fig.update_layout(
        title=dict(
            text=f'Distribución de Exclusiones de {chosen_team} y Rival',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec')  # Font color
    )

    # Update y-axis title (only needed once because of shared y-axis)
    fig.update_yaxes(title_text='Exclusiones', row=1, col=1)

    # Perform the Mann-Whitney U test
    mannwhitneyu_result = mannwhitneyu(df['exclusiones_my_team'], df['exclusiones_rival'])
    ks_result = ks_2samp(df['exclusiones_my_team'], df['exclusiones_rival'])

    # Add the Mann-Whitney U test result to the figure
    fig.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.2,  # Position the annotation in the center
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Mann-Whitney U test: p-value={mannwhitneyu_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    # Add the Kolmogorov-Smirnov test result to the figure
    fig.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.13,  # Position the annotation below the Mann-Whitney U test result
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Kolmogorov-Smirnov test: p-value={ks_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    return fig

def get_fig_resultados_5mins(df, chosen_team):
    """Devuelve un heatmap con los resultados en los primeros 5 minutos vs los resultados finales."""

    umbral = 1

    # Crear la columna del resultado final basado en diferencia de goles
    df['final_result'] = df['diferencia_goles'].apply(
        lambda x: 'Victoria (Partido)' if x > 0 else ('Derrota (Partido)' if x < 0 else 'Empate (Partido)')
    )

    # Crear la columna del resultado de los primeros 5 minutos
    df['resultado_5_min'] = df['parcial_1_my_team'] - df['parcial_1_rival']
    df['resultado_5_min'] = df['resultado_5_min'].apply(
        lambda x: 'Victoria (5 Minutos)' if x > umbral else ('Derrota (5 Minutos)' if x < -umbral else 'Empate (5 Minutos)')
    )

    # Agrupar los partidos por las combinaciones de resultados
    grouped = df.groupby(['resultado_5_min', 'final_result'])['partido'].apply(list).unstack(fill_value=[])

    # Crear una matriz de conteo para el heatmap
    heatmap_data = df.groupby(['resultado_5_min', 'final_result']).size().unstack(fill_value=0)

    # Agregar filas/columnas faltantes
    for col in ['Victoria (Partido)', 'Empate (Partido)', 'Derrota (Partido)']:
        if col not in heatmap_data.columns:
            heatmap_data[col] = 0

    for row in ['Victoria (5 Minutos)', 'Empate (5 Minutos)', 'Derrota (5 Minutos)']:
        if row not in heatmap_data.index:
            heatmap_data.loc[row] = 0

    # Reordenar filas y columnas
    heatmap_data = heatmap_data[['Derrota (Partido)', 'Empate (Partido)', 'Victoria (Partido)']]
    heatmap_data = heatmap_data.loc[['Derrota (5 Minutos)', 'Empate (5 Minutos)', 'Victoria (5 Minutos)']]

    # Crear el texto para hover mostrando los partidos
    hover_text = grouped.reindex(index=heatmap_data.index, columns=heatmap_data.columns, fill_value=[]).applymap(
        lambda partidos: '<br>'.join(map(str, partidos)) if partidos else 'Sin partidos'
    )

    # Crear el heatmap con hover text
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Cividis',
        text=hover_text.values,           # Mostrar los partidos en hover
        texttemplate="%{z}",              # Mostrar el conteo en las celdas
        hovertemplate='%{text}'           # Mostrar partidos al pasar el cursor
    ))

    # Update layout
    fig.update_layout(
        title_text="Número de Partidos por Resultado en los Primeros 5 Minutos y al Final del Partido",
        title=dict(
            x=0.5,  # Center the title
            xanchor='center',
            font=dict(size=12, color='#ececec')  # Font color
        ),
        xaxis_title='Resultado al Final del Partido',
        yaxis_title='Diferencia en los Primeros 5 Minutos',
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=[f'<-{umbral}', f'[-{umbral}, {umbral}]', f'>{umbral}']
        ),
        xaxis = dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['Derrota', 'Empate', 'Victoria']
        ),
        font=dict(size=16, color='#ececec'),  # Font color
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',    # Transparent background for the plot
    )

    return fig
    
def get_fig_resultados_mitad(df, chosen_team):
    """Devuelve un heatmap con los resultados en la mitad del partido vs los resultados finales."""



    umbral = 3

    # Crear la columna del resultado final basado en diferencia de goles
    df['final_result'] = df['diferencia_goles'].apply(
        lambda x: 'Victoria (Partido)' if x > 0 else ('Derrota (Partido)' if x < 0 else 'Empate (Partido)')
    )

    # Crear la columna del resultado al descanso
    df['resultado_1_parte'] = df['parcial_6_my_team'] - df['parcial_6_rival']
    df['resultado_1_parte'] = df['resultado_1_parte'].apply(
        lambda x: 'Victoria (Descanso)' if x > umbral else ('Derrota (Descanso)' if x < -umbral else 'Empate (Descanso)')
    )

    # Agrupar los partidos por combinaciones de resultados
    grouped = df.groupby(['resultado_1_parte', 'final_result'])['partido'].apply(list).unstack(fill_value=[])

    # Crear una matriz de conteo para el heatmap
    heatmap_data = df.groupby(['resultado_1_parte', 'final_result']).size().unstack(fill_value=0)

    # Asegurar que las filas/columnas existan
    for col in ['Victoria (Partido)', 'Empate (Partido)', 'Derrota (Partido)']:
        if col not in heatmap_data.columns:
            heatmap_data[col] = 0

    for row in ['Victoria (Descanso)', 'Empate (Descanso)', 'Derrota (Descanso)']:
        if row not in heatmap_data.index:
            heatmap_data.loc[row] = 0

    # Reordenar filas y columnas
    heatmap_data = heatmap_data[['Derrota (Partido)', 'Empate (Partido)', 'Victoria (Partido)']]
    heatmap_data = heatmap_data.reindex(['Derrota (Descanso)', 'Empate (Descanso)', 'Victoria (Descanso)'])

    # Crear el texto para hover mostrando los partidos
    hover_text = grouped.reindex(index=heatmap_data.index, columns=heatmap_data.columns, fill_value=[]).applymap(
        lambda partidos: '<br>'.join(map(str, partidos)) if partidos else 'Sin partidos'
    )

    # Crear el heatmap con hover text
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Cividis',
        text=hover_text.values,           # Mostrar los partidos en hover
        texttemplate="%{z}",              # Mostrar el conteo en las celdas
        hovertemplate='%{text}'           # Mostrar partidos al pasar el cursor
    ))

    # Update layout
    fig.update_layout(
        title_text="Número de Partidos por Resultado en el Descanso y al Final del Partido",
        title=dict(
            x=0.5,  # Center the title
            xanchor='center',
            font=dict(size=12, color='#ececec')  # Font color
        ),
        xaxis_title='Resultado al Final del Partido',
        yaxis_title='Diferencia en el Descanso',
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=[f'<-{umbral}', f'[-{umbral}, {umbral}]', f'>{umbral}']
        ),
        xaxis = dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['Derrota', 'Empate', 'Victoria']
        ),
        font=dict(size=16, color='#ececec'),  # Font color
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',    # Transparent background for the plot
    )

    return fig

def get_fig_resultados_min50(df, chosen_team):
    """Devuelve un heatmap con los resultados en el minuto 50 vs los resultados finales."""
    umbral = 2

    # Crear la columna para el resultado al minuto 50
    df['resultado_min_50'] = df['parcial_10_my_team'] - df['parcial_10_rival'] if num_parciales == 12 else df['parcial_8_my_team'] - df['parcial_8_rival']
    df['resultado_min_50'] = df['resultado_min_50'].apply(
        lambda x: 'Victoria (Min 50)' if x > umbral else ('Derrota (Min 50)' if x < -umbral else 'Empate (Min 50)')
    )

    # Agrupar los partidos por combinaciones de resultados
    grouped = df.groupby(['resultado_min_50', 'final_result'])['partido'].apply(list).unstack(fill_value=[])

    # Crear una matriz de conteo para el heatmap
    heatmap_data = df.groupby(['resultado_min_50', 'final_result']).size().unstack(fill_value=0)

    # Asegurar que las filas/columnas existan
    for col in ['Victoria (Partido)', 'Empate (Partido)', 'Derrota (Partido)']:
        if col not in heatmap_data.columns:
            heatmap_data[col] = 0

    for row in ['Victoria (Últimos 10 mins)', 'Empate (Últimos 10 mins)', 'Derrota (Últimos 10 mins)']:
        if row not in heatmap_data.index:
            heatmap_data.loc[row] = 0

    # Reordenar filas y columnas
    heatmap_data = heatmap_data[['Derrota (Partido)', 'Empate (Partido)', 'Victoria (Partido)']]
    heatmap_data = heatmap_data.reindex(['Derrota (Min 50)', 'Empate (Min 50)', 'Victoria (Min 50)'])

    # Crear el texto para hover mostrando los partidos
    hover_text = grouped.reindex(index=heatmap_data.index, columns=heatmap_data.columns, fill_value=[]).applymap(
        lambda partidos: '<br>'.join(map(str, partidos)) if partidos else 'Sin partidos'
    )

    # Crear el heatmap con hover text
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Cividis',
        text=hover_text.values,           # Mostrar los partidos en hover
        texttemplate="%{z}",              # Mostrar el conteo en las celdas
        hovertemplate='%{text}'           # Mostrar partidos al pasar el cursor
    ))

    # Update layout
    # Update layout
    fig.update_layout(
        title_text="Número de Partidos por Resultado en los últimos 10 mins y al Final del Partido",
        title=dict(
            x=0.5,  # Center the title
            xanchor='center',
            font=dict(size=12, color='#ececec')  # Font color
        ),
        xaxis_title='Resultado al Final del Partido',
        yaxis_title='Diferencia a falta de 10 mins',
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=[f'<-{umbral}', f'[-{umbral}, {umbral}]', f'>{umbral}']
        ),
        xaxis = dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['Derrota', 'Empate', 'Victoria']
            ),
            
        font=dict(size=16, color='#ececec'),  # Font color
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',    # Transparent background for the plot
    )

    return fig

def get_fig_resultados_parciales(df, chosen_team):

    # Crear listas para almacenar los resultados
    same_winner_counts = []
    different_winner_counts = []
    empate_counts = []
    parciales = list(range(1, 1+num_parciales))  # Números de parciales del 1 al 12

    # Loop sobre los parciales del 1 al 12
    for i in parciales:
        parcial_column = f'ganador_parcial_{i}'
        
        # Contar los casos en los que el ganador del parcial es el mismo equipo
        same_winner = df[df[parcial_column] == df['my_team_is']]
        same_winner_counts.append(same_winner.shape[0])
        
        # Contar los casos en los que hay empate
        empate = df[df[parcial_column] == 'Empate']
        empate_counts.append(empate.shape[0])
        
        # Contar los casos en los que el ganador del parcial es diferente equipo (rival)
        different_winner = df[df[parcial_column] != df['my_team_is']]
        different_winner = different_winner[different_winner[parcial_column] != 'Empate']
        different_winner_counts.append(different_winner.shape[0])

    # Crear un histograma apilado con los resultados de cada parcial
    fig = go.Figure()

    # Añadir las barras para los casos donde el ganador es el equipo del usuario
    fig.add_trace(go.Bar(
        x=parciales,
        y=same_winner_counts,
        name='Victoria',
        marker=dict(color='#FFA500')
    ))

    # Añadir las barras para los casos de empate
    fig.add_trace(go.Bar(
        x=parciales,
        y=empate_counts,
        name='Empate',
        marker=dict(color='#FFB6C1')
    ))

    # Añadir las barras para los casos donde el ganador es el rival
    fig.add_trace(go.Bar(
        x=parciales,
        y=different_winner_counts,
        name='Derrota',
        marker=dict(color='#00FFFF')
    ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        barmode='stack',
        title=dict(
            text=f'Resultados de los Parciales para {chosen_team}',
            x=0.5,
            xanchor='center',
            y=0.9,
            yanchor='top'
        ),
        xaxis=dict(
            title='Número de Parcial',
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        yaxis=dict(title='Cantidad de Partidos'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5,
            font=dict(size=14)
        )
    )

    median_same_winner = np.median(same_winner_counts)
    median_same_loser = np.median(different_winner_counts)

    print(f"Mediana de partidos ganados por {chosen_team}: {median_same_winner}")
    print(f"Mediana de partidos perdidos por {chosen_team}: {median_same_loser}")
    total_partidos = (sum(same_winner_counts) + sum(different_winner_counts) + sum(empate_counts))/num_parciales
    print(f"Total de partidos: {total_partidos}")

    # Add a line for the median
    fig.add_shape(
        type='line',
        x0=0.5,
        y0=median_same_winner,
        x1=num_parciales + 0.5,
        y1=median_same_winner,
        line=dict(color='Red', width=2, dash='dash')
    )

    fig.add_shape(
        type='line',
        x0=0.5,
        y0=total_partidos - median_same_loser,
        x1=num_parciales + 0.5,
        y1=total_partidos - median_same_loser,
        line=dict(color='Blue', width=2, dash='dash')
    )


    return fig

def get_fig_resultados_parciales_local(df, chosen_team):

    # Crear listas para almacenar los resultados solo para los partidos donde el equipo es Local y Visitante
    same_winner_counts_local = []
    different_winner_counts_local = []
    empate_counts_local = []
    
    parciales = list(range(1, 1+num_parciales))  # Números de parciales del 1 al 12

    local_matches_only = df[df['my_team_is'] == 'Local']

    # Loop sobre los parciales del 1 al 12 para Local
    for i in parciales:
        parcial_column = f'ganador_parcial_{i}'
        
        # Contar los casos en los que el ganador del parcial es el mismo equipo
        same_winner_local = local_matches_only[local_matches_only[parcial_column] == local_matches_only['my_team_is']]
        same_winner_counts_local.append(same_winner_local.shape[0])
        
        # Contar los casos en los que hay empate
        empate_local = local_matches_only[local_matches_only[parcial_column] == 'Empate']
        empate_counts_local.append(empate_local.shape[0])
        
        # Contar los casos en los que el ganador del parcial es diferente equipo (rival)
        different_winner_local = local_matches_only[local_matches_only[parcial_column] != local_matches_only['my_team_is']]
        different_winner_local = different_winner_local[different_winner_local[parcial_column] != 'Empate']
        different_winner_counts_local.append(different_winner_local.shape[0])

    # Crear un histograma apilado con los resultados de cada parcial para Local
    fig_local = go.Figure()

    # Añadir las barras para los casos donde el ganador es el equipo del usuario
    fig_local.add_trace(go.Bar(
        x=parciales,
        y=same_winner_counts_local,
        name=f'Victoria',
        marker=dict(color='#FFA500')
    ))

    # Añadir las barras para los casos de empate
    fig_local.add_trace(go.Bar(
        x=parciales,
        y=empate_counts_local,
        name='Empate',
        marker=dict(color='#FFB6C1')
    ))

    # Añadir las barras para los casos donde el ganador es el rival
    fig_local.add_trace(go.Bar(
        x=parciales,
        y=different_winner_counts_local,
        name='Derrota',
        marker=dict(color='#00FFFF')
    ))

    # Actualizar el diseño del gráfico
    fig_local.update_layout(
        barmode='stack',
        title=dict(
            text=f'Resultados de los Parciales para {chosen_team} (Local)',
            x=0.5,
            xanchor='center',
            y=0.9,
            yanchor='top'
        ),
        xaxis=dict(
            title='Número de Parcial',
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        yaxis=dict(title='Cantidad de Partidos'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5,
            font=dict(size=14)
        )
    )


    median_same_winner_local = np.median(same_winner_counts_local)
    median_same_loser_local = np.median(different_winner_counts_local)
    total_partidos_local = (sum(same_winner_counts_local) + sum(different_winner_counts_local) + sum(empate_counts_local))/num_parciales


    # Add a line for the median
    fig_local.add_shape(
        type='line',
        x0=0.5,
        y0=median_same_winner_local,
        x1=num_parciales + 0.5,
        y1=median_same_winner_local,
        line=dict(color='Red', width=2, dash='dash')
    )

    fig_local.add_shape(
        type='line',
        x0=0.5,
        y0=total_partidos_local - median_same_loser_local,
        x1=num_parciales + 0.5,
        y1=total_partidos_local - median_same_loser_local,
        line=dict(color='Blue', width=2, dash='dash')
    )

    return fig_local

def get_fig_resultados_parciales_visitante(df, chosen_team):
    
    # Crear listas para almacenar los resultados solo para los partidos donde el equipo es Local y Visitante
    same_winner_counts_visitante = []
    different_winner_counts_visitante = []
    empate_counts_visitante = []
    
    parciales = list(range(1, 1+num_parciales))  # Números de parciales del 1 al 12

    visiting_matches_only = df[df['my_team_is'] == 'Visitante']

    # Loop sobre los parciales del 1 al 12 para Visitante
    for i in parciales:
        parcial_column = f'ganador_parcial_{i}'
        
        # Contar los casos en los que el ganador del parcial es el mismo equipo
        same_winner_visitante = visiting_matches_only[visiting_matches_only[parcial_column] == visiting_matches_only['my_team_is']]
        same_winner_counts_visitante.append(same_winner_visitante.shape[0])
        
        # Contar los casos en los que hay empate
        empate_visitante = visiting_matches_only[visiting_matches_only[parcial_column] == 'Empate']
        empate_counts_visitante.append(empate_visitante.shape[0])
        
        # Contar los casos en los que el ganador del parcial es diferente equipo (rival)
        different_winner_visitante = visiting_matches_only[visiting_matches_only[parcial_column] != visiting_matches_only['my_team_is']]
        different_winner_visitante = different_winner_visitante[different_winner_visitante[parcial_column] != 'Empate']
        different_winner_counts_visitante.append(different_winner_visitante.shape[0])

    # Crear un histograma apilado con los resultados de cada parcial para Visitante
    fig_visitante = go.Figure()

    # Añadir las barras para los casos donde el ganador es el equipo del usuario
    fig_visitante.add_trace(go.Bar(
        x=parciales,
        y=same_winner_counts_visitante,
        name=f'Victoria',
        marker=dict(color='#FFA500')
    ))

    # Añadir las barras para los casos de empate
    fig_visitante.add_trace(go.Bar(
        x=parciales,
        y=empate_counts_visitante,
        name='Empate',
        marker=dict(color='#FFB6C1')
    ))

    # Añadir las barras para los casos donde el ganador es el rival
    fig_visitante.add_trace(go.Bar(
        x=parciales,
        y=different_winner_counts_visitante,
        name='Derrota',
        marker=dict(color='#00FFFF')
    ))

    # Actualizar el diseño del gráfico
    fig_visitante.update_layout(
        barmode='stack',
        title=dict(
            text=f'Resultados de los Parciales para {chosen_team} (Visitante)',
            x=0.5,
            xanchor='center',
            y=0.9,
            yanchor='top'
        ),
        xaxis=dict(
            title='Número de Parcial',
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        yaxis=dict(title='Cantidad de Partidos'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5,
            font=dict(size=14)
        )
    )

    median_same_winner_visitante = np.median(same_winner_counts_visitante)
    median_same_loser_visitante = np.median(different_winner_counts_visitante)
    total_partidos_visitante = (sum(same_winner_counts_visitante) + sum(different_winner_counts_visitante) + sum(empate_counts_visitante))/num_parciales

    # Add a line for the median
    fig_visitante.add_shape(
        type='line',
        x0=0.5,
        y0=median_same_winner_visitante,
        x1=num_parciales + 0.5,
        y1=median_same_winner_visitante,
        line=dict(color='Red', width=2, dash='dash')
    )

    fig_visitante.add_shape(
        type='line',
        x0=0.5,
        y0=total_partidos_visitante - median_same_loser_visitante,
        x1=num_parciales + 0.5,
        y1=total_partidos_visitante - median_same_loser_visitante,
        line=dict(color='Blue', width=2, dash='dash')
    )

    return fig_visitante

def get_fig_diferencias_parciales(df, chosen_team):
    differences = {}

    # Calcular la diferencia del primer parcial.
    differences[f'parcial_1_diff'] = df['parcial_1_my_team'] - df['parcial_1_rival']
    differences[f'partido'] = df['partido']

    # Calcular la diferencia para los demás parciales ajustando por el incremento exclusivo de ese parcial.
    for i in range(2, 1+num_parciales):
        differences[f'parcial_{i}_diff'] = (
            (df[f'parcial_{i}_my_team'] - df[f'parcial_{i}_rival']) -
            (df[f'parcial_{i-1}_my_team'] - df[f'parcial_{i-1}_rival'])
        )

    # Convertir el diccionario a un DataFrame.
    differences_df = pd.DataFrame(differences)

    # Transformar los datos a formato largo (long format) para Plotly.
    long_format = differences_df.melt(var_name='Parcial', value_name='Diferencia de Goles', id_vars='partido')

    # Renombrar los parciales para hacerlos más legibles.
    long_format['Parcial'] = long_format['Parcial'].str.replace('_diff', '').str.replace('parcial_', '')

    # Calcular las medianas por cada parcial.
    long_format['Parcial'] = pd.Categorical(long_format['Parcial'], 
                                            categories=[f'{i}' for i in range(1, 1+num_parciales)], 
                                            ordered=True)

    medianas = long_format.groupby('Parcial')['Diferencia de Goles'].median().reset_index()

    # Crear el gráfico de boxplots.
    fig = px.box(
        long_format,
        x='Parcial',
        y='Diferencia de Goles',
        color_discrete_sequence=['#FFA500'],
        title=f'Distribución de Diferencia de Goles por Parcial para {chosen_team}',
        hover_data={'Parcial': False, 'Diferencia de Goles': True, "partido": True}
    )

    # Añadir una línea que conecte las medianas asegurándose de que las etiquetas coincidan con los parciales.
    fig.add_trace(go.Scatter(
        x=medianas['Parcial'],
        y=medianas['Diferencia de Goles'],
        mode='lines+markers',
        name='Mediana',
        line=dict(color= '#00FFFF', width=3, dash='dash')
                
    ))

    # Actualizar el diseño del gráfico.
    fig.update_layout(
        xaxis=dict(title='Parcial'),
        yaxis=dict(title='Diferencia de Goles'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec'),
        title = dict(
            x=0.5,
            xanchor='center',
            y=0.9,
            yanchor='top'
        )
        
    )

    # add ticks very x units
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)

    # line at y = 0 for reference
    fig.add_shape(
        type='line',
        x0=0.5,
        y0=0,
        x1=num_parciales + 0.5,
        y1=0,
        line=dict(color='rgba(255, 0, 0, 0.35)', width=2)

    )

    return fig

def get_fig_diferencias_parciales_local(df, chosen_team):

    # filter df for local matches only
    local_matches_only = df[df['my_team_is'] == 'Local']

    fig = get_fig_diferencias_parciales(local_matches_only, f'{chosen_team} (Local)')


    # Add median line fot all matches (df) for each parcial

    differences = {}


    # Calcular la diferencia del primer parcial.
    differences[f'parcial_1_diff'] = df['parcial_1_my_team'] - df['parcial_1_rival']
    differences[f'partido'] = df['partido']

    # Calcular la diferencia para los demás parciales ajustando por el incremento exclusivo de ese parcial.
    for i in range(2, 1+num_parciales):
        differences[f'parcial_{i}_diff'] = (
            (df[f'parcial_{i}_my_team'] - df[f'parcial_{i}_rival']) -
            (df[f'parcial_{i-1}_my_team'] - df[f'parcial_{i-1}_rival'])
        )

    # Convertir el diccionario a un DataFrame.
    differences_df = pd.DataFrame(differences)

    # Transformar los datos a formato largo (long format) para Plotly.
    long_format = differences_df.melt(var_name='Parcial', value_name='Diferencia de Goles', id_vars='partido')

    # Renombrar los parciales para hacerlos más legibles.
    long_format['Parcial'] = long_format['Parcial'].str.replace('_diff', '').str.replace('parcial_', '')

    # Calcular las medianas por cada parcial.
    long_format['Parcial'] = pd.Categorical(long_format['Parcial'], 
                                            categories=[f'{i}' for i in range(1, 1+num_parciales)], 
                                            ordered=True)

    medianas = long_format.groupby('Parcial')['Diferencia de Goles'].median().reset_index()

    # Add a line connecting the medians for all matches
    fig.add_trace(go.Scatter(
        x=medianas['Parcial'],
        y=medianas['Diferencia de Goles'],
        mode='lines+markers',
        name='Mediana de Todos los Partidos',
        line=dict(color='#FFA500', width=3, dash='dash')
    ))

    return fig

def get_fig_diferencias_parciales_visitante(df, chosen_team):
    
    # filter df for visiting matches only
    visiting_matches_only = df[df['my_team_is'] == 'Visitante']

    fig = get_fig_diferencias_parciales(visiting_matches_only, f'{chosen_team} (Visitante)')

    
    # Add median line fot all matches (df) for each parcial

    differences = {}


    # Calcular la diferencia del primer parcial.
    differences[f'parcial_1_diff'] = df['parcial_1_my_team'] - df['parcial_1_rival']
    differences[f'partido'] = df['partido']

    # Calcular la diferencia para los demás parciales ajustando por el incremento exclusivo de ese parcial.
    for i in range(2, 1+num_parciales):
        differences[f'parcial_{i}_diff'] = (
            (df[f'parcial_{i}_my_team'] - df[f'parcial_{i}_rival']) -
            (df[f'parcial_{i-1}_my_team'] - df[f'parcial_{i-1}_rival'])
        )

    # Convertir el diccionario a un DataFrame.
    differences_df = pd.DataFrame(differences)

    # Transformar los datos a formato largo (long format) para Plotly.
    long_format = differences_df.melt(var_name='Parcial', value_name='Diferencia de Goles', id_vars='partido')

    # Renombrar los parciales para hacerlos más legibles.
    long_format['Parcial'] = long_format['Parcial'].str.replace('_diff', '').str.replace('parcial_', '')

    # Calcular las medianas por cada parcial.
    long_format['Parcial'] = pd.Categorical(long_format['Parcial'], 
                                            categories=[f'{i}' for i in range(1, 1+num_parciales)], 
                                            ordered=True)

    medianas = long_format.groupby('Parcial')['Diferencia de Goles'].median().reset_index()

    # Add a line connecting the medians for all matches
    fig.add_trace(go.Scatter(
        x=medianas['Parcial'],
        y=medianas['Diferencia de Goles'],
        mode='lines+markers',
        name='Mediana de Todos los Partidos',
        line=dict(color='#FFA500', width=3, dash='dash')
    ))

    return fig


def get_fig_rendimiento_parciales(df, chosen_team):

        # Crear listas para almacenar los resultados
    same_winner_counts = []
    different_winner_counts = []
    empate_counts = []
    parciales = list(range(1, 1+num_parciales))  # Números de parciales del 1 al 12

    # Loop sobre los parciales del 1 al 12
    for i in parciales:
        parcial_column = f'ganador_parcial_{i}'
        
        # Contar los casos en los que el ganador del parcial es el mismo equipo
        same_winner = df[df[parcial_column] == df['my_team_is']]
        same_winner_counts.append(same_winner.shape[0])
        
        # Contar los casos en los que hay empate
        empate = df[df[parcial_column] == 'Empate']
        empate_counts.append(empate.shape[0])
        
        # Contar los casos en los que el ganador del parcial es diferente equipo (rival)
        different_winner = df[df[parcial_column] != df['my_team_is']]
        different_winner = different_winner[different_winner[parcial_column] != 'Empate']
        different_winner_counts.append(different_winner.shape[0])


    summary_table = df.pivot_table(index='my_team_is', columns='resultado', aggfunc='size', fill_value=0)

        
    # Ensure that 'Victoria', 'Empate', and 'Derrota' columns exist
    for col in ['Victoria', 'Empate', 'Derrota']:
        if col not in summary_table.columns:
            summary_table[col] = 0

    # Reorder the columns 
    summary_table = summary_table[['Victoria', 'Empate', 'Derrota']]

    # Rename the index to the chosen team
    summary_table.index = [f"{chosen_team} (Local)" if idx == 'Local' else f'{chosen_team} (Visitante)' for idx in summary_table.index]



    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'En Partidos', f'En Parciales'),
                        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                        horizontal_spacing=0.3)

    # Add the pie chart for victories, draws, and losses
    fig.add_trace(
        go.Pie(labels=['Victorias', 'Empates', 'Derrotas'],
            values=[summary_table['Victoria'].sum(), summary_table['Empate'].sum(), summary_table['Derrota'].sum()],
            marker=dict(colors=['#FFA500', '#FFB6C1', '#00FFFF']),
            hole=0.3,
            textinfo='label+value+percent',
            textposition='outside',
            textfont=dict(color='#ececec')),
        row=1, col=1
    )
    # Add the pie chart for parciales won by the chosen team
    fig.add_trace(
        go.Pie(labels=['Victorias', 'Derrotas', 'Empates'],
            values=[sum(same_winner_counts), sum(different_winner_counts), sum(empate_counts)],
            marker=dict(colors=['#FFA500', '#00FFFF', '#FFB6C1']),
            hole=0.3,
            textinfo='label+value+percent',
            textposition='outside',
            textfont=dict(color='#ececec')),
        row=1, col=2
    )

    # Update layout for the entire figure
    fig.update_layout(
        title=dict(
            text=f'Rendimiento del de {chosen_team}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec'),  # Font color
        showlegend=False
    )
    puntos_reales = 2 * summary_table['Victoria'].sum() + summary_table['Empate'].sum()
    ratio_puntos_reales = puntos_reales / (2*(summary_table['Victoria'].sum() + summary_table['Empate'].sum() + summary_table['Derrota'].sum()))
    puntos_parciales = 2 * sum(same_winner_counts) + sum(empate_counts)
    ratio_puntos_parciales = puntos_parciales / (2*(sum(same_winner_counts) + sum(different_winner_counts) + sum(empate_counts)))
    performance_ratio = ratio_puntos_reales / ratio_puntos_parciales 

    # Text with performance_ratio under the plots

    fig.add_annotation(
        x=0.5,
        y=-0.1,
        xref='paper',
        yref='paper',
        text=f"Rendimiento: {performance_ratio:.2f}",
        showarrow=False,
        font=dict(color='#ececec', size = 16)
    )

    return fig


def get_analisis_tiempos_muertos(df, chosen_team):

    # Print the values of 'tiempo_muerto_1_my_team' column
    tiempos_muertos_1 = df['tiempo_muerto_1_my_team'].values
    tiempos_muertos_2 = df['tiempo_muerto_2_my_team'].values
    tiempos_muertos_3 = df['tiempo_muerto_3_my_team'].values

    # make a single list
    tiempos_muertos = []
    for tm in tiempos_muertos_1:
        tiempos_muertos.append(tm)
    for tm in tiempos_muertos_2:
        tiempos_muertos.append(tm)
    for tm in tiempos_muertos_3:
        tiempos_muertos.append(tm)

    # Convert each tm to the format 'minutos:segundos', trucate if its in the format 'minutos:segundos:centesimas'
    formatted_tiempos_muertos = []
    for tm in tiempos_muertos:
        if pd.isna(tm):
            continue
        if len(tm) > 5:
            tm = tm[:5]
        formatted_tiempos_muertos.append(tm)

    # Truncate only to the minute

    for i, tm in enumerate(formatted_tiempos_muertos):
        if pd.isna(tm):
            continue
        if len(tm) > 5:
            tm = tm[:5]
        minutos, segundos = tm.split(':')
        segundos = int(segundos)
        if segundos >= 30:
            minutos = int(minutos) 
        else:
            minutos = int(minutos)
        formatted_tiempos_muertos[i] = int(minutos)



    # Create a histogram with the formatted tiempos_muertos, range 0-60 5 by 5
    fig = go.Figure()
    # Add the histogram
    fig.add_trace(go.Histogram(x=formatted_tiempos_muertos, xbins=dict(start=0, end=num_parciales*5+5, size=5), 
                            marker_color='#FFA500', opacity=0.75))

    # Update layout for the entire figure
    fig.update_layout(
        title=dict(
            text=f'Tiempos Muertos de {chosen_team}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        xaxis=dict(title='Minutos', tickmode='linear', tick0=0, dtick=5, range=[0, 5 * num_parciales]),
        yaxis=dict(title='Cantidad de Tiempos Muertos'),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec'),  # Font color
        bargap=0.2  # Pad between columns
    )


    return fig

def get_maximos_goleadores(df, chosen_team):
    players_dict = {}

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Print the index and the 'nombre' column
        for i in range(1, 17):
            nombre = row[f'jugador_{i}_my_team_nombre']
            goles = row[f'jugador_{i}_my_team_goles']  
            # if name is NaN, skip
            if pd.isna(nombre):
                continue

            if nombre in players_dict:
                players_dict[nombre].append(goles)
            else:
                players_dict[nombre] = [goles]





    # Make a df. Players as index and sum(goles), mediana(goles), avg(goles), sd(goles) as values
    players_df = pd.DataFrame(columns=['goles_totales','goles_mediana'])

    for player, goles in players_dict.items():
        players_df.loc[player] = [sum(goles), pd.Series(goles).median()]

    # Sort the DataFrame by the 'goles_totales' column
    players_df = players_df.sort_values(by='goles_totales', ascending=False)

    # Create a bar chart with the top 10 players
    fig = go.Figure(data=go.Bar(
        x=players_df['goles_totales'].head(10),
        y=players_df.index[:10],
        name='Goles',
        orientation='h',
        marker_color='#FFA500',
        hovertext=players_df['goles_totales'].head(10),
        hoverinfo='text'
    ))

    # Add the text with the value 
    fig.update_traces(texttemplate='%{x}', textposition='outside')


    maximo_goles = players_df['goles_totales'].max()
    mediana_maxima = players_df['goles_mediana'].max()

    factor_escala = int(np.floor(maximo_goles / mediana_maxima / 5) * 5)

    factor_escala = 1 if factor_escala == 0 else factor_escala

    players_df['goles_mediana_escalado'] = factor_escala * players_df['goles_mediana']

    # Añadir la barra de la mediana
    fig.add_trace(go.Bar(
        x=players_df['goles_mediana_escalado'].head(10),
        y=players_df.index[:10],
        orientation='h',
        name=f'Mediana Goles (1:{factor_escala})',
        marker_color='#1f77b4',
        hovertext=players_df['goles_mediana'].head(10),
        hoverinfo='text'
    ))

    # Add the text with the value
    fig.update_traces(texttemplate='%{hovertext}', textposition='outside')



    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Máximos Goleadores de {chosen_team}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        xaxis=dict(title='Goles'),
        yaxis=dict(title='Jugador', autorange='reversed'),  # Reverse the y-axis to show most on top
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec')  # Font color
    )

    return fig

def get_maximos_infractores(df, chosen_team):
    players_dict = {}

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Print the index and the 'nombre' column
        for i in range(1, 17):
            nombre = row[f'jugador_{i}_my_team_nombre']
            amarillas = 1 if pd.notna(row[f'jugador_{i}_my_team_amarilla']) else 0
            exclusion1 = 1 if pd.notna(row[f'jugador_{i}_my_team_exclusion_1']) else 0
            exclusion2 = 1 if pd.notna(row[f'jugador_{i}_my_team_exclusion_2']) else 0
            roja = 1 if pd.notna(row[f'jugador_{i}_my_team_roja']) else 0
            azul = 1 if pd.notna(row[f'jugador_{i}_my_team_azul']) else 0
            total_exclusiones = exclusion1 + exclusion2 + roja

            # if name is NaN, skip
            if pd.isna(nombre):
                continue

            if nombre in players_dict:
                players_dict[nombre]['amarillas'].append(amarillas)
                players_dict[nombre]['exclusiones'].append(total_exclusiones)
                players_dict[nombre]['roja'].append(roja)
                players_dict[nombre]['azul'].append(azul)

            else:
                players_dict[nombre] = {'amarillas': [amarillas], 'exclusiones': [total_exclusiones], 'roja': [roja], 'azul': [azul]}

    # Make a df. Players as index and sum(amarillas), sum(exclusiones), sum(roja), sum(azul) as values

    players_df = pd.DataFrame(columns=['amarillas','exclusiones', 'roja', 'azul'])

    for player, values in players_dict.items():
        players_df.loc[player] = [sum(values['amarillas']), sum(values['exclusiones']), sum(values['roja']), sum(values['azul'])]

    # Sort the DataFrame by the 'exclusiones' column

    players_df = players_df.sort_values(by='exclusiones', ascending=False)

    # Add a row for the total of each column
    players_df.loc['TOTAL'] = players_df.sum()

    # Create a bar chart with the top 10 players
    fig = go.Figure(data=go.Bar(
        x=players_df['exclusiones'].head(10),
        y=players_df.index[:10],
        name='Exclusiones',
        orientation='h',
        marker_color='#FFA500',
        hovertext=players_df['exclusiones'].head(10),
        hoverinfo='text'
    ))

    # Add the text with the value
    fig.update_traces(texttemplate='%{x}', textposition='outside')


    # Crate a bar chart with the top 10 players with rojas
    fig.add_trace(go.Bar(
        x=players_df['roja'].head(10),
        y=players_df.index[:10],
        name='Rojas',
        orientation='h',
        marker_color='#1f77b4',
        hovertext=players_df['roja'].head(10),
        hoverinfo='text'
    ))

    # Add the text with the value
    fig.update_traces(texttemplate='%{x}', textposition='outside')


    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Máximos Infractores de {chosen_team}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        xaxis=dict(title='Exclusiones'),
        yaxis=dict(title='Jugador', autorange='reversed'),  # Reverse the y-axis to show most on top
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec')  # Font color
    )

    return fig




def get_gt_vs_dif(df, chosen_team):
    # Scatter plot with the goles totales (y) vs diferencia de goles (x)
    fig = go.Figure()

    # Add a line at x = 0 for reference

    # x = diferencia de goles, y = goles totales, color = local o visitante
    fig = px.scatter(df,
                     x='diferencia_goles',
                     y='goles_totales',
                     color='my_team_is',
                     hover_name='partido',
                     hover_data={'diferencia_goles': True, 'goles_totales': True},
                     title=f'Goles Totales vs Diferencia de Goles de {chosen_team}',
                     labels={'diferencia_goles': 'Diferencia de Goles', 'goles_totales': 'Goles Totales', 'my_team_is': 'Lugar'},
                     color_discrete_sequence=['#FFA500', '#00FFFF'],
                     opacity=0.75
                     )
    fig.update_traces(marker=dict(size=16))
    

    fig.add_shape(
        type='line',
        x0=0,
        y0=0,
        x1=0,
        y1=df['goles_totales'].max(),
        line=dict(color='rgba(255, 0, 0, 0.35)', width=2, dash='dash')
    )


    


        


    # Update layout

    fig.update_layout(
        title=dict(
            text=f'Goles Totales vs Diferencia de Goles de {chosen_team}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        xaxis=dict(title='Diferencia de Goles'),
        yaxis=dict(title='Goles Totales'),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec')  # Font color
    )
    


    return fig


def get_fig_rachas(df, chosen_team):

    # Obtener rachas:
    rachas_df = df[["partido","racha_my_team"]]

    transiciones = []
    for i in range(1, len(rachas_df)):
        if rachas_df.iloc[i-1, 1] != rachas_df.iloc[i, 1]:
            transiciones.append((rachas_df.iloc[i-1, 1], rachas_df.iloc[i, 1], rachas_df.iloc[i, 0]))
    # Crear un DataFrame con las transiciones
    transiciones_df = pd.DataFrame(transiciones, columns=['source', 'target', 'partido'])


    # Crear una cuenta de las transiciones
    transiciones_counts = transiciones_df.groupby(['source', 'target']).size().reset_index(name='count')

   
    # Agrupar las transiciones y registrar los partidos involucrados
    transiciones_counts = transiciones_df.groupby(['source', 'target']).agg(
        count=('partido', 'size'),
        partidos=('partido', lambda x: '<br>'.join(x))  # Usar \n para crear líneas nuevas
    ).reset_index()

    # Crear lista de nodos únicos
    unique_states = pd.unique(transiciones_counts[['source', 'target']].values.ravel())
    state_to_index = {state: i for i, state in enumerate(unique_states)}

    # Cast count to int

    transiciones_counts['source_index'] = transiciones_counts['source'].map(state_to_index)
    transiciones_counts['target_index'] = transiciones_counts['target'].map(state_to_index)

    transiciones_counts['count'] = transiciones_counts['count'].astype(int)
    
    
    # Crear el gráfico Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[str(state) for state in unique_states],
        ),
        link=dict(
            source=transiciones_counts['source_index'],
            target=transiciones_counts['target_index'],
            value=transiciones_counts['count'],
            customdata=transiciones_counts['partidos'],  # Datos personalizados para hover
            hovertemplate=(
                "<b style='font-size: 16px;'>De %{source.label} a %{target.label} (%{value:d} partidos)</b><br>"  # Texto de hover
                "%{customdata}<extra></extra>"  # Mostrar los partidos con salto de línea
            )
        )
    )])

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title=dict(
            text=f'Rachas de {chosen_team}',
            x=0.5,
            xanchor='center',
            y=0.9,
            yanchor='top'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec')
    )

    

    return fig

def get_fig_rachas_puntos(df, chosen_team):

    df["final_result"] = df["final_result"].apply(lambda x: x.split(" ")[0])
    rachas_df = df[["partido","final_result", "diferencia_goles", "puntos_ultimos_5_my_team"]]

    # Calcula la correlación de Pearson y Spearman
    pearson_corr, pearson_p = pearsonr(rachas_df['diferencia_goles'], rachas_df['puntos_ultimos_5_my_team'])
    spearman_corr, spearman_p = spearmanr(rachas_df['diferencia_goles'], rachas_df['puntos_ultimos_5_my_team'])

    # Ajustar la linea de una regresión lineal
    slope, intercept, r_value, p_value, std_err = linregress(rachas_df['diferencia_goles'], rachas_df['puntos_ultimos_5_my_team'])
    
    # Crear el scatter plot con diferencia_goles y puntos_ultimos_5_my_team
    fig = px.scatter(
        rachas_df,
        x='diferencia_goles',
        y='puntos_ultimos_5_my_team',
        color='final_result',
        title=f'Rachas de Puntos de {chosen_team}',
        labels={'diferencia_goles': 'Diferencia de Goles', 'puntos_ultimos_5_my_team': 'Puntos Últimos 5 Partidos', 'final_result': 'Resultado', 'partido': 'Partido'},
        hover_data={'partido': True}
    )

    # Pintar la regresión lineal
    fig.add_trace(go.Scatter(
        x=rachas_df['diferencia_goles'],
        y=intercept + slope * rachas_df['diferencia_goles'],
        mode='lines',
        name=f'Regresión Lineal (R²={r_value**2:.2f})',
        line=dict(color=f'rgba(255, 165, 0,{2* r_value**2:.2f})')
    ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title=dict(
            text=f'Rachas de Puntos y Resultados de {chosen_team}',
            x=0.5,
            xanchor='center',
            y=0.9,
            yanchor='top'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec')
    )

    # Actualizar la leyenda
    fig.update_traces(marker=dict(size=12))

    # Dibujar una línea en x = 0 para referencia
    maxima_racha = rachas_df['puntos_ultimos_5_my_team'].max()
    fig.add_shape(
        type='line',
        x0=0,
        y0=0,
        x1=0,
        y1=maxima_racha+1,
        line=dict(color='rgba(255, 0, 0, 0.35)', width=2, dash='dash')
    )






    return fig

def get_tabla_probabilidades_rachas(df, chosen_team):


    # Obtener probabiliadd total de victoria, empate y derrota
    total_victorias = df[df['resultado'] == 'Victoria'].shape[0]
    total_empates = df[df['resultado'] == 'Empate'].shape[0]
    total_derrotas = df[df['resultado'] == 'Derrota'].shape[0]

    probabilidad_victoria = round(total_victorias / df.shape[0] * 100, 1)
    probabilidad_empate = round(total_empates / df.shape[0] * 100, 1)
    probabilidad_derrota = round(total_derrotas / df.shape[0] * 100, 1)


    # Obtener rachas:
    rachas_df = df[["partido","racha_my_team"]]

    transiciones = []
    for i in range(1, len(rachas_df)):
        if rachas_df.iloc[i-1, 1] != rachas_df.iloc[i, 1]:
            transiciones.append((rachas_df.iloc[i-1, 1], rachas_df.iloc[i, 1], rachas_df.iloc[i, 0]))
    
    transiciones_df = pd.DataFrame(transiciones, columns=['source', 'target', 'partido'])

    # Cast source and target to int
    transiciones_df = transiciones_df.astype({'source': int, 'target': int})


        # Categorización de la columna 'target'
    conditions = [
        transiciones_df['target'] < 0,
        transiciones_df['target'] == 0,
        transiciones_df['target'] > 0
    ]
    choices = ['Derrota', 'Empate', 'Victoria']
    transiciones_df['target'] = np.select(conditions, choices, default='Unknown')
    sources = transiciones_df['source'].unique()

    # Crear un diccionario para almacenar las probabilidades de transición
    probabilidades = {}

    # Loop sobre las fuentes
    for source in sources:
        # Obtener las transiciones desde la fuente
        transiciones_source = transiciones_df[transiciones_df['source'] == source]
        # Contar las transiciones
        total_transiciones = transiciones_source.shape[0]
        # Contar las transiciones a cada destino
        conteo_transiciones = transiciones_source['target'].value_counts()
        # Calcular las probabilidades
        probabilidades[source] = (conteo_transiciones / total_transiciones *100).round(1)

    # Crear un DataFrame con las probabilidades
    probabilidades_df = pd.DataFrame(probabilidades).T.fillna(0)

    # Rename the columns in the order Victoria, Empate, Derrota
    probabilidades_df = probabilidades_df[['Victoria', 'Empate', 'Derrota']]

    # Sort by ascending index
    probabilidades_df = probabilidades_df.sort_index()

    # Add a new row total probabilities (probabilidad_victoria, probabilidad_empate, probabilidad_derrota)
    probabilidades_df.loc['Total'] = [probabilidad_victoria, probabilidad_empate, probabilidad_derrota]



    return probabilidades_df