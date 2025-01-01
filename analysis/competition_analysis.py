import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu, ks_2samp
import numpy as np



def run_analysis():
    st.header("Análisis de Competición")

    files = [f for f in os.listdir("data/partidos/") if f.endswith(".xlsx")]

    if not files:
        st.error("No hay archivos disponibles en el directorio 'data/equipos/'.")
        st.stop()

    st.write("Selecciona una competición a analizar:")

    selected_competition = st.selectbox("Competición:", files)

    if not selected_competition:
        st.warning("Selecciona una competición para continuar.")
        st.stop()

    nombre_competicion = selected_competition.split('_')[1].split('.')[0]

    # Cargamos el df de partidos
    df = pd.read_excel(f"data/partidos/{selected_competition}")
    df['partido'] = df['nombre_local'] + ' vs ' + df['nombre_visitante']


    st.write("Número de partidos:", len(df))

    st.divider()

    tabs = st.tabs(["Partidos", "Flujo de resultados", "Análisis de exclusiones", "Análisis de jugadores"])



    # Tab 1: Partidos
    with tabs[0]:
        fig_resultados = get_fig_resultados(df, nombre_competicion)

        st.plotly_chart(fig_resultados)

        fig_gt_diferencia = get_fig_gt_diferencia(df, nombre_competicion)

        st.plotly_chart(fig_gt_diferencia)

        fig_dist_diferencia = get_dist_diferencia(df, nombre_competicion)

        st.plotly_chart(fig_dist_diferencia)

        fig_dist_diferencia_absoluta = get_dist_diferencia_absoluta(df, nombre_competicion)

        st.plotly_chart(fig_dist_diferencia_absoluta)


    with tabs[1]:
        # Tab 2: Flujo de resultados
        fig_flujo_min5 = get_fig_flujo_min5(df, nombre_competicion)
        st.plotly_chart(fig_flujo_min5)

        fig_flujo_min30 = get_fig_flujo_min30(df, nombre_competicion)

        st.plotly_chart(fig_flujo_min30)

        fig_flujo_min50 = get_fig_flujo_min50(df, nombre_competicion)

        st.plotly_chart(fig_flujo_min50)

    # Tab 3: Análisis de exclusiones
    with tabs[2]:

        fig_dist_exclusiones = get_dist_exclusiones_local_visitante(df, nombre_competicion)

        st.plotly_chart(fig_dist_exclusiones)

        fig_dist_exclusiones_ganador_perdedor = get_dist_exclusiones_ganador_perdedor(df, nombre_competicion)

        st.plotly_chart(fig_dist_exclusiones_ganador_perdedor)

        fig_equipos_mas_excluidos = get_equipos_mas_excluidos(df, nombre_competicion)

        st.plotly_chart(fig_equipos_mas_excluidos)

    # Tab 4: Análisis de jugadores

    with tabs[3]:
        fig_top_goleadores = get_maximo_goleador_partido(df, nombre_competicion)

        st.plotly_chart(fig_top_goleadores)

        fig_top_goleadores_total = get_maximo_goleador_total(df, nombre_competicion)

        st.plotly_chart(fig_top_goleadores_total)

        fig_top_excluidos = get_maximo_excluido(df, nombre_competicion)

        st.plotly_chart(fig_top_excluidos)






def get_fig_resultados(df, nombre_competicion):
    color = ['#FFA500', '#00FFFF', '#FFD700']
    color_map = {'Local': "#FFA500", 'Visitante': "#00FFFF", 'Empate': "#FFD700"}
    fig = px.histogram(df, x='ganador_partido', title='Histogram of Ganador Partido', color='ganador_partido', color_discrete_map=color_map, category_orders={'ganador_partido': ['Local', 'Empate', 'Visitante']})
    fig.update_layout(
        title=dict(
            text=f'Distribución de Resultados de {nombre_competicion}',
            x=0.5,  # Position the title on the right
            xanchor='center',  # Anchor the title to the right
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        xaxis_title='Resultado',
        yaxis_title='Cantidad',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec'),  # Font color
        showlegend=False  # Remove legend
    )

    return fig


def get_fig_gt_diferencia(df, nombre_competicion):

    # Count the occurrences of each point
    df['count'] = df.groupby(['diferencia_goles_absoluta', 'goles_totales'])['goles_totales'].transform('count')
    # delete NaN values in the count column
    df = df.dropna(subset=['count'])
    
    print(df["count"])


    # Create the scatter plot with size based on the count
    fig_scatter = px.scatter(df, x='diferencia_goles_absoluta', y='goles_totales', 
                            size='count',
                            title='Scatter Plot of Goles Totales vs Diferencia Goles Absoluta',
                            color_discrete_sequence=['#f4a800'])
    fig_scatter.update_traces(marker=dict(line=dict(width=0)))  # Remove stroke

    fig_scatter.update_layout(
        title=dict(
            text=f'Goles Totales vs Diferencia Goles Absoluta de {nombre_competicion}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        xaxis_title='Diferencia Goles Absoluta',
        yaxis_title='Goles Totales',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec')  # Font color
    )
    avg_dif = df['diferencia_goles_absoluta'].mean()
    std_dif = df['diferencia_goles_absoluta'].std()
    avg_goles_totales = df['goles_totales'].mean()
    std_goles_totales = df['goles_totales'].std()


    # BEGIN: Add average lines
    fig_scatter.add_shape(
        type='line',
        x0=avg_dif, y0=min(df['goles_totales']),
        x1=avg_dif, y1=max(df['goles_totales'])+3,
        line=dict(color='Red', width=2, dash='dash')
    )

    fig_scatter.add_shape(
        type='line',
        x0=min(df['diferencia_goles_absoluta'])-1, y0=avg_goles_totales,
        x1=max(df['diferencia_goles_absoluta'])+1, y1=avg_goles_totales,
        line=dict(color='Red', width=2, dash='dash')
    )

    # Añadir un cuadro de texto en el gráfico con las medias y desviaciones estándar
    fig_scatter.add_annotation(
        text=(
            f"<b>Estadísticas:</b><br>"
            f"Media Goles Totales: {avg_goles_totales:.2f} ± {std_goles_totales:.2f}<br>"
            f"Media Dif. Goles Abs: {avg_dif:.2f} ± {std_dif:.2f}<br>"
        ),
        xref='paper', yref='paper',  # Referencia a coordenadas del papel (en vez de los datos)
        x=1 ,y=0,  # Posición del cuadro de texto (fuera del gráfico)
        showarrow=False,
        bordercolor='black',
        borderwidth=1,
        borderpad=10,
        bgcolor='rgba(255, 255, 255, 0.7)',  # Fondo semitransparente
        font=dict(color='black', size=12)  
    )

    
    df['partido'] = df['nombre_local'] + ' vs ' + df['nombre_visitante']
    # Add partido name (local y visitante)when hover
    fig_scatter.update_traces(
        hovertemplate='<b>Partido:</b> %{customdata[0]}<br>'
                    '<b>Diferencia Goles Absoluta:</b> %{x}<br>'
                    '<b>Goles Totales:</b> %{y}<br>'
                    '<b>Count:</b> %{marker.size}<extra></extra>',
        customdata=df[['partido']]
    )

    return fig_scatter


def get_dist_diferencia(df, nombre_cmpeticion):


    fig_box = px.box(df, y='diferencia_goles', title='Box Plot of Diferencia Goles', color_discrete_sequence=['#FFA500'])
    


    fig_box.update_layout(
        title=dict(
            text=f'Distribución de Diferencia de Goles de {nombre_cmpeticion}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        yaxis_title='Diferencia de Goles',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec')  # Font color
    )

    # Add hover info with the partido name
    fig_box.update_traces(
        hovertemplate='<b>Partido:</b> %{customdata[0]}<br>'
                    '<b>Diferencia Goles:</b> %{y}<extra></extra>',
        customdata=df[['partido']]
    )

    return fig_box

def get_dist_diferencia_absoluta(df, nombre_cmpeticion):


    fig_box = px.box(df, y='diferencia_goles_absoluta', title='Box Plot of Diferencia Goles Absoluta', color_discrete_sequence=['#FFA500'])
    


    fig_box.update_layout(
        title=dict(
            text=f'Distribución de Diferencia de Goles Absoluta de {nombre_cmpeticion}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top'  # Anchor the title to the top
        ),
        yaxis_title='Diferencia de Goles Absoluta',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec')  # Font color
    )

    # Add hover info with the partido name
    fig_box.update_traces(
        hovertemplate='<b>Partido:</b> %{customdata[0]}<br>'
                    '<b>Diferencia Goles Absoluta:</b> %{y}<extra></extra>',
        customdata=df[['partido']]
    )

    return fig_box

def get_fig_flujo_min30(df, nombre_competicion):
    umbral = 3

    # Add column 'resultado_locales_primera_parte' to the DataFrame
    df['resultado_locales_primera_parte'] = 'Empate'
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        diferencia_primera_parte = row['parcial_6_local'] - row['parcial_6_visitante']
        resultado = f'Local > {umbral}' if diferencia_primera_parte > umbral else f'Visitante > {umbral}' if diferencia_primera_parte < -umbral else f'Empate [-{umbral}, {umbral}]'
        df.at[index, 'resultado_locales_primera_parte'] = resultado

    # Create a Sankey plot with the winner at the first partial (ganador_parcial_1) and the final winner (ganador_partido)
    sankey_data = df.groupby(['resultado_locales_primera_parte', 'ganador_partido']).size().reset_index(name='count')

    # Merge the partido information for hover display
    sankey_data = sankey_data.merge(df[['resultado_locales_primera_parte', 'ganador_partido', 'partido']],
                                    on=['resultado_locales_primera_parte', 'ganador_partido'],
                                    how='left')

    # Define the nodes and links for the Sankey diagram
    nodes = list(set(sankey_data['resultado_locales_primera_parte']).union(set(sankey_data['ganador_partido'])))
    node_indices = {node: i for i, node in enumerate(nodes)}

    links = {
        'source': [node_indices[resultado_locales_primera_parte] for resultado_locales_primera_parte in sankey_data['resultado_locales_primera_parte']],
        'target': [node_indices[ganador_partido] for ganador_partido in sankey_data['ganador_partido']],
        'value': [1] * len(sankey_data),
        'customdata': sankey_data['partido'],  # Add partido names for hover
        'color': 'rgba(255, 255, 255, 0.8)'
    }

    # Create the Sankey diagram
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=40,
            line=dict(color='black', width=0.5),
            label=nodes
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            customdata=links['customdata'],
            hovertemplate='Partido: %{customdata}',  # Custom hover text
        )
    )])

    fig_sankey.update_layout(
        title_text='Flujo de resultados entre el primer tiempo y el resultado final',
        font=dict(size=16, color='#8b0000'),
        title_font=dict(size=16, color='#ececec'),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)'    # Transparent background for the plot
    )

    return fig_sankey

def get_fig_flujo_min5(df, nombre_competicion):

    umbral = 1

    # Add column 'resultado_locales_primera_parte' to the DataFrame
    df['resultado_locales_primera_parte'] = 'Empate'
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        diferencia_primera_parte = row['parcial_1_local'] - row['parcial_1_visitante']
        resultado = f'Local > {umbral}' if diferencia_primera_parte > umbral else f'Visitante > {umbral}' if diferencia_primera_parte < -umbral else f'Empate [-{umbral}, {umbral}]'
        df.at[index, 'resultado_locales_primera_parte'] = resultado

    # Create a Sankey plot with the winner at the first partial (ganador_parcial_1) and the final winner (ganador_partido)
    sankey_data = df.groupby(['resultado_locales_primera_parte', 'ganador_partido']).size().reset_index(name='count')

    # Merge the partido information for hover display
    sankey_data = sankey_data.merge(df[['resultado_locales_primera_parte', 'ganador_partido', 'partido']],
                                    on=['resultado_locales_primera_parte', 'ganador_partido'],
                                    how='left')

    # Define the nodes and links for the Sankey diagram
    nodes = list(set(sankey_data['resultado_locales_primera_parte']).union(set(sankey_data['ganador_partido'])))
    node_indices = {node: i for i, node in enumerate(nodes)}

    links = {
        'source': [node_indices[resultado_locales_primera_parte] for resultado_locales_primera_parte in sankey_data['resultado_locales_primera_parte']],
        'target': [node_indices[ganador_partido] for ganador_partido in sankey_data['ganador_partido']],
        'value': [1] * len(sankey_data),
        'customdata': sankey_data['partido'],  # Add partido names for hover
        'color': 'rgba(255, 255, 255, 0.8)'
    }

    # Create the Sankey diagram
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=40,
            line=dict(color='black', width=0.5),
            label=nodes
        ),

        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            customdata=links['customdata'],
            hovertemplate='Partido: %{customdata}',  # Custom hover text
        )

    )])

    fig_sankey.update_layout(
        title_text=f'Flujo de resultados tras 5 minutos y el resultado final de {nombre_competicion}',
        font=dict(size=16, color='#8b0000'),
        title_font=dict(size=16, color='#ececec'),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)'    # Transparent background for the plot
    )


    return fig_sankey

def get_fig_flujo_min50(df, nombre_competicion):
    
    umbral = 2

    # Add column 'resultado_locales_primera_parte' to the DataFrame
    df['resultado_locales_primera_parte'] = 'Empate'
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        diferencia_primera_parte = row['parcial_3_local'] - row['parcial_3_visitante']
        resultado = f'Local > {umbral}' if diferencia_primera_parte > umbral else f'Visitante > {umbral}' if diferencia_primera_parte < -umbral else f'Empate [-{umbral}, {umbral}]'
        df.at[index, 'resultado_locales_primera_parte'] = resultado

    # Create a Sankey plot with the winner at the first partial (ganador_parcial_1) and the final winner (ganador_partido)
    sankey_data = df.groupby(['resultado_locales_primera_parte', 'ganador_partido']).size().reset_index(name='count')

    # Merge the partido information for hover display
    sankey_data = sankey_data.merge(df[['resultado_locales_primera_parte', 'ganador_partido', 'partido']],
                                    on=['resultado_locales_primera_parte', 'ganador_partido'],
                                    how='left')

    # Define the nodes and links for the Sankey diagram
    nodes = list(set(sankey_data['resultado_locales_primera_parte']).union(set(sankey_data['ganador_partido'])))
    node_indices = {node: i for i, node in enumerate(nodes)}

    links = {
        'source': [node_indices[resultado_locales_primera_parte] for resultado_locales_primera_parte in sankey_data['resultado_locales_primera_parte']],
        'target': [node_indices[ganador_partido] for ganador_partido in sankey_data['ganador_partido']],
        'value': [1] * len(sankey_data),
        'customdata': sankey_data['partido'],  # Add partido names for hover
        'color': 'rgba(255, 255, 255, 0.8)'
    }

    # Create the Sankey diagram
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=40,
            line=dict(color='black', width=0.5),
            label=nodes
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            customdata=links['customdata'],
            hovertemplate='Partido: %{customdata}',  # Custom hover text
        )

    )])

    fig_sankey.update_layout(
        title_text=f'Flujo de resultados entre los últimos 10 minutos y el resultado final de {nombre_competicion}',
        font=dict(size=16, color='#8b0000'),
        title_font=dict(size=16, color='#ececec'),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)'    # Transparent background for the plot
    )

    return fig_sankey


def get_dist_exclusiones_local_visitante(df, nombre_competicion):
    # Create a pie chart for exclusiones_local and exclusiones_visitante
    exclusiones_data = df[['exclusiones_local', 'exclusiones_visitante']].sum()
    labels = ['Exclusiones Local', 'Exclusiones Visitante']
    values = [exclusiones_data['exclusiones_local'], exclusiones_data['exclusiones_visitante']]
    colors = ['#FFA500', '#00FFFF']

    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors), hole=0.3, textinfo='label+value+percent',insidetextorientation='radial', textfont=dict(color='#ececec'))])
    fig_pie.update_layout(
        title=dict(
            text='Distribución de Exclusiones Local y Visitante',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top',  # Anchor the title to the top
            font=dict(color='#ececec', family="Arial", size=20)  # Color del texto en amarillo y en negrita

        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec', size = 16)  # Font color
    )

    mannwhitneyu_result = mannwhitneyu(df['exclusiones_local'], df['exclusiones_visitante'])
    ks_result = ks_2samp(df['exclusiones_local'], df['exclusiones_visitante'])

    fig_pie.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.2,  # Position the annotation in the center
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Mann-Whitney U test: p-value={mannwhitneyu_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    fig_pie.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.13,  # Position the annotation in the center
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Kolmogorov-Smirnov test: p-value={ks_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    return fig_pie

def get_dist_exclusiones_ganador_perdedor(df, nombre_competicion):

    # Create a pie chart for exclusiones_ganador and exclusiones_perdedor
    exclusiones_ganador_perdedor_data = df[['exclusiones_ganador', 'exclusiones_perdedor']].sum()
    labels_ganador_perdedor = ['Exclusiones Ganador', 'Exclusiones Perdedor']
    values_ganador_perdedor = [exclusiones_ganador_perdedor_data['exclusiones_ganador'], exclusiones_ganador_perdedor_data['exclusiones_perdedor']]
    colors_ganador_perdedor = ['#FFA500', '#00FFFF']

    fig_pie_ganador_perdedor = go.Figure(data=[go.Pie(labels=labels_ganador_perdedor, values=values_ganador_perdedor, marker=dict(colors=colors_ganador_perdedor), hole=0.3, textinfo='label+value+percent',insidetextorientation='radial', textfont=dict(color='#ececec'))])
    fig_pie_ganador_perdedor.update_layout(
        title=dict(
            text='Distribución de Exclusiones Ganador y Perdedor',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top',  # Anchor the title to the top
            font=dict(color='#ececec', family="Arial", size=20)  # Color del texto en amarillo y en negrita
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec', size = 16)  # Font color
    )

    # Remove NaN values from the columns before performing the test
    df_clean = df[['exclusiones_ganador', 'exclusiones_perdedor']].dropna()

    mannwhitneyu_result = mannwhitneyu(df_clean['exclusiones_ganador'], df_clean['exclusiones_perdedor'])
    ks_result = ks_2samp(df_clean['exclusiones_ganador'], df_clean['exclusiones_perdedor'])
    # Add the Mann-Whitney U test result to the plot
    fig_pie_ganador_perdedor.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.2,  # Position the annotation in the center
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Mann-Whitney U test: p-value={mannwhitneyu_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    # Add the Kolmogorov-Smirnov test result to the plot
    fig_pie_ganador_perdedor.add_annotation(
        x=0.5,  # Position the annotation in the center
        y=-0.13,  # Position the annotation in the center
        xref='paper',  # Reference the x position to the paper
        yref='paper',  # Reference the y position to the paper
        text=f'Kolmogorov-Smirnov test: p-value={ks_result.pvalue:.4f}',
        showarrow=False,  # Do not show an arrow
        font=dict(color='#ececec', size = 14)  # Font color
    )

    return fig_pie_ganador_perdedor


def get_maximo_goleador_partido(df, nombre_competicion):

    # Extract columns related to player goals
    local_goles_columns = [col for col in df.columns if 'jugador_' in col and '_local_goles' in col]
    visitante_goles_columns = [col for col in df.columns if 'jugador_' in col and '_visitante_goles' in col]

    # Combine both lists
    all_goles_columns = local_goles_columns + visitante_goles_columns

    # display(all_goles_columns)


    # Create a list to store the players and their goals
    top_players = []

    # Iterate over all rows in the DataFrame
    # Create a list to store the players and their goals
    top_players = []

    # Iterate over all rows in the DataFrame
    for index, row in df.iterrows():
        # Iterate over the columns with player goals
        for col in all_goles_columns:
            # Get the player name and goals
            team = row['nombre_local'] if col in local_goles_columns else row['nombre_visitante']
            rival = row['nombre_visitante'] if col in local_goles_columns else row['nombre_local']
            player_name = f"{row[col.replace('_goles', '_nombre')]} vs {rival}"
            goals = row[col]
            partido = row['partido']
            # Append the player and their goals to the list
            top_players.append((partido, player_name, goals, team))

    # Create a DataFrame from the top_players list
    top_players_df = pd.DataFrame(top_players, columns=['Acta ID', 'Player Name', 'Goals', 'Team'])


    # Create a DataFrame from the top_players list
    top_players_df = pd.DataFrame(top_players, columns=['Acta ID', 'Player Name', 'Goals', 'Team'])

    # Sort the DataFrame by the number of goals in descending order
    top_players_df = top_players_df.sort_values('Goals', ascending=False)

    # Get the top 10 players
    top_10_players = top_players_df.head(10)

    top_10_players = top_10_players.sort_values('Goals', ascending=False)


    # Create a bar plot with the top 10 players
    fig_top_players = px.bar(
        top_10_players, 
        x='Goals', 
        y='Player Name', 
        color='Team', 
        orientation='h', 
        title=f'Top 10 goleadores de {nombre_competicion}',
        category_orders={"Player Name": top_10_players['Player Name'].tolist()}
    )

    fig_top_players.update_layout(
        title=dict(
            text=f'Top 10 goleadores de {nombre_competicion} en un partido',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top',  # Anchor the title to the top
            font=dict(color='#ececec', family="Arial", size=20)  # Color del texto en amarillo y en negrita
        ),
        xaxis_title='Goles',
        yaxis_title='Jugador',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec', size = 16)  # Font color
    )
    return fig_top_players

def get_maximo_goleador_total(df, nombre_competicion):
    
    # Extract columns related to player goals
    local_goles_columns = [col for col in df.columns if 'jugador_' in col and '_local_goles' in col]
    visitante_goles_columns = [col for col in df.columns if 'jugador_' in col and '_visitante_goles' in col]

    # Combine both lists
    all_goles_columns = local_goles_columns + visitante_goles_columns

    # Create a list to store the players and their goals
    top_players = []

    # Iterate over all rows in the DataFrame
    for index, row in df.iterrows():
        # Iterate over the columns with player goals
        for col in all_goles_columns:
            # Get the player name and goals
            team = row['nombre_local'] if col in local_goles_columns else row['nombre_visitante']
            player_name = row[col.replace('_goles', '_nombre')]
            goals = row[col]
            # Append the player and their goals to the list
            top_players.append((player_name, goals, team))

    # Create a DataFrame from the top_players list
    top_players_df = pd.DataFrame(top_players, columns=['Player Name', 'Goals', 'Team'])

    # Join the DataFrame with the player names and calculate the sum and median of goals
    top_players_df = top_players_df.groupby(['Player Name', 'Team']).agg({'Goals': ['sum', 'median']}).reset_index()
    top_players_df.columns = ['Player Name', 'Team', 'Goals', 'Median Goals']


    # Sort the DataFrame by the number of goals in descending order
    top_players_df = top_players_df.sort_values('Goals', ascending=False)


    # Reset the index of the DataFrame
    top_players_df = top_players_df.reset_index(drop=True)
    print(top_players_df.head())

    # Get the top 10 players
    top_10_players = top_players_df.head(10)

    top_10_players = top_10_players.sort_values('Goals', ascending=False)
    top_10_players['Total Goles'] = top_10_players.apply(lambda x: get_goles_equipo(df, x['Team']), axis=1)
    top_10_players['Percentage Goals'] = top_10_players['Goals'] / top_10_players['Total Goles'] * 100
    print(top_10_players.head())

    # Create a bar plot with the top 10 players
    fig_top_players = px.bar(
        top_10_players, 
        x='Goals', 
        y='Player Name', 
        color='Team', 
        orientation='h', 
        title=f'Top 10 goleadores de {nombre_competicion}',
        category_orders={"Player Name": top_10_players['Player Name'].tolist()}  # Ordenar según los nombres en el DataFrame ordenado
    )



    fig_top_players.update_layout(
        title=dict(
            text=f'Top 10 goleadores de {nombre_competicion} en total',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top',  # Anchor the title to the top
            font=dict(color='#ececec', family="Arial", size=20)  # Color del texto en amarillo y en negrita
        ),
        xaxis_title='Goles',
        yaxis_title='Jugador',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec', size = 16)  # Font color
    )

    # Add hover info with the percentage of goals
    fig_top_players.update_traces(
        hovertemplate='<b>Jugador:</b> %{y}<br>'
                      '<b>Goles:</b> %{x}<br>'
                      '<b>Porcentaje de Goles:</b> %{customdata[0]:.2f}%<extra></extra>',
        customdata=top_10_players[['Percentage Goals']]
    )


 


    
    
    return fig_top_players

def get_goles_equipo(df, nombre_equipo) -> int:

    # Extract partidos where nombre_local == nombre_equipo 
    partidos_local = df[df['nombre_local'] == nombre_equipo]
    # Extract partidos where nombre_visitante == nombre_equipo
    partidos_visitante = df[df['nombre_visitante'] == nombre_equipo]

    # Calculate the total number of goals scored by the team, use goles_local and goles_visitante columns
    goles_local = partidos_local['goles_local'].sum()
    goles_visitante = partidos_visitante['goles_visitante'].sum()

    # Calculate the total number of goals scored by the team
    total_goles = goles_local + goles_visitante

    return total_goles
    


def get_maximo_excluido(df, nombre_competicion):

    exclusiones_local_columns = [col for col in df.columns if 'jugador_' in col and '_local_exclusion' in col] + [col for col in df.columns if 'jugador_' in col and '_local_roja' in col] + [col for col in df.columns if 'jugador_' in col and '_local_azul' in col]

    exclusiones_visitante_columns = [col for col in df.columns if 'jugador_' in col and '_visitante_exclusion' in col] + [col for col in df.columns if 'jugador_' in col and '_visitante_roja' in col] + [col for col in df.columns if 'jugador_' in col and '_visitante_azul' in col]

    # Combine both lists
    all_exclusiones_columns = exclusiones_local_columns + exclusiones_visitante_columns

    # Create a list to store the players and their goals
    top_players_exclusiones = []

    # Iterate over all rows in the DataFrame

    for index, row in df.iterrows():
        equipo_local = row['nombre_local']
        equipo_visitante = row['nombre_visitante']
        actaid = row['actaid']
        # Iterate over the columns with player goals
        for i in range (1, 17):
            exclusiones_jugador_local = (row[f'jugador_{i}_local_exclusion_1'], row[f'jugador_{i}_local_exclusion_2'], row[f'jugador_{i}_local_roja'], row[f'jugador_{i}_local_azul'])
            exclusiones_jugador_visitante = (row[f'jugador_{i}_visitante_exclusion_1'], row[f'jugador_{i}_visitante_exclusion_2'], row[f'jugador_{i}_visitante_roja'], row[f'jugador_{i}_visitante_azul'])
            total_exclusiones_jugador_local = sum([1 for exclusion in exclusiones_jugador_local if not pd.isna(exclusion)])
            total_exclusiones_jugador_visitante = sum([1 for exclusion in exclusiones_jugador_visitante if not pd.isna(exclusion)])
            nombre_jugador_local = row[f'jugador_{i}_local_nombre']
            nombre_jugador_visitante = row[f'jugador_{i}_visitante_nombre']
            # Append the player and their goals to the list
            top_players_exclusiones.append((actaid, nombre_jugador_local, total_exclusiones_jugador_local, equipo_local))
            top_players_exclusiones.append((actaid, nombre_jugador_visitante, total_exclusiones_jugador_visitante, equipo_visitante))

    # Create a DataFrame from the top_players list
    top_players_exclusiones_df = pd.DataFrame(top_players_exclusiones, columns=['Acta ID', 'Player Name', 'Exclusiones', 'Team'])

    # Join the DataFrame with the player names
    top_players_exclusiones_df = top_players_exclusiones_df.groupby(['Player Name', 'Team'])['Exclusiones'].sum().reset_index()

    # Sort the DataFrame by the number of goals in descending order
    top_players_exclusiones_df = top_players_exclusiones_df.sort_values('Exclusiones', ascending=False)

    # Reset the index of the DataFrame
    top_players_exclusiones_df = top_players_exclusiones_df.reset_index(drop=True)

    # Get the top 10 players
    top_10_players_exclusiones = top_players_exclusiones_df.head(10)

    top_10_players_exclusiones = top_10_players_exclusiones.sort_values('Exclusiones', ascending=True)

    # Create a bar plot with the top 10 players
    fig_top_players_exclusiones = px.bar(top_10_players_exclusiones, x='Exclusiones', y='Player Name', color='Team', orientation='h', title=f'Top 10 jugadores más excluidos de {nombre_competicion}')

    fig_top_players_exclusiones.update_layout(
        title=dict(
            text=f'Top 10 jugadores más excluidos de {nombre_competicion}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top',  # Anchor the title to the top
            font=dict(color='#ececec', family="Arial", size=20)  # Color del texto en amarillo y en negrita
        ),
        xaxis_title='Exclusiones',
        yaxis_title='Jugador',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec', size = 16)  # Font color
    )

    return fig_top_players_exclusiones

def get_equipos_mas_excluidos (df, nombre_competicion):
    # Create a DataFrame with columns Team, Exclusiones using the specified columns
    exclusiones_df_local = df[['nombre_local','exclusiones_local']].copy()
    exclusiones_df_visitante = df[['nombre_visitante','exclusiones_visitante']].copy()

    # Rename the columns
    exclusiones_df_local.columns = ['Team', 'Exclusiones']
    exclusiones_df_visitante.columns = ['Team', 'Exclusiones']

    # Concatenate the local and visitante DataFrames
    exclusiones_df = pd.concat([exclusiones_df_local, exclusiones_df_visitante])

    # Create a DataFrame with local_nombre, visitante_nombre and the sum of exclusiones_local and exclusiones_visitante
    exclusiones_partidos = df[['nombre_local', 'nombre_visitante', 'exclusiones_local', 'exclusiones_visitante']].copy()
    exclusiones_partidos['total_exclusiones'] = exclusiones_partidos['exclusiones_local'] + exclusiones_partidos['exclusiones_visitante']

    # Sort the DataFrame by total_exclusiones in descending order
    exclusiones_partidos = exclusiones_partidos.sort_values('total_exclusiones', ascending=False)

    # Rename the columns
    exclusiones_df.columns = ['Team', 'Exclusiones']

    # Group by Team and sum the Exclusiones
    exclusiones_df = exclusiones_df.groupby('Team').sum()

    # Sort the DataFrame by Exclusiones in descending order
    exclusiones_df = exclusiones_df.sort_values('Exclusiones', ascending=False)

    # Reset the index of the DataFrame
    exclusiones_df = exclusiones_df.reset_index()

    # Get the top 10 teams
    top_10_exclusiones = exclusiones_df.head(10)

    top_10_exclusiones = top_10_exclusiones.sort_values('Exclusiones', ascending=True)

    # Create a bar plot with the top 10 teams

    fig_top_exclusiones = px.bar(top_10_exclusiones, x='Exclusiones', y='Team', orientation='h', title=f'Top 10 equipos más excluidos de {nombre_competicion}')

    fig_top_exclusiones.update_layout(
        title=dict(
            text=f'Top 10 equipos más excluidos de {nombre_competicion}',
            x=0.5,  # Position the title in the center
            xanchor='center',  # Anchor the title to the center
            y=0.9,  # Position the title at the top
            yanchor='top',  # Anchor the title to the top
            font=dict(color='#ececec', family="Arial", size=20)  # Color del texto en amarillo y en negrita
        ),
        xaxis_title='Exclusiones',
        yaxis_title='Equipo',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent background for the plot
        font=dict(color='#ececec', size = 16)  # Font color
    )

    return fig_top_exclusiones


