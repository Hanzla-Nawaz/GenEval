import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

def plotly_visualizations(df, vis_choice, x_axis, y_axis, hue):
    if vis_choice == 'Bar Chart':
        fig = px.bar(df, x=x_axis, y=y_axis, color=hue, title="Bar Chart")
    elif vis_choice == 'Scatter Plot':
        fig = px.scatter(df, x=x_axis, y=y_axis, color=hue, title="Scatter Plot")
    elif vis_choice == 'Line Chart':
        fig = px.line(df, x=x_axis, y=y_axis, color=hue, title="Line Chart")
    elif vis_choice == 'Histogram':
        fig = px.histogram(df, x=x_axis, color=hue, title="Histogram")
    elif vis_choice == 'Box Plot':
        fig = px.box(df, x=x_axis, y=y_axis, color=hue, title="Box Plot")
    elif vis_choice == 'Heatmap':
        fig = go.Figure(data=go.Heatmap(
            z=df.corr().values,
            x=df.columns,
            y=df.columns,
            colorscale='Viridis'))
        fig.update_layout(title="Heatmap")
    elif vis_choice == 'Pie Chart':
        fig = px.pie(df, names=x_axis, title="Pie Chart")
    elif vis_choice == 'Violin Plot':
        fig = px.violin(df, y=y_axis, x=x_axis, color=hue, title="Violin Plot")
    else:
        st.warning("Unsupported visualization type selected.")
        return None
    
    fig.update_layout(template="plotly_white")
    return fig

def altair_visualizations(df, vis_choice, x_axis, y_axis, hue):
    if vis_choice == 'Bar Chart':
        chart = alt.Chart(df).mark_bar().encode(x=x_axis, y=y_axis, color=hue)
    elif vis_choice == 'Scatter Plot':
        chart = alt.Chart(df).mark_point().encode(x=x_axis, y=y_axis, color=hue)
    elif vis_choice == 'Line Chart':
        chart = alt.Chart(df).mark_line().encode(x=x_axis, y=y_axis, color=hue)
    elif vis_choice == 'Histogram':
        chart = alt.Chart(df).mark_bar().encode(alt.X(x_axis, bin=True), y='count()', color=hue)
    elif vis_choice == 'Box Plot':
        chart = alt.Chart(df).mark_boxplot().encode(x=x_axis, y=y_axis, color=hue)
    else:
        st.warning("Unsupported visualization type selected.")
        return None
    
    return chart.interactive()

def seaborn_visualizations(df, vis_choice, x_axis, y_axis, hue):
    plt.figure(figsize=(10, 6))
    if vis_choice == 'Bar Chart':
        sns.barplot(data=df, x=x_axis, y=y_axis, hue=hue)
    elif vis_choice == 'Scatter Plot':
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue)
    elif vis_choice == 'Line Chart':
        sns.lineplot(data=df, x=x_axis, y=y_axis, hue=hue)
    elif vis_choice == 'Histogram':
        sns.histplot(data=df, x=x_axis, hue=hue, kde=True)
    elif vis_choice == 'Box Plot':
        sns.boxplot(data=df, x=x_axis, y=y_axis, hue=hue)
    elif vis_choice == 'Heatmap':
        sns.heatmap(df.corr(), annot=True, cmap='viridis')
    elif vis_choice == 'Pie Chart':
        pie_data = df[x_axis].value_counts()
        plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
    else:
        st.warning("Unsupported visualization type selected.")
        return None
    
    st.pyplot(plt.gcf())