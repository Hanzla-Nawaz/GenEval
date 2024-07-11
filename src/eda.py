# src/eda/eda.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


# Define EDA function
def perform_eda(df, choice):
    try:
        if choice == "Summary Statistics":
            st.write(df.describe())
        elif choice == "Correlation Matrix":
            st.write(df.corr())
        elif choice == "Pairplot":
            fig = sns.pairplot(df)
            st.pyplot(fig)
        elif choice == "Heatmap":
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    except Exception as e:
        logger.error(f"Error performing EDA: {e}")
        st.error(f"An error occurred while performing EDA: {e}")