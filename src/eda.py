import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from scipy.stats import skew, kurtosis
import io
import streamlit as st

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dataset Overview
def dataset_overview(df):
    """Generates an overview of the dataset."""
    logger.info("Generating dataset overview...")
    overview = {
        "Shape": df.shape,
        "Data Types": df.dtypes.to_dict(),
        "Missing Values": df.isnull().sum().to_dict(),
        "Duplicate Rows": df.duplicated().sum(),
        "Memory Usage (MB)": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
    }
    logger.info(f"Dataset Overview: {overview}")
    return overview

# Numerical Feature Analysis
def analyze_numerical(df):
    """Performs statistical analysis on numerical features and visualizes their distributions."""
    logger.info("Analyzing numerical features...")
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        logger.warning("No numerical columns found.")
        return None, None

    summary = numeric_cols.describe().T
    summary["Skewness"] = numeric_cols.apply(skew).values
    summary["Kurtosis"] = numeric_cols.apply(kurtosis).values

    plots = {}
    for column in numeric_cols.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(numeric_cols[column], kde=True, bins=30, color='blue', ax=ax)
        ax.set_title(f"Distribution of {column}")
        plt.tight_layout()
        plots[column] = fig
        plt.close(fig)

    return summary, plots

# Categorical Feature Analysis
def analyze_categorical(df):
    """Analyzes categorical features and visualizes their counts."""
    logger.info("Analyzing categorical features...")
    categorical_cols = df.select_dtypes(include=['object', 'category'])
    if categorical_cols.empty:
        logger.warning("No categorical columns found.")
        return None

    plots = {}
    for column in categorical_cols.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, y=column, order=df[column].value_counts().index, palette="viridis", ax=ax)
        ax.set_title(f"Count Plot for {column}")
        plt.tight_layout()
        plots[column] = fig
        plt.close(fig)

    return plots

# Analyze Target Variable
def analyze_target_variable(df, target_column):
    """Analyzes the target variable."""
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found.")
        return None

    target_data = df[target_column]
    fig, ax = plt.subplots(figsize=(8, 6))

    if target_data.dtype in ['int64', 'float64']:
        sns.histplot(target_data, kde=True, bins=30, color='blue', ax=ax)
        ax.set_title(f"Distribution of Target Variable: {target_column}")
    elif target_data.nunique() < 20:
        sns.countplot(data=df, x=target_column, palette="viridis", ax=ax)
        ax.set_title(f"Frequency of Categories in {target_column}")
    else:
        logger.warning(f"Target variable '{target_column}' is not suitable for analysis.")
        return None

    plt.tight_layout()
    return fig

# Analyze Text Data
def analyze_text_data(df, text_column):
    """Analyzes text data and generates WordCloud and length distribution plots."""
    if text_column not in df.columns:
        logger.error(f"Text column '{text_column}' not found.")
        return None, None

    text_data = df[text_column].dropna()
    text_content = " ".join(text_data.astype(str))

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_content)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    ax_wc.set_title(f"WordCloud for {text_column}")

    text_lengths = text_data.astype(str).apply(len)
    fig_len, ax_len = plt.subplots(figsize=(8, 6))
    sns.histplot(text_lengths, kde=True, color="purple", bins=30, ax=ax_len)
    ax_len.set_title(f"Distribution of Text Lengths in {text_column}")

    return fig_wc, fig_len

# Correlation Heatmap
def correlation_analysis(df):
    """Generates a correlation heatmap for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logger.warning("No numeric columns available for correlation analysis.")
        return None

    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    return fig

# Outlier Analysis
def visualize_outliers(df):
    """Visualizes outliers using boxplots for numerical features."""
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        logger.warning("No numerical columns found for outlier analysis.")
        return None

    plots = {}
    for column in numeric_cols.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=column, color='orange', ax=ax)
        ax.set_title(f"Boxplot for {column}")
        plt.tight_layout()
        plots[column] = fig
        plt.close(fig)

    return plots

# Missing Value Analysis
def visualize_missing_values(df):
    """Visualizes missing data in the dataset."""
    missing = df.isnull().mean()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("No missing values in the dataset.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    missing.sort_values(ascending=False).plot(kind="bar", color="red", ax=ax)
    ax.set_title("Missing Values Ratio")
    ax.set_xlabel("Features")
    ax.set_ylabel("Ratio")
    plt.tight_layout()
    return fig

# Main EDA Function
def perform_eda(df, target_column=None, text_column=None):
    """Executes the full exploratory data analysis process."""
    logger.info("Starting Exploratory Data Analysis (EDA)...")
    results = {}

    # Dataset Overview
    results["overview"] = dataset_overview(df)

    # Missing Values Visualization
    results["missing_values"] = visualize_missing_values(df)

    # Numerical Feature Analysis
    results["numerical_summary"], results["numerical_plots"] = analyze_numerical(df)

    # Categorical Feature Analysis
    results["categorical_plots"] = analyze_categorical(df)

    # Target Variable Analysis
    if target_column:
        results["target_analysis"] = analyze_target_variable(df, target_column)

    # Text Data Analysis
    if text_column:
        results["text_analysis"] = analyze_text_data(df, text_column)

    # Correlation Analysis
    results["correlation_heatmap"] = correlation_analysis(df)

    # Outlier Visualization
    results["outlier_plots"] = visualize_outliers(df)

    logger.info("EDA completed successfully!")
    return results
