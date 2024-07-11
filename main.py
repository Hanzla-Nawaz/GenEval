from matplotlib import _preprocess_data
import streamlit as st 
import pandas as pd
import plotly.express as px 
import os
import io
from io import StringIO
import warnings
from src.data_ingestion import load_data
from src.data_preprocessing import *
from src.eda import perform_eda
from src.feature_engineering import *
from src.model_training import *
from src.visualization import *
import logging
from src.chatbot import chat_with_bot


# Warning control
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Streamlit app
def main():
    st.title("Data Science Assistant")

    # Sidebar navigation
    st.sidebar.title("Data Science Steps")
    option = st.sidebar.selectbox("Select a step:", 
                                  ("Dataset Understanding", "Data Cleaning", "Exploratory Data Analysis (EDA)", "Feature Engineering", "Modeling", "Visualization", "Chat with Bot"))

    # File upload widget
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            st.write("Dataset Preview:")
            st.dataframe(df.head())

            if option == "Dataset Understanding":
                st.subheader("Dataset Understanding")
                st.write("### Dataset Info")
                buffer = StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
                st.write("### Dataset Shape")
                st.write(df.shape)
                st.write("### Dataset Columns")
                st.write(df.columns)
                st.write("### Missing Values")
                st.write(df.isnull().sum())
                st.write("### Duplicate Rows")
                st.write(df.duplicated().sum())

            elif option == "Data Cleaning":
                st.subheader("Data Cleaning")

                # Missing Values Handling
                st.write("### Handle Missing Values")
                missing_method = st.selectbox("Select a method:", ["Drop Rows", "Drop Columns", "Simple", "KNN"])
                if st.button("Handle Missing Values"):
                    df = handle_missing_values(df, missing_method)
                    st.write("### Missing Values Handled Successfully!")
                    st.dataframe(df.head())

                # Remove Duplicates
                st.write("### Remove Duplicates")
                if st.button("Remove Duplicates"):
                    df = remove_duplicates(df)
                    st.write("### Duplicates Removed Successfully!")
                    st.dataframe(df.head())

                # Encode Categorical Variables
                st.write("### Encode Categorical Variables")
                encoding_method = st.selectbox("Select an encoding method:", ["Label", "OneHot"])
                if st.button("Encode Categorical Variables"):
                    df = encode_categorical(df, encoding_method)
                    st.write("### Categorical Variables Encoded Successfully!")
                    st.dataframe(df.head())

                # Handle Outliers
                st.write("### Handle Outliers")
                outlier_method = st.selectbox("Select an outlier handling method:", ["IQR", "Z-Score"])
                if st.button("Handle Outliers"):
                    df = handle_outliers(df, outlier_method)
                    st.write("### Outliers Handled Successfully!")
                    st.dataframe(df.head())

                # Scale and Normalize
                st.write("### Scale and Normalize")
                scaling_method = st.selectbox("Select a scaling method:", ["Standard", "MinMax", "Robust"])
                if st.button("Scale and Normalize"):
                    df = scale_and_normalize(df, scaling_method)
                    st.write("### Scaling and Normalization Done Successfully!")
                    st.dataframe(df.head())

            elif option == "Exploratory Data Analysis (EDA)":
                st.subheader("Exploratory Data Analysis (EDA)")
                eda_choice = st.selectbox("Select an EDA method:", ("Summary Statistics", "Correlation Matrix", "Pairplot", "Heatmap"))
                if st.button("Perform EDA"):
                    perform_eda(df, eda_choice)

            elif option == "Feature Engineering":
                st.subheader("Feature Engineering")

                # Create New Feature
                st.write("### Create New Feature")
                new_feature_name = st.text_input("Enter new feature name:")
                expression = st.text_input("Enter expression (e.g., 'col1 + col2'):")
                if st.button("Create New Feature"):
                    df = create_new_feature(df, new_feature_name, expression)
                    st.write(f"### New Feature '{new_feature_name}' Created Successfully!")
                    st.dataframe(df.head())

                # Feature Selection
                target_column = st.selectbox("Select Target Column", df.columns)
                selection_method = st.selectbox("Select a feature selection method:", ["Correlation", "Chi2", "ANOVA"])
                if selection_method in ["Chi2", "ANOVA"]:
                    k = st.slider("Select number of top features:", min_value=1, max_value=len(df.columns)-1, value=10)
                else:
                    correlation_threshold = st.slider("Select correlation threshold:", min_value=0.0, max_value=1.0, value=0.1)
                if st.button("Select Features"):
                    if selection_method == "Correlation":
                        df = feature_selection(df, target_column, method=selection_method, correlation_threshold=correlation_threshold)
                    else:
                        df = feature_selection(df, target_column, method=selection_method, k=k)
                    st.write(f"### Features Selected Using {selection_method}")
                    st.dataframe(df.head())

                # Polynomial Features
                st.write("### Polynomial Features")
                degree = st.slider("Select degree of polynomial features:", min_value=2, max_value=5, value=2)
                if st.button("Generate Polynomial Features"):
                    df = polynomial_features(df, degree=degree)
                    st.write(f"### Polynomial Features of Degree {degree} Created Successfully!")
                    st.dataframe(df.head())

                # Interaction Features
                st.write("### Interaction Features")
                if st.button("Generate Interaction Features"):
                    df = interaction_features(df)
                    st.write("### Interaction Features Created Successfully!")
                    st.dataframe(df.head())

            elif option == "Visualization":
                st.sidebar.title("Visualization Options")
                vis_choice = st.sidebar.selectbox(
                    "Select Visualization Type",
                    ('Bar Chart', 'Scatter Plot', 'Line Chart', 'Histogram', 'Box Plot', 'Heatmap', 'Pie Chart', 'Violin Plot')
                )

                library_choice = st.sidebar.selectbox(
                    "Select Visualization Library",
                    ('Plotly', 'Altair', 'Seaborn')
                )

                if vis_choice != 'Heatmap' and vis_choice != 'Pie Chart':
                    x_axis = st.sidebar.selectbox("Select X-axis", options=df.columns)
                    y_axis = st.sidebar.selectbox("Select Y-axis", options=df.columns)
                    hue = st.sidebar.selectbox("Select color/hue", options=[None] + list(df.columns), index=0)
                else:
                    x_axis = st.sidebar.selectbox("Select X-axis", options=df.columns) if vis_choice == 'Pie Chart' else None
                    y_axis, hue = None, None

                if st.sidebar.button("Generate Visualization"):
                    if library_choice == 'Plotly':
                        fig = plotly_visualizations(df, vis_choice, x_axis, y_axis, hue)
                        if fig:
                            st.plotly_chart(fig)
                    elif library_choice == 'Altair':
                        chart = altair_visualizations(df, vis_choice, x_axis, y_axis, hue)
                        if chart:
                            st.altair_chart(chart, use_container_width=True)
                    elif library_choice == 'Seaborn':
                        seaborn_visualizations(df, vis_choice, x_axis, y_axis, hue)

            elif option == "Modeling":
                st.subheader("Modeling")

                # Select Target Column
                target_column = st.selectbox("Select target column", df.columns)

                problem_type = determine_problem_type(df, target_column)
                st.write(f"Detected problem type: {problem_type}")

                algorithm = st.selectbox("Select Algorithm", ["Random Forest", "Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"])

                hyperparameter_tuning = st.checkbox("Perform Hyperparameter Tuning")

                params = None
                if hyperparameter_tuning:
                    if problem_type == "classification" and algorithm == "Random Forest":
                        params = {'classifier__n_estimators': [50, 100, 200]}
                    elif problem_type == "classification" and algorithm == "Logistic Regression":
                        params = {'classifier__C': [0.1, 1.0, 10.0]}
                    elif problem_type == "classification" and algorithm == "Support Vector Machine":
                        params = {'classifier__C': [0.1, 1.0, 10.0]}
                    elif problem_type == "classification" and algorithm == "K-Nearest Neighbors":
                        params = {'classifier__n_neighbors': [3, 5, 7]}
                    elif problem_type == "regression" and algorithm == "Random Forest":
                        params = {'regressor__n_estimators': [50, 100, 200]}
                    elif problem_type == "regression" and algorithm == "Support Vector Machine":
                        params = {'regressor__C': [0.1, 1.0, 10.0]}
                    elif problem_type == "regression" and algorithm == "K-Nearest Neighbors":
                        params = {'regressor__n_neighbors': [3, 5, 7]}

                # Train Model Button
                if st.button("Train Model"):
                    try:
                        X, y = df.drop(target_column, axis=1), df[target_column]
                        model, metric1, metric2, additional_info = train_model(X, y, problem_type, algorithm, params if hyperparameter_tuning else None)

                        if model:
                            save_model(model, 'trained_model')
                            st.success("Model trained successfully and saved!")
                            st.write("### Model Performance")
                            if problem_type == "classification":
                                st.write(f"Accuracy: {metric1:.2f}")
                                st.write("Classification Report:")
                                st.text(metric2)
                                st.write("Confusion Matrix:")
                                st.write(additional_info)
                            elif problem_type == "regression":
                                st.write(f"Mean Squared Error: {metric1:.2f}")
                                st.write(f"R2 Score: {metric2:.2f}S")
                            elif problem_type == "unsupervised":
                                if algorithm == "K-Means Clustering":
                                    st.write("### K-Means Clustering Labels")
                                    st.write(metric1)
                                elif algorithm == "Principal Component Analysis (PCA)":
                                    st.write("### PCA Components")
                                    st.write(metric1)
                        else:
                            st.warning("Model training failed. Please check the dataset and parameters.")
                    except Exception as e:
                        logger.error(f"Error during model training: {e}")
                        st.error(f"An error occurred during model training: {e}")

            # Model loading and prediction
            if st.button("Load and Predict"):
                try:
                    model = load_model('trained_model')
                    if model:
                        st.success("Model loaded successfully!")
                        st.write("Model details:")
                        st.write(model)
                    else:
                        st.warning("Model loading failed.")
                except Exception as e:
                    logger.error(f"Error during model loading: {e}")
                    st.error(f"An error occurred during model loading: {e}")

    

            elif option == "Chat with Bot":
                st.subheader("Chat with Bot")
                user_input = st.text_input("Ask a question about data science:")
                if user_input:
                    response = chat_with_bot(user_input)
                    st.write("Bot Response:")
                    st.write(response)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            logger.error(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
