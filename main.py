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
from src.eda import *
from src.feature_engineering import *
from src.model_training import *
from src.visualization import *
import logging
#from src.chatbot import chat_with_bot


# Warning control
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Streamlit App Configuration
st.set_page_config(page_title="Data Science App", layout="wide")

# Streamlit app
def main():
    st.title("Data Science Assistant")

    # Sidebar navigation
    st.sidebar.title("Data Science Steps")
    option = st.sidebar.selectbox("Select a step:", 
                                  ("Dataset Understanding", "Data Cleaning", "Exploratory Data Analysis (EDA)", "Feature Engineering", "Modeling", "Visualization", "Chat with Bot"))

# File upload widget
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json", "parquet"])
    if uploaded_file:
        # Read uploaded file
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return

            st.write("### Dataset Preview")
            st.dataframe(df.head())

            if option == "Dataset Understanding":
                st.subheader("Dataset Understanding")

                st.write("### Dataset Shape")
                st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

                st.write("### Missing Values")
                st.write(df.isnull().sum())

                st.write("### Duplicate Rows")
                st.write(f"Duplicates: {df.duplicated().sum()}")
            if st.button("Save Modified Dataset"):
                save_data(df)  # Ensure save_data handles saving the dataframe correctly
                st.write("Dataset Saved Successfully!")

            elif option == "Data Cleaning":
                st.subheader("Data Cleaning")


                st.write("### Handle Missing Values")
                missing_method = st.selectbox("Select a method:", ["Drop Rows", "Drop Columns", "Simple", "KNN"])
                #columns = st.sidebar.multiselect("Select Columns", df.columns)
                if st.button("Handle Missing Values"):
                    df = handle_missing_values(df, method=missing_method)
                    st.write("### Missing Values Handled Successfully!")
                    st.dataframe(df.head())    

                # Remove Duplicates
                st.write("### Remove Duplicates")
                if st.button("Remove Duplicates"):
                    df = remove_duplicates(df)
                    st.write("### Duplicates Removed Successfully!")
                    st.dataframe(df.head())

                
                # Remove Unnecessary Columns
                columns_to_remove = st.multiselect("Select columns to remove:", df.columns)
                if st.button("Remove Columns"):
                    df = remove_unnecessary_columns(df, columns=columns_to_remove)
                    st.write("Columns removed successfully!")
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
                columns = st.sidebar.multiselect("Select Columns", df.columns)
                scaling_method = st.selectbox("Select a scaling method:", ["Standard", "MinMax", "Robust"])
                if st.button("Scale and Normalize"):
                    df = scale_and_normalize(df, scaling_method, columns=columns)
                    st.write("### Scaling and Normalization Done Successfully!")
                    st.dataframe(df.head())

                # Encode Categorical Variables
                st.write("### Encode Categorical Columns")
                categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
                if categorical_columns:
                    columns_to_encode = st.multiselect("Select columns to encode:", categorical_columns)
                    encoding_method = st.selectbox("Select an encoding method:", ["Label", "OneHot"])
                    if st.button("Encode Selected Columns"):
                        if columns_to_encode:
                            try:
                                # Correct function call with 'method' and 'columns' arguments
                                df = encode_categorical(df, method=encoding_method, columns=columns_to_encode)
                                st.write("### Selected Columns Encoded Successfully!")
                                st.dataframe(df.head())
                            except Exception as e:
                                st.error(f"Error while encoding: {e}")
                        else:
                            st.warning("Please select at least one column to encode.")
                else:
                    st.write("No categorical columns found for encoding.")

  

                   # Fix Data Types
                st.write("### Fix Data Types")
                #columns = st.sidebar.multiselect("Select Columns", df.columns)
                if st.button("Fix Data Types"):
                    dtype = st.sidebar.text_input("Enter Data Type (e.g., int, float, str)")
                    df = fix_data_types(df, columns, dtype)
                    st.success(f"Data type of {columns} changed to {dtype}!")
                    st.dataframe(df.head())


                # Apply PCA for Dimensionality Reduction
                st.write("### Apply PCA")
                n_components = st.slider("Select number of components:", 1, min(df.shape[1], 10))
                if st.button("Apply PCA"):
                    df = apply_pca(df, n_components=n_components)
                    st.session_state.df = df
                    st.write(f"### PCA Applied with {n_components} Components!")
                    st.dataframe(df.head())

                # Add Time-Series Features
                st.write("### Add Time-Series Features")
                datetime_column = st.selectbox("Select Datetime Column", df.columns)
                if st.button("Add Time-Series Features"):
                    df = time_series_features(df, datetime_column)
                    st.session_state.df = df
                    st.success("Time-series features added!")
                    st.dataframe(df.head())

                # Handle Class Imbalance
                st.write("### Handle Class Imbalance")
                target_column = st.selectbox("Select Target Column", df.columns)
                method = st.selectbox("Select Method", ["SMOTE", "undersample"])
                if st.button("Handle Class Imbalance"):
                    df = handle_imbalance(df, target_column, method)
                    st.session_state.df = df
                    st.success("Class imbalance handled!")
                    st.dataframe(df.head())

                # Feature Selection
                st.write("### Feature Selection")
                target_column = st.selectbox("Select Target Column for Feature Selection", df.columns)
                k = st.slider("Select Number of Features", 1, min(len(df.columns) - 1, 10))
                if st.button("Select Features"):
                    df = feature_selection(df, target_column, k=k)
                    st.session_state.df = df
                    st.success(f"Top {k} features selected!")
                    st.dataframe(df.head())

                # Advanced Text Preprocessing
                st.write("### Advanced Text Preprocessing")
                text_column = st.selectbox("Select Text Column", df.columns)
                remove_stopwords = st.checkbox("Remove Stopwords", value=True)
                lemmatize = st.checkbox("Apply Lemmatization", value=True)
                stem = st.checkbox("Apply Stemming", value=False)
                sentiment_analysis = st.checkbox("Perform Sentiment Analysis", value=False)
                if st.button("Preprocess Text"):
                    df[text_column] = text_preprocessing_advanced(
                        df[text_column],
                        remove_stopwords=remove_stopwords,
                        lemmatize=lemmatize,
                        stem=stem,
                        sentiment_analysis=sentiment_analysis,
                    )
                    st.session_state.df = df
                    st.success("Text data preprocessed successfully!")
                    st.dataframe(df.head())

                # Save Processed Data
                if uploaded_file and st.button("Download Processed Dataset"):
                    output_file = "processed_data.csv"
                    save_data(df, output_file)
                    st.download_button(
                        label="Download Processed Dataset",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name=output_file,
                        mime="text/csv",
                    )


            elif option == "Exploratory Data Analysis (EDA)":
                st.subheader("Exploratory Data Analysis (EDA)")
                
                eda_choice = st.selectbox(
                    "Select an EDA method:", 
                    ("Dataset Overview", "Numerical Feature Analysis", "Categorical Feature Analysis", 
                    "Correlation Heatmap", "Outlier Analysis", "Target Variable Analysis", 
                    "Text Data Analysis", "Full EDA")
                )
                
                            # Perform EDA based on user choice
                if eda_choice == "Dataset Overview":
                    st.write("### Dataset Overview")
                    overview = dataset_overview(df)
                    st.json(overview)

                elif eda_choice == "Numerical Feature Analysis":
                    st.write("### Numerical Feature Analysis")
                    summary, plots = analyze_numerical(df)
                    if summary is not None:
                        st.write("**Statistical Summary**")
                        st.dataframe(summary)
                        st.write("**Distribution Plots**")
                        for column, fig in plots.items():
                            st.pyplot(fig)
                    else:
                        st.warning("No numerical features found in the dataset.")

                elif eda_choice == "Categorical Feature Analysis":
                    st.write("### Categorical Feature Analysis")
                    plots = analyze_categorical(df)
                    if plots:
                        for column, fig in plots.items():
                            st.write(f"**Count Plot for {column}**")
                            st.pyplot(fig)
                    else:
                        st.warning("No categorical features found in the dataset.")

                elif eda_choice == "Correlation Heatmap":
                    st.write("### Correlation Heatmap")
                    heatmap_fig = correlation_analysis(df)
                    if heatmap_fig:
                        st.pyplot(heatmap_fig)
                    else:
                        st.warning("No numeric columns available for correlation analysis.")

                elif eda_choice == "Outlier Analysis":
                    st.write("### Outlier Analysis")
                    plots = visualize_outliers(df)
                    if plots:
                        for column, fig in plots.items():
                            st.write(f"**Boxplot for {column}**")
                            st.pyplot(fig)
                    else:
                        st.warning("No numerical features available for outlier analysis.")

                elif eda_choice == "Target Variable Analysis":
                    target_column = st.selectbox("Select Target Column", df.columns.tolist())
                    if target_column:
                        st.write(f"### Target Variable Analysis: {target_column}")
                        target_fig = analyze_target_variable(df, target_column)
                        if target_fig:
                            st.pyplot(target_fig)
                        else:
                            st.warning(f"Target variable '{target_column}' is not suitable for analysis.")

                elif eda_choice == "Text Data Analysis":
                    text_column = st.selectbox("Select Text Column", df.columns.tolist())
                    if text_column:
                        st.write(f"### Text Data Analysis: {text_column}")
                        wordcloud_fig, length_dist_fig = analyze_text_data(df, text_column)
                        if wordcloud_fig and length_dist_fig:
                            st.write("**WordCloud**")
                            st.pyplot(wordcloud_fig)
                            st.write("**Text Length Distribution**")
                            st.pyplot(length_dist_fig)
                        else:
                            st.warning(f"Text column '{text_column}' does not contain sufficient data.")

                elif eda_choice == "Full EDA":
                    st.write("### Full EDA (All Steps)")
                    target_column = st.selectbox("Select Target Column (Optional)", ["None"] + df.columns.tolist())
                    text_column = st.selectbox("Select Text Column (Optional)", ["None"] + df.columns.tolist())
                    
                    # Set target and text columns to None if not selected
                    target_column = None if target_column == "None" else target_column
                    text_column = None if text_column == "None" else text_column
                    
                    results = perform_eda(df, target_column=target_column, text_column=text_column)
                    
                    # Display results
                    st.write("**Dataset Overview**")
                    st.json(results["overview"])
                    
                    if results["missing_values"]:
                        st.write("**Missing Values**")
                        st.pyplot(results["missing_values"])
                    
                    if results["numerical_summary"] and results["numerical_plots"]:
                        st.write("**Numerical Feature Analysis**")
                        st.dataframe(results["numerical_summary"])
                        for column, fig in results["numerical_plots"].items():
                            st.pyplot(fig)
                    
                    if results["categorical_plots"]:
                        st.write("**Categorical Feature Analysis**")
                        for column, fig in results["categorical_plots"].items():
                            st.pyplot(fig)
                    
                    if results["correlation_heatmap"]:
                        st.write("**Correlation Heatmap**")
                        st.pyplot(results["correlation_heatmap"])
                    
                    if results["outlier_plots"]:
                        st.write("**Outlier Analysis**")
                        for column, fig in results["outlier_plots"].items():
                            st.pyplot(fig)
                    
                    if target_column and results.get("target_analysis"):
                        st.write(f"**Target Variable Analysis: {target_column}**")
                        st.pyplot(results["target_analysis"])
                    
                    if text_column and results.get("text_analysis"):
                        st.write(f"**Text Data Analysis: {text_column}**")
                        wordcloud_fig, length_dist_fig = results["text_analysis"]
                        st.pyplot(wordcloud_fig)
                        st.pyplot(length_dist_fig)

                    st.success("Full EDA completed.")





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
