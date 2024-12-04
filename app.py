from matplotlib import _preprocess_data
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
from io import StringIO
import warnings
from src.data_ingestion import load_data
from src.data_preprocessing import *  # Import all preprocessing functions
from src.eda import perform_eda  # Import EDA function
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

if "modified_df" not in st.session_state:
    st.session_state.modified_df = None  # Initialize session state if not already set

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
            st.session_state.modified_df = df.copy()  # Store the dataframe in session state
            st.write("Dataset Preview:")
            st.dataframe(df.head())
    
            if st.session_state.modified_df is not None:
                modified_df = st.session_state.modified_df  # Always use session state modified_df

            # Add a call to summarize_columns
            if option == "Dataset Understanding":
                st.subheader("Dataset Understanding")
                st.write("### Dataset Info")
                summarize_columns(modified_df)  # You can define this function to provide detailed column summaries
                st.write("### Dataset Shape")
                st.write(modified_df.shape)
                st.write("### Dataset Columns")
                st.write(modified_df.columns)
                st.write("### Missing Values")
                st.write(modified_df.isnull().sum())
                st.write("### Duplicate Rows")
                st.write(modified_df.duplicated().sum())
            if st.button("Save Modified Dataset"):
                save_data(modified_df)  # Ensure save_data handles saving the dataframe correctly
                st.write("Dataset Saved Successfully!")


            elif option == "Data Cleaning":
                st.subheader("Data Cleaning")

               
                # Missing Values Handling
                st.write("### Handle Missing Values")
                missing_method = st.selectbox("Select a method:", ["Drop Rows", "Drop Columns", "Simple", "KNN"])
                if st.button("Handle Missing Values"):
                    modified_df = handle_missing_values(modified_df, method=missing_method)
                    st.session_state.modified_df = modified_df  # Save changes to session state
                    st.write("### Missing Values Handled Successfully!")
                    st.dataframe(modified_df.head())

                # Remove Duplicates
                st.write("### Remove Duplicates")
                if st.button("Remove Duplicates"):
                    modified_df = remove_duplicates(modified_df)
                    st.session_state.modified_df = modified_df
                    st.write("### Duplicates Removed Successfully!")
                    st.dataframe(modified_df.head())

                # Remove Unnecessary Columns
                columns_to_remove = st.multiselect("Select columns to remove:", modified_df.columns)
                if st.button("Remove Columns"):
                    modified_df = remove_unnecessary_columns(modified_df, columns=columns_to_remove)
                    st.session_state.modified_df = modified_df
                    st.write("Columns removed successfully!")
                    st.dataframe(modified_df.head())

                # Encode Categorical Variables
                st.write("### Encode Categorical Columns")
                categorical_columns = [col for col in modified_df.columns if modified_df[col].dtype == 'object']
                if categorical_columns:
                    columns_to_encode = st.multiselect("Select columns to encode:", categorical_columns)
                    encoding_method = st.selectbox("Select an encoding method:", ["Label", "OneHot"])
                    if st.button("Encode Selected Columns"):
                        if columns_to_encode:
                            try:
                                for column in columns_to_encode:
                                    modified_df = encode_categorical(modified_df, column_name=column, encoding_type=encoding_method)
                                st.session_state.modified_df = modified_df
                                st.write("### Selected Columns Encoded Successfully!")
                                st.dataframe(modified_df.head())
                            except Exception as e:
                                st.error(f"Error while encoding: {e}")
                        else:
                            st.warning("Please select at least one column to encode.")
                else:
                    st.write("No categorical columns found for encoding.")

                # Handle Outliers
                st.write("### Handle Outliers")
                outlier_method = st.selectbox("Select an outlier handling method:", ["IQR", "Z-Score"])
                if st.button("Handle Outliers"):
                    modified_df = handle_outliers(modified_df, method=outlier_method)
                    st.session_state.modified_df = modified_df
                    st.write("### Outliers Handled Successfully!")
                    st.dataframe(modified_df.head())

                # Scale and Normalize
                st.write("### Scale and Normalize")
                scaling_method = st.selectbox("Select a scaling method:", ["Standard", "MinMax", "Robust"])
                if st.button("Scale and Normalize"):
                    modified_df = scale_and_normalize(modified_df, method=scaling_method, library="sklearn")
                    st.session_state.modified_df = modified_df
                    st.write("### Scaling and Normalization Done Successfully!")
                    st.dataframe(modified_df.head())
                # Fix Data Types
                st.write("### Fix Data Types")
                if st.button("Fix Data Types"):
                    modified_df = fix_data_types(modified_df)  # Fix incorrect data types
                    st.session_state.modified_df = modified_df
                    st.write("### Data Types Fixed Successfully!")
                    st.dataframe(modified_df.head())

                # Handle Typos
                st.write("### Handle Typos in Categorical Columns")
                if st.button("Handle Typos"):
                    modified_df = handle_typos(modified_df)  # Fix typos in categorical columns
                    st.session_state.modified_df = modified_df
                    st.write("### Typos Handled Successfully!")
                    st.dataframe(modified_df.head())


                # Apply PCA for Dimensionality Reduction
                st.write("### Apply PCA")
                n_components = st.slider("Select number of components:", 1, min(modified_df.shape[1], 10))
                if st.button("Apply PCA"):
                    modified_df = apply_pca(modified_df, n_components=n_components)
                    st.session_state.modified_df = modified_df
                    st.write(f"### PCA Applied with {n_components} Components!")
                    st.dataframe(modified_df.head())

                if st.button("Download Modified Dataset"):
                    csv = modified_df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "encoded_dataset.csv", "text/csv")

            elif option == "Exploratory Data Analysis (EDA)":
                st.subheader("Exploratory Data Analysis (EDA)")
                eda_choice = st.selectbox("Select an EDA method:", ("Summary Statistics", "Correlation Matrix", "Pairplot", "Heatmap"))
                if st.button("Perform EDA"):
                    perform_eda(modified_df, eda_choice)  # Use the modified dataframe for EDA

            elif option == "Feature Engineering":
                st.subheader("Feature Engineering")

                # Create New Feature
                st.write("### Create New Feature")
                new_feature_name = st.text_input("Enter new feature name:")
                expression = st.text_input("Enter expression (e.g., 'col1 + col2'):")
                if st.button("Create New Feature"):
                    modified_df = create_new_feature(modified_df, new_feature_name, expression)
                    st.session_state.modified_df = modified_df
                    st.write(f"### New Feature '{new_feature_name}' Created Successfully!")
                    st.dataframe(modified_df.head())

                # Feature Selection
                target_column = st.selectbox("Select Target Column", modified_df.columns)
                selection_method = st.selectbox("Select a feature selection method:", ["Correlation", "Chi2", "ANOVA"])
                if selection_method in ["Chi2", "ANOVA"]:
                    k = st.slider("Select number of top features:", min_value=1, max_value=len(modified_df.columns)-1, value=10)
                else:
                    correlation_threshold = st.slider("Select correlation threshold:", min_value=0.0, max_value=1.0, value=0.1)
                if st.button("Select Features"):
                    if selection_method == "Correlation":
                        modified_df = feature_selection(modified_df, target_column, method=selection_method, correlation_threshold=correlation_threshold)
                        st.session_state.modified_df = modified_df
                    else:
                        modified_df = feature_selection(modified_df, target_column, method=selection_method, k=k)
                        st.session_state.modified_df = modified_df
                    st.write(f"### Features Selected Using {selection_method}")
                    st.dataframe(modified_df.head())

                # Polynomial Features
                st.write("### Polynomial Features")
                degree = st.slider("Select degree of polynomial features:", min_value=2, max_value=5, value=2)
                if st.button("Generate Polynomial Features"):
                    modified_df = polynomial_features(modified_df, degree=degree)
                    st.session_state.modified_df = modified_df
                    st.write(f"### Polynomial Features of Degree {degree} Created Successfully!")
                    st.dataframe(modified_df.head())

                # Interaction Features
                st.write("### Interaction Features")
                if st.button("Generate Interaction Features"):
                    modified_df = interaction_features(modified_df)
                    st.session_state.modified_df = modified_df
                    st.write("### Interaction Features Created Successfully!")
                    st.dataframe(modified_df.head())

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
                    x_axis = st.sidebar.selectbox("Select X-axis", options=modified_df.columns)
                    y_axis = st.sidebar.selectbox("Select Y-axis", options=modified_df.columns)
                    hue = st.sidebar.selectbox("Select color/hue", options=[None] + list(modified_df.columns), index=0)
                else:
                    x_axis = st.sidebar.selectbox("Select X-axis", options=modified_df.columns) if vis_choice == 'Heatmap' else None
                    y_axis = st.sidebar.selectbox("Select Y-axis", options=modified_df.columns)

                if st.button(f"Create {vis_choice}"):
                    if vis_choice == "Bar Chart":
                        create_bar_chart(modified_df, x_axis, y_axis, hue, library_choice)
                    elif vis_choice == "Scatter Plot":
                        create_scatter_plot(modified_df, x_axis, y_axis, hue, library_choice)
                    elif vis_choice == "Line Chart":
                        create_line_chart(modified_df, x_axis, y_axis, hue, library_choice)
                    elif vis_choice == "Histogram":
                        create_histogram(modified_df, x_axis, hue, library_choice)
                    elif vis_choice == "Box Plot":
                        create_box_plot(modified_df, x_axis, y_axis, hue, library_choice)
                    elif vis_choice == "Heatmap":
                        create_heatmap(modified_df, library_choice)
                    elif vis_choice == "Pie Chart":
                        create_pie_chart(modified_df, library_choice)
                    elif vis_choice == "Violin Plot":
                        create_violin_plot(modified_df, x_axis, y_axis, hue, library_choice)

            elif option == "Modeling":
                st.subheader("Modeling")
                model_choice = st.selectbox("Select a model:", ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN'])
                if st.button("Train Model"):
                    trained_model = train_model(modified_df, model_choice)
                    st.write("### Model Trained Successfully!")
                    st.write(trained_model)

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
