import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import openai
import io  # Add missing import

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API
openai.api_key = 'sk-proj-55KmRDqImHLPdYCp2nFmT3BlbkFJxcPIv2abOoP7E5xIr525'

# Function to interact with the chatbot using OpenAI GPT-3.5
def chat_with_bot(prompt, history=[]):
    try:
        # Prepare the conversation history for OpenAI API
        conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
        for message in history:
            if message['role'] == 'user':
                conversation_history.append({"role": "user", "content": message['content']})
            else:
                conversation_history.append({"role": "assistant", "content": message['content']})

        conversation_history.append({"role": "user", "content": prompt})

        # Generate a response from the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.9,
        )

        # Get the response content
        bot_message = response['choices'][0]['message']['content']

        # Append user and bot messages to history
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": bot_message})

        return bot_message, history

    except Exception as e:
        logger.error(f"Error during chat: {e}")
        return "An error occurred while chatting with the bot.", history

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

# Define data cleaning functions
def handle_missing_values(df, method):
    if method == "Drop Rows":
        df = df.dropna()
    elif method == "Drop Columns":
        df = df.dropna(axis=1)
    elif method == "Fill with Mean":
        df = df.fillna(df.mean())
    elif method == "Fill with Median":
        df = df.fillna(df.median())
    elif method == "Fill with Mode":
        for column in df.columns:
            df[column].fillna(df[column].mode()[0], inplace=True)
    elif method == "KNN Imputation":
        imputer = KNNImputer(n_neighbors=5)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

def remove_duplicates(df):
    df = df.drop_duplicates()
    return df

def encode_categorical(df):
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column].astype(str))
    return df

def handle_outliers(df):
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def scale_and_normalize(df, method):
    if method == "Standard Scaling":
        scaler = StandardScaler()
    elif method == "MinMax Scaling":
        scaler = MinMaxScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return df

# Define function to determine problem type
def determine_problem_type(df, target_column):
    if df[target_column].dtype in [np.int64, np.float64]:  # Numeric target column
        unique_values = df[target_column].nunique()
        if unique_values <= 10:  # Treat as classification if less than or equal to 10 unique values
            return "classification"
        else:  # Treat as regression if more than 10 unique values
            return "regression"
    else:  # Non-numeric target column
        return "unsupervised"

# Define modeling function
def train_model(df, target_column, problem_type, algorithm):
    try:
        # Split data into features and target variable
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        if problem_type == "classification":
            if algorithm == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == "Logistic Regression":
                model = LogisticRegression(random_state=42)
            elif algorithm == "Support Vector Machine":
                model = SVC(kernel='rbf', random_state=42)
            elif algorithm == "K-Nearest Neighbors":
                model = KNeighborsClassifier(n_neighbors=5)
        elif problem_type == "regression":
            if algorithm == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "Support Vector Machine":
                model = SVR(kernel='linear')
            elif algorithm == "K-Nearest Neighbors":
                model = KNeighborsRegressor(n_neighbors=5)
        elif problem_type == "unsupervised":
            if algorithm == "K-Means Clustering":
                model = KMeans(n_clusters=3, random_state=42)
            elif algorithm == "Principal Component Analysis (PCA)":
                model = PCA(n_components=2)

        model.fit(X_train, y_train)

        # Make predictions and evaluate the model
        if problem_type in ["classification", "regression"]:
            y_pred = model.predict(X_test)
            if problem_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                confusion = confusion_matrix(y_test, y_pred)
                return model, accuracy, report, confusion
            elif problem_type == "regression":
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                return model, mse, r2, None
        elif problem_type == "unsupervised":
            return model, X_train, X_test, None
    except Exception as e:
        logger.error(f"Error training model: {e}")
        st.error(f"An error occurred while training the model: {e}")

# Streamlit app
def main():
    st.title("Data Science Assistant")

    # Sidebar navigation
    st.sidebar.title("Data Science Steps")
    option = st.sidebar.selectbox("Select a step:", 
                                  ("Dataset Understanding", "Data Cleaning", "Exploratory Data Analysis (EDA)", "Feature Engineering", "Modelling", "Visualization", "Chat with Bot"))

    # File upload widget
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Dataset Loaded Successfully!")
            st.write(df.head())

            # Display dataset information based on the selected option
            if option == "Dataset Understanding":
                st.subheader("Dataset Understanding")
                st.write("### Dataset Info")
                buffer = io.StringIO()
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
                st.write("Data Cleaning Options:")
                
                # Missing Values Handling
                st.write("### Handle Missing Values")
                missing_method = st.selectbox("Select a method:", ("Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode", "KNN Imputation"))
                if st.button("Handle Missing Values"):
                    df = handle_missing_values(df, missing_method)
                    st.write("### Missing Values Handled Successfully!")
                    st.write(df.head())

                # Remove Duplicates
                st.write("### Remove Duplicates")
                if st.button("Remove Duplicates"):
                    df = remove_duplicates(df)
                    st.write("### Duplicates Removed Successfully!")
                    st.write(df.head())

                # Encode Categorical Variables
                st.write("### Encode Categorical Variables")
                if st.button("Encode Categorical Variables"):
                    df = encode_categorical(df)
                    st.write("### Categorical Variables Encoded Successfully!")
                    st.write(df.head())

                # Handle Outliers
                st.write("### Handle Outliers")
                if st.button("Handle Outliers"):
                    df = handle_outliers(df)
                    st.write("### Outliers Handled Successfully!")
                    st.write(df.head())

                # Scale and Normalize
                st.write("### Scale and Normalize")
                scaling_method = st.selectbox("Select a method:", ("Standard Scaling", "MinMax Scaling"))
                if st.button("Scale and Normalize"):
                    df = scale_and_normalize(df, scaling_method)
                    st.write("### Scaling and Normalization Done Successfully!")
                    st.write(df.head())

            elif option == "Exploratory Data Analysis (EDA)":
                st.subheader("Exploratory Data Analysis (EDA)")
                eda_choice = st.selectbox("Select an EDA method:", ("Summary Statistics", "Correlation Matrix", "Pairplot", "Heatmap"))
                if st.button("Perform EDA"):
                    perform_eda(df, eda_choice)

            elif option == "Feature Engineering":
                st.write("### Feature Engineering")
                st.write("Feature engineering tasks can include creating new features, transforming existing ones, or selecting relevant features for your model.")
                st.write("#### Example:")
                st.write("You can create new features by combining existing ones, such as creating a 'total_income' feature by adding 'ApplicantIncome' and 'CoapplicantIncome'.")
                st.write("#### Your Task:")
                st.write("Perform any feature engineering tasks necessary for your dataset.")

            elif option == "Modelling":
                st.write("### Modelling")
                st.write("#### Select Target Column")
                target_column = st.selectbox("Select the target column:", df.columns)
                if target_column:
                    problem_type = determine_problem_type(df, target_column)
                    st.write(f"#### Problem Type: {problem_type.capitalize()}")

                    algorithm = st.selectbox("Select an algorithm:", ("Random Forest", "Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"))
                    if st.button("Train Model"):
                        model, metric1, metric2, metric3 = train_model(df, target_column, problem_type, algorithm)
                        st.write("### Model Trained Successfully!")
                        if problem_type == "classification":
                            st.write(f"Accuracy Score: {metric1}")
                            st.write(f"Classification Report:\n{metric2}")
                            st.write(f"Confusion Matrix:\n{metric3}")
                        elif problem_type == "regression":
                            st.write(f"Mean Squared Error: {metric1}")
                            st.write(f"R-squared Score: {metric2}")
                        elif problem_type == "unsupervised":
                            st.write(f"Unsupervised Model Training Completed!")

            elif option == "Visualization":
                st.write("### Visualization")
                visualization_type = st.selectbox("Select a visualization type:", ("Bar Chart", "Histogram", "Line Chart", "Scatter Plot"))
                if visualization_type:
                    st.write("### Create Visualization")
                    if visualization_type == "Bar Chart":
                        st.write("#### Bar Chart")
                        x_axis = st.selectbox("Select X axis:", df.columns)
                        y_axis = st.selectbox("Select Y axis:", df.columns)
                        fig = px.bar(df, x=x_axis, y=y_axis)
                        st.plotly_chart(fig)
                    elif visualization_type == "Histogram":
                        st.write("#### Histogram")
                        column = st.selectbox("Select column:", df.columns)
                        fig = px.histogram(df, x=column)
                        st.plotly_chart(fig)
                    elif visualization_type == "Line Chart":
                        st.write("#### Line Chart")
                        x_axis = st.selectbox("Select X axis:", df.columns)
                        y_axis = st.selectbox("Select Y axis:", df.columns)
                        fig = px.line(df, x=x_axis, y=y_axis)
                        st.plotly_chart(fig)
                    elif visualization_type == "Scatter Plot":
                        st.write("#### Scatter Plot")
                        x_axis = st.selectbox("Select X axis:", df.columns)
                        y_axis = st.selectbox("Select Y axis:", df.columns)
                        fig = px.scatter(df, x=x_axis, y=y_axis)
                        st.plotly_chart(fig)

            elif option == "Chat with Bot":
                st.write("### Chat with Bot")
                chat_history = st.session_state.get("chat_history", [])
                user_input = st.text_input("You:")
                if st.button("Send"):
                    if user_input:
                        bot_response, chat_history = chat_with_bot(user_input, chat_history)
                        st.write("Bot:", bot_response)
                        st.session_state["chat_history"] = chat_history

        except Exception as e:
            logger.error(f"Error: {e}")
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()