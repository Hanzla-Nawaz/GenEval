import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def determine_problem_type(df, target_column):
    if df[target_column].dtype in [np.int64, np.float64]:
        unique_values = df[target_column].nunique()
        if unique_values <= 10:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"

def get_preprocessing_pipeline(df, numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    return preprocessor

def save_model(model, model_name):
    try:
        with open(f'{model_name}.pkl', 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Model saved as {model_name}.pkl")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def load_model(model_name):
    try:
        with open(f'{model_name}.pkl', 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model {model_name}.pkl loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def train_model(X, y, problem_type, algorithm, param_grid):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        preprocessor = get_preprocessing_pipeline(X, numeric_features, categorical_features)

        if problem_type == "classification":
            if algorithm == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif algorithm == "Logistic Regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif algorithm == "Support Vector Machine":
                model = SVC(kernel='rbf', random_state=42)
            elif algorithm == "K-Nearest Neighbors":
                model = KNeighborsClassifier()
            model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        elif problem_type == "regression":
            if algorithm == "Random Forest":
                model = RandomForestRegressor(random_state=42)
            elif algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "Support Vector Machine":
                model = SVR(kernel='linear')
            elif algorithm == "K-Nearest Neighbors":
                model = KNeighborsRegressor()
            model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        elif problem_type == "unsupervised":
            if algorithm == "K-Means Clustering":
                model = KMeans(n_clusters=3, random_state=42)
                model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('kmeans', model)])
            elif algorithm == "Principal Component Analysis (PCA)":
                model = PCA(n_components=2)
                model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('pca', model)])
            param_grid = None

        if param_grid:
            search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
            search.fit(X_train, y_train)
            model_pipeline = search.best_estimator_
            logger.info(f"Best parameters: {search.best_params_}")
        else:
            model_pipeline.fit(X_train, y_train)

        if problem_type in ["classification", "regression"]:
            y_pred = model_pipeline.predict(X_test)
            if problem_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                confusion = confusion_matrix(y_test, y_pred)
                logger.info(f"Model trained with accuracy: {accuracy}")
                return model_pipeline, accuracy, report, confusion
            elif problem_type == "regression":
                mse = mean_squared_error(y_test, y_predS)
                r2 = r2_score(y_test, y_pred)
                logger.info(f"Model trained with MSE: {mse}, R2: {r2}")
                return model_pipeline, mse, r2, None
        elif problem_type == "unsupervised":
            if algorithm == "K-Means Clustering":
                labels = model_pipeline.named_steps['kmeans'].labels_
                return model_pipeline, labels, None, None
            elif algorithm == "Principal Component Analysis (PCA)":
                components = model_pipeline.named_steps['pca'].transform(X_test)
                return model_pipeline, components, None, None

    except Exception as e:
        logger.error(f"Error training model: {e}")
        st.error(f"An error occurred while training the model: {e}")
        return None, None, None, None