# src/feature_engineering/feature_engineering.py
import streamlit as st
import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_new_feature(df, new_feature_name, expression):
    """
    Create a new feature in the dataframe based on an expression.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    new_feature_name (str): The name of the new feature to be created.
    expression (str): The expression to evaluate for creating the new feature.
    
    Returns:
    pd.DataFrame: Dataframe with the new feature added.
    """
    try:
        df[new_feature_name] = df.eval(expression)
        logger.info(f"New feature '{new_feature_name}' created successfully using expression: {expression}")
    except Exception as e:
        logger.error(f"Error creating new feature '{new_feature_name}': {e}")
    return df

def feature_selection(df, target_column, method="Correlation", k=10, correlation_threshold=0.1):
    """
    Select features based on a specified method.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    target_column (str): The target column for feature selection.
    method (str): The method for feature selection ('Correlation', 'Chi2', 'ANOVA').
    k (int): Number of top features to select (applicable for Chi2 and ANOVA).
    correlation_threshold (float): Threshold for correlation-based feature selection.
    
    Returns:
    pd.DataFrame: Dataframe with selected features.
    """
    try:
        if method == "Correlation":
            correlation_matrix = df.corr()
            target_corr = correlation_matrix[target_column].abs().sort_values(ascending=False)
            selected_features = target_corr[target_corr > correlation_threshold].index.tolist()
            logger.info(f"Selected features based on correlation threshold {correlation_threshold}: {selected_features}")
            return df[selected_features]
        
        elif method == "Chi2":
            X = df.drop(columns=[target_column])
            y = df[target_column]
            chi2_selector = SelectKBest(chi2, k=k)
            chi2_selector.fit(X, y)
            selected_features = chi2_selector.get_support(indices=True)
            selected_columns = X.columns[selected_features]
            logger.info(f"Selected top {k} features based on Chi2: {selected_columns}")
            return df[selected_columns.to_list() + [target_column]]
        
        elif method == "ANOVA":
            X = df.drop(columns=[target_column])
            y = df[target_column]
            anova_selector = SelectKBest(f_classif, k=k)
            anova_selector.fit(X, y)
            selected_features = anova_selector.get_support(indices=True)
            selected_columns = X.columns[selected_features]
            logger.info(f"Selected top {k} features based on ANOVA: {selected_columns}")
            return df[selected_columns.to_list() + [target_column]]
        
        else:
            logger.warning(f"Unknown feature selection method: {method}. No feature selection applied.")
            return df
        
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        return df

def polynomial_features(df, degree=2):
    """
    Generate polynomial features.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    degree (int): The degree of polynomial features.
    
    Returns:
    pd.DataFrame: Dataframe with polynomial features added.
    """
    try:
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree)
        numeric_df = df.select_dtypes(include=[np.number])
        poly_features = poly.fit_transform(numeric_df)
        poly_feature_names = poly.get_feature_names_out(numeric_df.columns)
        
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
        poly_df.index = df.index
        
        non_numeric_df = df.select_dtypes(exclude=[np.number])
        final_df = pd.concat([non_numeric_df, poly_df], axis=1)
        
        logger.info(f"Polynomial features of degree {degree} created successfully.")
        return final_df
    
    except Exception as e:
        logger.error(f"Error creating polynomial features: {e}")
        return df

def interaction_features(df):
    """
    Generate interaction features.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    
    Returns:
    pd.DataFrame: Dataframe with interaction features added.
    """
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        interaction_df = pd.DataFrame(index=df.index)
        
        for i, col1 in enumerate(numeric_df.columns):
            for col2 in numeric_df.columns[i+1:]:
                interaction_df[f'{col1}_x_{col2}'] = numeric_df[col1] * numeric_df[col2]
        
        final_df = pd.concat([df, interaction_df], axis=1)
        
        logger.info("Interaction features created successfully.")
        return final_df
    
    except Exception as e:
        logger.error(f"Error creating interaction features: {e}")
        return df
