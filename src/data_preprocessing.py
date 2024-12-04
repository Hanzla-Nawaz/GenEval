import logging
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split 
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression 
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')


# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__) 

# Load Data
def load_data(file_path, dtype=None):
    """
    Load data from the given file path.
    Supports CSV, Excel, JSON, and Parquet formats.
    """
    logger.info(f"Loading data from: {file_path}")
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path, dtype=dtype)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path, dtype=dtype)
        elif file_path.endswith(".json"):
            return pd.read_json(file_path)
        elif file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format!")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# Save Data
def save_data(df, file_path):
    """
    Save the DataFrame to a specified file path.
    Supports CSV, Excel, and Parquet formats.
    """
    logger.info(f"Saving data to: {file_path}")
    try:
        if file_path.endswith(".csv"):
            df.to_csv(file_path, index=False)
        elif file_path.endswith(".xlsx"):
            df.to_excel(file_path, index=False)
        elif file_path.endswith(".parquet"):
            df.to_parquet(file_path)
        else:
            raise ValueError("Unsupported file format!")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise



def handle_missing_values(df, method="Simple"):
    logger.info(f"Handling missing values using method: {method}")

    if method == "Drop Rows":
        df = df.dropna()
    elif method == "Drop Columns":
        df = df.dropna(axis=1)
    elif method == "Simple":
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[df.select_dtypes(include=[np.number]).columns] = num_imputer.fit_transform(df.select_dtypes(include=[np.number]))
        df[df.select_dtypes(include=['object']).columns] = cat_imputer.fit_transform(df.select_dtypes(include=['object']))
    elif method == "KNN":
        imputer = KNNImputer(n_neighbors=5)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    else:
        logger.warning(f"Unknown method: {method}. No changes applied.")

    return df

def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    """
    try:
        df = df.drop_duplicates()
        return df
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        raise

def remove_unnecessary_columns(df, columns):
    """
    Remove specified columns from the DataFrame.
    """
    try:
        df = df.drop(columns=columns, errors='ignore')
        return df
    except Exception as e:
        logger.error(f"Error removing columns: {e}")
        raise



# Outlier Handling
def handle_outliers(df, method="IQR", columns=None, z_thresh=3):
    """
    Handle outliers in specified columns.
    """
    logger.info(f"Handling outliers using method: {method}")
    try:
        columns = columns or df.select_dtypes(include=[np.number]).columns
        for column in columns:
            if method == "IQR":
                Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            elif method == "Z-Score":
                df = df[np.abs(zscore(df[column])) < z_thresh]
    except Exception as e:
        logger.error(f"Error handling outliers: {e}")
        raise
    return df


# Scaling and Normalization
def scale_and_normalize(df, method="Standard", columns=None):
    """
    Scale and normalize data.
    """
    logger.info(f"Scaling and normalizing data using method: {method}")
    try:
        scaler = {"Standard": StandardScaler(), "MinMax": MinMaxScaler(), "Robust": RobustScaler()}.get(method)
        if scaler:
            columns = columns or df.select_dtypes(include=[np.number]).columns
            df[columns] = scaler.fit_transform(df[columns])
    except Exception as e:
        logger.error(f"Error scaling and normalizing data: {e}")
        raise
    return df

# Encoding
def encode_categorical(df, method="Label", columns=None):
    """
    Encode categorical variables.
    """
    logger.info(f"Encoding categorical data using method: {method}")
    try:
        columns = columns or df.select_dtypes(include=["object"]).columns
        if method == "Label":
            for col in columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        elif method == "OneHot":
            df = pd.get_dummies(df, columns=columns, drop_first=True)
    except Exception as e:
        logger.error(f"Error encoding categorical data: {e}")
        raise
    return df

def fix_data_types(df, column, dtype):
    df[column] = df[column].astype(dtype)
    return df

def apply_pca(df, method="PCA", n_components=2):
    if method == "PCA":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        numeric_cols = df.select_dtypes(include=np.number)
        reduced_data = pca.fit_transform(numeric_cols)
        return pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(n_components)])
    raise ValueError("Invalid reduction method.")

# # Advanced Text Preprocessing
def text_preprocessing_advanced(text_column, remove_stopwords=True, lemmatize=True, stem=True, sentiment_analysis=True):
    """
    Preprocess text data with advanced techniques.
    """
    logger.info("Applying advanced text preprocessing")
    try:
        stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        sentiment_analyzer = SentimentIntensityAnalyzer()

        def preprocess(text):
            tokens = word_tokenize(str(text).lower())
            tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            if lemmatize:
                tokens = [lemmatizer.lemmatize(word) for word in tokens]
            if stem:
                tokens = [stemmer.stem(word) for word in tokens]
            if sentiment_analysis:
                sentiment_scores = sentiment_analyzer.polarity_scores(text)
                tokens.extend([f"sentiment_{score}" for score, value in sentiment_scores.items() if value > 0.5])
            return " ".join(tokens)

        return text_column.apply(preprocess)
    except Exception as e:
        logger.error(f"Error during text preprocessing: {e}")
        raise


# Time-Series Preprocessing
def time_series_features(df, datetime_column):
    logger.info("Adding time-series features")
    try:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df['year'] = df[datetime_column].dt.year
        df['month'] = df[datetime_column].dt.month
        df['day'] = df[datetime_column].dt.day
        df['weekday'] = df[datetime_column].dt.weekday
        df['hour'] = df[datetime_column].dt.hour
    except Exception as e:
        logger.error(f"Error adding time-series features: {e}")
        raise
    return df

# Handle Class Imbalance
def handle_imbalance(df, target_column, method="SMOTE"):
    logger.info(f"Handling class imbalance using method: {method}")
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        if method == "SMOTE":
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        elif method == "undersample":
            majority = df[df[target_column] == df[target_column].mode()[0]]
            minority = df[df[target_column] != df[target_column].mode()[0]]
            majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
            return pd.concat([majority_downsampled, minority], axis=0)
    except Exception as e:
        logger.error(f"Error handling class imbalance: {e}")
        raise

# Feature Selection
def feature_selection(df, target_column, k=10):
    logger.info("Selecting top features using chi-squared test")
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        selector = SelectKBest(score_func=chi2, k=k)
        X_new = selector.fit_transform(X.select_dtypes(include=[np.number]), y)
        return pd.DataFrame(X_new, columns=X.columns[selector.get_support()])
    except Exception as e:
        logger.error(f"Error during feature selection: {e}")
        raise

# Seasonal Decomposition for Time-Series
def decompose_time_series(df, column, model="additive"):
    logger.info(f"Decomposing time series for column: {column}")
    try:
        result = seasonal_decompose(df[column], model=model, period=12)
        return result
    except Exception as e:
        logger.error(f"Error during time series decomposition: {e}")
        raise