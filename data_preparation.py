import pandas as pd
import numpy as np
from sqlalchemy import create_engine

class DataCleaner:
    """
    A class to handle all data cleaning and preparation steps.
    It learns parameters from the training data and applies them to new data.
    """
    def __init__(self):
        self.columns_to_drop = [
            'get_updates_on_dm_content', 'update_me_on_supply_chain_content',
            'i_agree_to_pay_the_amount_through_cheque', 'receive_more_updates_about_our_courses',
            'magazine', 'prospect_id', 'lead_number'
        ]
        self.categorical_impute_values = {}
        self.numerical_impute_values = {}
        self.specialization_mapping = {
            "Human Resource Management": "HR", "Finance Management": "Finance",
            "Marketing Management": "Marketing", "Supply Chain Management": "Operations",
            "IT Projects Management": "IT", "Operations Management": "Operations",
            "International Business": "Business", "Business Administration": "Business",
            "Media and Advertising": "Marketing", "Digital Marketing": "Marketing",
            "Retail Management": "Retail", "Health Care Management": "Health",
            "Hospitality Management": "Hospitality", "Travel and Tourism": "Hospitality",
            "Rural and Agribusiness": "Agribusiness", "Select": "Unknown", "": "Unknown"
        }
        # Add other mappings here...
        self.city_mapping = {
            "Mumbai": "Metro", "Thane & Outskirts": "Metro", "Other Metro Cities": "Metro",
            "Other Cities": "Non-Metro", "Other Cities of Maharashtra": "Non-Metro",
            "Tier II Cities": "Non-Metro", "Select": "Unknown", "": "Unknown"
        }
        self.occupation_mapping = {
            "Unemployed": "Unemployed", "Student": "Student", "Working Professional": "Professional",
            "Businessman": "Business", "Housewife": "Other", "Other": "Other",
            "Select": "Unknown", "": "Unknown"
        }
        self.log_features = ['totalvisits', 'total_time_spent_on_website', 'page_views_per_visit']


    def fit(self, df):
        """
        Learns the imputation values from the training dataframe.
        """
        print("Fitting DataCleaner...")
        # Clean column names first
        df = self._clean_column_names(df)
        
        # Identify column types
        categorical_cols = df.select_dtypes(include=['object']).columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('converted', errors='ignore')

        # Learn imputation values
        for col in categorical_cols:
            self.categorical_impute_values[col] = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        for col in numerical_cols:
            self.numerical_impute_values[col] = df[col].median()
        
        print("DataCleaner fitted successfully.")
        return self

    def transform(self, df):
        """
        Applies the cleaning and preparation steps to the dataframe.
        """
        print("Transforming data...")
        df_copy = df.copy()
        
        # Clean column names
        df_copy = self._clean_column_names(df_copy)
        
        # Drop columns
        df_copy = df_copy.drop(columns=self.columns_to_drop, errors='ignore')
        
        # Replace 'Select' with NaN
        df_copy.replace('Select', np.nan, inplace=True)

        # Impute missing values using learned values
        for col, val in self.categorical_impute_values.items():
            if col in df_copy.columns:
                df_copy[col].fillna(val, inplace=True)
        for col, val in self.numerical_impute_values.items():
             if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                df_copy[col].fillna(val, inplace=True)

        # Apply mappings
        if 'specialization' in df_copy.columns:
            df_copy['specialization'] = df_copy['specialization'].map(self.specialization_mapping).fillna('Unknown')
        if 'city' in df_copy.columns:
            df_copy['city'] = df_copy['city'].map(self.city_mapping).fillna('Unknown')
        if 'what_is_your_current_occupation' in df_copy.columns:
            df_copy['what_is_your_current_occupation'] = df_copy['what_is_your_current_occupation'].map(self.occupation_mapping).fillna('Unknown')

        # Log transformation   
        for col in self.log_features:
            if col in df_copy.columns:
                df_copy[col] = np.log1p(df_copy[col])

        print("Data transformation completed.")
        return df_copy

    def _clean_column_names(self, df):
        """A helper function to clean column names."""
        cols = df.columns
        new_cols = [col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for col in cols]
        df.columns = new_cols
        return df

def load_data(file_path='Lead Scoring.csv'):
    """Loads data from a PostgreSQL database or CSV file as fallback."""
    try:
        # Create SQLAlchemy engine for PostgreSQL
        engine = create_engine("postgresql+psycopg2://postgres:1234@localhost:5432/cap")
        
        # Load data from CSV and upload to PostgreSQL (replace table if exists)
        df_base = pd.read_csv(file_path)
        df_base.to_sql('leadstable', con=engine, if_exists='replace', schema='public', index=False)
        
        # Read data back from PostgreSQL
        query = "SELECT * FROM leadstable"
        df = pd.read_sql_query(query, con=engine)
        print("Data read back from PostgreSQL successfully")
        return df
    except Exception as e:
        print(f"Error during data ingestion: {e}")
        # Fallback to CSV if database fails
        try:
            df = pd.read_csv(file_path)
            print("Loaded data from CSV as fallback")
            return df
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return None
