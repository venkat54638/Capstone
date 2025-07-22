import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    """
    Builds and manages the preprocessing pipeline for numerical and categorical features.
    """
    def __init__(self):
        self.preprocessor = None
        self.numerical_features = []
        self.categorical_features = []

    def fit(self, X, y):
        """
        Identifies feature types and fits the preprocessing pipeline on the training data.
        """
        print("Fitting Preprocessor...")
        # Identify numerical and categorical features from the training data
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Define transformers
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('target_encoder', TargetEncoder(min_samples_leaf=5, smoothing=1.0))
        ])

        # Combine transformers into a ColumnTransformer
        self.preprocessor = ColumnTransformer([
            ('num', numerical_transformer, self.numerical_features),
            ('cat', categorical_transformer, self.categorical_features)
        ], remainder='passthrough')

        # Fit the preprocessor
        self.preprocessor.fit(X, y)
        print("Preprocessor fitted successfully.")
        return self

    def transform(self, X):
        """
        Applies the fitted preprocessing pipeline to new data.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")
        
        print("Transforming features...")
        X_transformed = self.preprocessor.transform(X)
        
        # Get feature names after transformation for interpretability
        feature_names = self.numerical_features + self.categorical_features
        
        return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
