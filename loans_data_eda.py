#%%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Task 1 
class DataTransform:
    def __init__ (self, df):
        self.df = df

    def transform_dates(self):
        """Convert object columns to datetime format."""
        date_columns = [
            'last_credit_pull_date', 
            'next_payment_date', 
            'last_payment_date', 
            'earliest_credit_line', 
            'issue_date'
        ]
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], format='%b-%Y', errors='coerce')

    def transform_categories(self):
        """Convert columns to category format."""
        category_columns = [
            'term',
            'grade', 
            'sub_grade',
            'employment_length',
            'home_ownership', 
            'verification_status', 
            'loan_status',
            'purpose', 
            'application_type'
        ]
        for col in category_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
    
    def transform_all(self):
        """Apply all transformations."""
        self.transform_dates()
        self.transform_categories()
        return self.df

# Task 2 
class DataFrameInfo:
    def __init__(self, df):
        self.df = df

    def describe_columns_and_datatypes(self):
        ''''''
        self.df.info()
    
    def extract_statistical_values(self):
        ''''''
        return self.df.describe()
    
    def count_category_distinct_values(self):
        ''''''
        category_columns = [
            'term',
            'grade', 
            'sub_grade',
            'employment_length',
            'home_ownership', 
            'verification_status', 
            'loan_status',
            'purpose', 
            'application_type'
        ]
        for col in category_columns:
            if col in self.df.columns:
                print("Distinct values count for column '{col}':")
                print(self.df[col].value_counts(), "\n")
    
    def print_dataframe_shape(self):
        ''''''
        shape = self.df.shape
        print(f'This dataset has {shape[0]} rows and {shape[1]} columns')

    def percentage_of_nulls(self):
        ''''''
        print("Percentage of missing values in each column:")
        return self.df.isna().mean() * 100

# Task 3
class DataFrameTransform():
    def __init__(self, df):
        self.df = df
    
    def check_percentage_of_nulls(self):
        ''''''
        null_percentage = self.df.isna().mean() * 100
        print(f"Percentage of null values in columns: \n{null_percentage}")
        return null_percentage
    
    def drop_columns(self):
        ''''''
        columns_to_drop = [
            'mths_since_last_delinq', 
            'mths_since_last_record', 
            'mths_since_last_major_derog', 
            'next_payment_date'
        ]
        for col in columns_to_drop:
            if col in self.df.columns:
                self.df = self.df.drop(col, axis=1)

    def impute_mean(self):
        ''''''
        columns_to_impute_mean = [
            'funded_amount',  
            'int_rate',  
        ]
        for col in columns_to_impute_mean:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
    
    def impute_mode(self):
        ''''''
        mode_value = self.df['term'].mode()[0]
        self.df['term'] = self.df['term'].fillna(mode_value)

    def drop_rows(self):
        ''''''
        columns_to_drop_rows = [
            'last_payment_date',
            'last_credit_pull_date',
            'collections_12_mths_ex_med',
            'employment_length'
        ]
        self.df = self.df.dropna(subset=columns_to_drop_rows)
    
    def transform_all_nulls(self):
        """Apply all transformations."""
        self.drop_columns()
        self.impute_mean()
        self.impute_mode()
        self.drop_rows()
        return self.df
# Task 4
    def transform_skewed_columns(self, df, columns):
        '''
        Transforms skewed columns with Yeo Johnson tranformation.
        
        Parameters:
        df: The DataFrame to transform.
        columns: List of the columns to transform.
        
        Returns:
        The DataFrame with the columns transformed.
        '''
        for col in columns:
            yeojohnson_trans_column = stats.yeojohnson(df[col])
            df[col] = yeojohnson_trans_column[0]
        return df
# Task 5
    def remove_outliers(self, df, columns):
        ''''''
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df
# Task 6  
    def drop_collinear_columns(self, df, columns):
        ''''''
        for col in columns:
            df = df.drop(col, axis=1)
        return df

# Task 3
class Plotter():
    def visualise_nulls_removal(self, df, transformed_df):
        ''''''
        null_percentage_before = df.isna().mean() * 100
        null_percentage_after = transformed_df.isna().mean() * 100

        fig, axes = plt.subplots(1,2, figsize=(10, 5))

        sns.barplot(null_percentage_before, ax=axes[0])
        axes[0].set_title('Percentage of Missing Values Before Transformation')
        plt.xticks(rotation=45)

        sns.barplot(null_percentage_after, ax=axes[1])
        axes[1].set_title('Percentage of Missing Values After Transformation')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
# Task 4     
    def plot_skew_transformations(self, df, columns):
        ''''''
        for col in columns:
            fig, axes = plt.subplots(1,3, figsize=(8, 2))

            sns.histplot(df[col], kde=True, ax=axes[0])
            axes[0].set_title(f'Original skewness: {df[col].skew():.2f}')
            axes[0].set_xlabel(col)

            log_transformed_column = df[col].apply(lambda x: np.log(x) if x > 0 else 0)
            sns.histplot(log_transformed_column, kde=True, ax=axes[1])
            axes[1].set_title(f'Log transform: {log_transformed_column.skew():.2f}')
            axes[1].set_xlabel(col)

            yeojohnson_transformed_column = df[col]
            yeojohnson_transformed_column = stats.yeojohnson(yeojohnson_transformed_column)
            yeojohnson_transformed_column = pd.Series(yeojohnson_transformed_column[0])
            sns.histplot(yeojohnson_transformed_column, kde=True, ax=axes[2])
            axes[2].set_title(f'Yeo-johnson transform: {yeojohnson_transformed_column.skew():.2f}')
            axes[2].set_xlabel(col)

        plt.show()
# Task 5
    def visualise_outliers(self, df, columns):
        ''''''
        for col in columns:
            plt.figure(figsize=(5, 2))
            sns.boxplot(x=df[col], color='lightgreen', showfliers=True)
            plt.title(f'Box plot of {col}')
            plt.show()
# Task 6
    def plot_correlation_matrix(self, df):
        ''''''
        sns.heatmap(df.corr(), square=True, linewidths=.5, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.show()
