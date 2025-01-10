import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Task 1 
class DataTransform:
    '''
    Contains methods to convert columns to appropriate format.
    
    Methods:
        transform_dates(self)
        transform_categories(self)
        transform_all(self)
    '''
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
        """Apply all transformations.
        
        Return:
            df: Transformed DataFrame.
        """
        self.transform_dates()
        self.transform_categories()
        return self.df

# Task 2 
class DataFrameInfo:
    '''
    Contains methods to extract information from the DataFrame and its columns.
    
    Methods:
        describe_columns_and_datatypes(self)
        extract_statistical_values(self)
        count_category_distinct_values(self)
        print_dataframe_shape(self)
    '''
    def __init__(self, df):
        self.df = df

    def describe_columns_and_datatypes(self):
        '''Extracts data type and non-null count of each column in dataframe.'''
        self.df.info()
    
    def extract_statistical_values(self):
        '''Calculates count, mean, std, min, , 25%, 50%, 75%, max for each column.'''
        return self.df.describe()
    
    def count_category_distinct_values(self):
        '''Calculates the number of each unique value for category type columns.'''
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
        '''Calculates number of rows and columns in dataframe.'''
        shape = self.df.shape
        print(f'This dataset has {shape[0]} rows and {shape[1]} columns')

# Task 3
class DataFrameTransform():
    '''
    Contains methods to perform EDA transformations on the data. 
    
    Methods:
        check_percentage_of_nulls(self, df)
        drop_columns(self)
        impute_mean(self)
        impute_mode(self)
        drop_rows(self)
        transform_all_nulls(self)
        transform_skewed_columns(self, df, columns)
        remove_outliers(self, df, columns)
        drop_collinear_columns(self, df, columns)
    '''
    def __init__(self, df):
        self.df = df
    
    def check_percentage_of_nulls(self, df):
        '''
        Calculates the percentage of missing values in DataFrame. 
        
        Parameters:
            df: DataFrame the method will be applied to.

        Returns:
            null_percentage: percentage of missing values in columns in the DataFrame 
        '''
        null_percentage = df.isna().mean() * 100
        print(f"Percentage of null values in columns: \n{null_percentage}")
        return null_percentage
    
    def drop_columns(self):
        '''Drops columns with too many missing values.'''
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
        '''Replaces missing values with mean for selected columns. '''
        columns_to_impute_mean = [
            'funded_amount',  
            'int_rate'  
        ]
        for col in columns_to_impute_mean:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
    
    def impute_mode(self):
        '''Replaces missing values with mode for term column'''
        mode_value = self.df['term'].mode()[0]
        self.df['term'] = self.df['term'].fillna(mode_value)

    def drop_rows(self):
        '''Drops rows with missing values for selected columns'''
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
            df: DataFrame the method will be applied to.
            columns: Columns the method will transform.
        
        Returns:
            df: Transformed DataFrame.
        '''
        for col in columns:
            yeojohnson_trans_column = stats.yeojohnson(df[col])
            df[col] = yeojohnson_trans_column[0]
        return df
# Task 5
    def remove_outliers(self, df, columns):
        '''
        Removes calculated outliers from DataFrame.

        Parameters:
            df: DataFrame the method will be applied to.
            columns: Columns the method will transform.
        
        Returns:
            df: Transformed DataFrame.
        '''
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
        '''
        Drops columns identified to be above threshold for collinearity.
        
        Parameters:
            df: DataFrame the method will be applied to.
            columns: Columns the method will transform.
        
        Returns:
            df: Transformed DataFrame.
        '''
        for col in columns:
            df = df.drop(col, axis=1)
        return df

# Task 3
class Plotter():
    '''
    Contains methods to visualise insights from the data.
    
    Methods:
        visualise_nulls_removal(self, df, transformed_df)
        plot_skew_transformations(self, df, columns)
        visualise_outliers(self, df, columns)
        plot_correlation_matrix(self, df)
    '''
    def visualise_nulls_removal(self, df, transformed_df):
        '''
        Plots barchart to visualise removal of missing values.

        Parameters:
            df: Original DataFrame to be visualised in comparison plot.
            transformed df: Transformed DataFrame to be visualised in comparison plot. 
        '''
        null_percentage_before = df.isna().mean() * 100
        null_percentage_after = transformed_df.isna().mean() * 100

        null_comparison_df = pd.DataFrame({
            'Original Percentage of Missing Values (%)': null_percentage_before,
            'Percentage of Missing Values After Transformation (%)': null_percentage_after
        }).fillna(0)

        null_comparison_df.plot(kind='bar', figsize=(10, 4), width=0.8)
        plt.title(f'Percentage of Missing Values Before and After Transformation')
        plt.ylabel('Percentage')
        plt.xlabel('Columns')
        plt.xticks(rotation=45)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
# Task 4     
    def plot_skew_transformations(self, df, columns):
        '''
        Plots histplot to visualise effect of log transformation and yeo-johnson transformation on distribution of data.

        Parameters:
            df: DataFrame to apply transformations to.
            columns: Columns that will be transformed and plotted. 
        '''
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
        '''
        Plots boxpot to visualise outliers in columns of the DataFrame. 

        Parameters:
            df: DataFrame to visualise outliers of. 
            columns: Columns to be plotted. 

        Returns:

        '''
        for col in columns:
            plt.figure(figsize=(5, 2))
            sns.boxplot(x=df[col], color='lightgreen', showfliers=True)
            plt.title(f'Box plot of {col}')
            plt.show()
# Task 6
    def plot_correlation_matrix(self, df):
        '''
        Plots correlation matrix for numerical columns.

        Parameters:
            df: DataFrame to visualise correlation of.
        '''
        sns.heatmap(df.corr(), square=True, linewidths=.5, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.show()
