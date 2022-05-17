import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join('../scripts')))
missing_values = ["n/a", "na", "undefined"]
df = pd.read_csv("../data/AdSmartABdata.csv", na_values=missing_values)


class Clean:

    def __init__(self):
        self.df = df
        pass

    def percent_missing(self):

        # Calculate total number of cells in dataframe
        totalCells = np.product(self.df.shape)

        # Count number of missing values per column
        missingCount = self.df.isnull().sum()

        # Calculate total number of missing values
        totalMissing = missingCount.sum()

        # Calculate percentage of missing values
        print("This dataset contains", round(((totalMissing / totalCells) * 100), 2), "%", "missing values.")

    def missing_values_table(self):
        # Total missing values
        mis_val = self.df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * mis_val / len(self.df)

        # dtype of missing values
        mis_val_dtype = self.df.dtypes

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

        # Sort the table by percentage of missing descending and remove columns with no missing values
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 0] != 0].sort_values(
            '% of Total Values', ascending=False).round(2)

        # Print some summary information
        print("Your selected dataframe has " + str(self.df.shape[1]) + " columns.\n"
                                                                       "There are " + str(
            mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        if mis_val_table_ren_columns.shape[0] == 0:
            return

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

    def fix_missing_value(self, col, value):

        count = self.df[col].isna().sum()
        self.df[col] = self.df[col].fillna(value)
        if type(value) == 'str':
            print(f"{count} missing values in the column {col} have been replaced by '{value}'.")
        else:
            print(f"{count} missing values in the column {col} have been replaced by {value}.")
        return self.df[col]

    # fill the forward value 
    def fix_missing_ffill(self, col):
        self.df[col] = self.df[col].fillna(method='ffill')
        return self.df[col]

    # fill the backward value
    def fix_missing_bfill(self, col):
        self.df[col] = self.df[col].fillna(method='bfill')
        return self.df[col]

    def drop_duplicate(self) -> pd.DataFrame:
        self.df.drop_duplicates(inplace=True)

        return self.df

    def percent_missing_rows(self):

        # Calculate total number rows with missing values
        missing_rows = sum([True for idx, row in self.df.iterrows() if any(row.isna())])

        # Calculate total number of rows
        total_rows = self.df.shape[0]

        # Calculate the percentage of missing rows
        print(round(((missing_rows / total_rows) * 100), 2), "%",
              "of the rows in the dataset contain atleast one missing value.")

    def convert_to_datetime(self, columns):
        for col in columns:
            self.df[col] = pd.to_datetime(self.df[col])

    def convert_to_int(self, columns):
        for col in columns:
            self.df[col] = self.df[col].astype("int64")
            return self.df

    def convert_to_string(self, columns):
        for col in columns:
            self.df[col] = self.df[col].astype("string")

    # drop columns with more than 30% missing values
    def drop_missing_value(self, col):
        perc = 30.0  # in percent %
        min_count = int(((100 - perc) / 100) * self.df.shape[0] + 1)
        self.df.dropna(axis=1, thresh=min_count, inplace=True)

        return self.df
