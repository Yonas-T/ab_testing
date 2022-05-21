import pandas as pd
import logging


def save_csv(df, csv_path, index=False):
    try:
        df.to_csv(csv_path, index=index)
        logging.info(f'Csv file saved in {csv_path}')
    except Exception:
        logging.exception('File saving failed.')


def read_csv(csv_path, missing_values=[]):
    try:
        df = pd.read_csv(csv_path, na_values=missing_values)
        logging.info(f'Csv file read from {csv_path}')
        return df
    except FileNotFoundError:
        logging.exception('File not found.')


df = read_csv("../data/AdSmartABdata.csv")
df.drop('platform_os', inplace=True, axis=1)
save_csv(df, "../data/AdSmartABdata.csv")
