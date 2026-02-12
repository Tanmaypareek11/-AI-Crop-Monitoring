import pandas as pd

def clean_crop_data(df):
#Cleans raw crop monitoring data.
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df = df.fillna(method="ffill")
    return df
