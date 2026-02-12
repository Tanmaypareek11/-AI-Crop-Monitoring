def generate_features(df):

# Performs feature engineering on cleaned crop data.
    if "severity_score" in df.columns:
        max_val = df["severity_score"].max()
        if max_val != 0:
            df["severity_index"] = df["severity_score"] / max_val
    return df
