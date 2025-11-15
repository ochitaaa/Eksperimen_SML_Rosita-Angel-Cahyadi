import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import os

def full_preprocessing_pipeline(
    df, 
    output_csv_path="/content/drive/MyDrive/Colab Notebooks/spotifychurn_preprocessing.csv"
):
    df = df.copy()

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 1. Hapus missing value & duplikat
    df = df.dropna().drop_duplicates()

    # 2. Membuat Feature Engineering
    
    # Engagement Score (loyal vs casual)
    df["engagement_score"] = (
        0.6 * df["listening_time"] +
        0.4 * df["songs_played_per_day"]
    )

    # Churn Risk Score (passive vs active)
    df["churn_risk_score"] = (
        0.7 * df["skip_rate"] -
        0.3 * df["offline_listening"]
    )

    # Consumption Efficiency (effective listening)
    df["consumption_efficiency"] = (
        df["songs_played_per_day"] / (df["listening_time"].abs() + 1)
    )

    # 3. Encoding fitur kategorikal
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # 4. Standarisasi hanya fitur numerik
    categorical_features = ['gender', 'country', 'subscription_type', 'device_type']

    num_cols = [
        col for col in df.select_dtypes(include=['int64', 'float64']).columns
        if col not in categorical_features + ['is_churned']
    ]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 5. Periksa outlier
    z_scores = np.abs(stats.zscore(df[num_cols]))
    df_clean = df[(z_scores < 3).all(axis=1)]

    # 6. Simpan hasil preprocessing
    df_clean.to_csv(output_csv_path, index=False, header=True)

    print("Preprocessing selesai")
    print(f"Dataset tersimpan di: {output_csv_path}")

    return df_clean
