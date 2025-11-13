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

    # Pastikan direktori simpan ada
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 1. Hapus missing value & duplikat
    df = df.dropna().drop_duplicates()

    # 2. Encoding fitur kategorikal
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # 3. Standarisasi hanya fitur numerik
    categorical_features = ['gender', 'country', 'subscription_type', 'device_type']

    num_cols = [
        col for col in df.select_dtypes(include=['int64', 'float64']).columns
        if col not in categorical_features + ['is_churned']
    ]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 4. Periksa outlier
    z_scores = np.abs(stats.zscore(df[num_cols]))
    df = df[(z_scores < 3).all(axis=1)]

    # 5. Simpan hasil preprocessing
    df.to_csv(output_csv_path, index=False, header=True)

    print("Preprocessing selesai")
    print(f"Dataset tersimpan di: {output_csv_path}")

    return df
