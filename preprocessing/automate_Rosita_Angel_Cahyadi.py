import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def full_preprocessing_pipeline(
    df,
    output_csv_path="Credit Score Dataset_preprocessing.csv"
):

    df = df.copy()

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 1. Hapus missing value dan duplikat
    df = df.dropna()
    df = df.drop_duplicates()

    # 2. Feature Definition
    TARGET_COLUMN = "Credit Score"
    num_cols = ["Age", "Income", "Number of Children"]
    cat_cols = ["Gender", "Education", "Marital Status", "Home Ownership"]

    # 3. Split Train-Test
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Imputasi Numerik
    num_imputer = SimpleImputer(strategy="median")
    X_train_num = num_imputer.fit_transform(X_train[num_cols])
    X_test_num = num_imputer.transform(X_test[num_cols])

    # 5. Imputasi Kategorik
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train_cat = cat_imputer.fit_transform(X_train[cat_cols])
    X_test_cat = cat_imputer.transform(X_test[cat_cols])

    # 6. One-Hot Encoding
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat_enc = ohe.fit_transform(X_train_cat)
    X_test_cat_enc = ohe.transform(X_test_cat)

    ohe_features = ohe.get_feature_names_out(cat_cols).tolist()

    # 7. Standarisasi Fitur Numerik
    scaler = StandardScaler()
    X_train_scaled_num = scaler.fit_transform(X_train_num)
    X_test_scaled_num = scaler.transform(X_test_num)

    # 8. Menggabungkan kembali data
    X_train_final = np.hstack([X_train_scaled_num, X_train_cat_enc])
    X_test_final = np.hstack([X_test_scaled_num, X_test_cat_enc])

    final_features = num_cols + ohe_features

    df_train_final = pd.DataFrame(X_train_final, columns=final_features)
    df_test_final = pd.DataFrame(X_test_final, columns=final_features)

    df_train_final[TARGET_COLUMN] = y_train.values
    df_test_final[TARGET_COLUMN] = y_test.values

    df_train_final["is_train"] = 1
    df_test_final["is_train"] = 0

    # 9. Merge & Save
    df_all = pd.concat([df_train_final, df_test_final], axis=0)
    df_all.to_csv(output_csv_path, index=False)

    print("\nPREPROCESSING SELESAI!")
    print(f"File disimpan di: {output_csv_path}")

    return df_all
