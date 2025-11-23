import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def full_preprocessing_pipeline(
    df,
    output_csv_path="diabetes_dataset_2019_preprocessing.csv"
):
    df = df.copy()
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 1. Feature Definition
    TARGET_COLUMN = 'Diabetic'
    num_cols = ['BMI', 'Sleep', 'SoundSleep', 'Pregancies']
    ordinal_cols = [
        'Age', 'PhysicallyActive', 'Stress', 'JunkFood',
        'UriationFreq', 'BPLevel'
    ]
    nominal_cols = [
        'Gender', 'Family_Diabetes', 'highBP', 'Smoking',
        'Alcohol', 'RegularMedicine', 'Pdiabetes'
    ]

    # 2. Split Train-Test
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Ordinal Encoding untuk fitur ordinal
    ordinal_mappings = {
        'Age': {'less than 40': 0, '40-49': 1, '50-59': 2, '60 or older': 3},
        'PhysicallyActive': {
            'not active': 0,
            'less than half an hour': 1,
            'more than half an hour': 2,
            'one hour or more': 3
        },
        'Stress': {
            'not at all': 0,
            'sometimes': 1,
            'very often': 2,
            'always': 3
        },
        'JunkFood': {
            'occasionally': 0,
            'often': 1,
            'very often': 2,
            'always': 3
        },
        'UriationFreq': {
            'not much': 0,
            'quite often': 1
        },
        'BPLevel': {
            'Low': 0,
            'Normal': 1,
            'High': 2
        }
    }

    for col, mapping in ordinal_mappings.items():
        X_train[col] = X_train[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)

    # 4. One-Hot Encoding untuk fitur nominal
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    X_train_ohe = ohe.fit_transform(X_train[nominal_cols])
    X_test_ohe = ohe.transform(X_test[nominal_cols])

    ohe_feature_names = ohe.get_feature_names_out(nominal_cols)

    # 5. Standarisasi fitur numerik
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_num_scaled = scaler.transform(X_test[num_cols])

    # 6. Gabung semua fitur
    X_train_final = np.hstack([
        X_train_num_scaled,
        X_train[ordinal_cols].values,
        X_train_ohe
    ])
    X_test_final = np.hstack([
        X_test_num_scaled,
        X_test[ordinal_cols].values,
        X_test_ohe
    ])

    # 7. Gabungkan semua fitur
    final_features = num_cols + ordinal_cols + list(ohe_feature_names)

    df_train_final = pd.DataFrame(X_train_final, columns=final_features)
    df_train_final[TARGET_COLUMN] = y_train.values
    df_train_final["is_train"] = 1

    df_test_final = pd.DataFrame(X_test_final, columns=final_features)
    df_test_final[TARGET_COLUMN] = y_test.values
    df_test_final["is_train"] = 0

    # 8. Gabung dan simpan
    df_all = pd.concat([df_train_final, df_test_final], axis=0).reset_index(drop=True)
    df_all.to_csv(output_csv_path, index=False)

    print("\nPREPROCESSING SELESAI!")
    print(f"File disimpan di: {output_csv_path}")
    
    return df_all