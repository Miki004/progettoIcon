import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(file_path):
    """ Carica e pulisce il dataset: rimuove duplicati e gestisce outlier """
    data = pd.read_csv(file_path)

    # Rimozione duplicati
    data = data.drop_duplicates()

    # Trasformazione logaritmica per rendere la distribuzione più normale
    data['Age'] = np.log1p(data['Age'])
    data['Weight'] = np.log1p(data['Weight'])

    # Rimozione outlier con IQR
    numerical_features = data.select_dtypes(include=['float64', 'int64'])
    Q1 = numerical_features.quantile(0.25)
    Q3 = numerical_features.quantile(0.75)
    IQR = Q3 - Q1
    data = data.loc[~((numerical_features < (Q1 - 1.5 * IQR)) | (numerical_features > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return data

def encode_features(data):
    """ Applica Label Encoding alla variabile target e One-Hot Encoding alle feature categoriche """
    
    categorical_columns = ['FAVC', 'SCC', 'family_history', 'SMOKE', 'Gender', 'CAEC', 'CALC', 'MTRANS']

    # **1. Encoding della variabile target "Obesity"**
    le_obesity = LabelEncoder()
    data['Obesity'] = le_obesity.fit_transform(data['Obesity'])
    joblib.dump(le_obesity, "obesity_encoder.pkl")

    # **2. One-Hot Encoding delle variabili categoriche**
    data = pd.get_dummies(data, columns=categorical_columns)

    # **3. Salviamo tutte le feature generate nel training**
    joblib.dump(data.columns.tolist(), "model_features.pkl")

    return data

def scale_features(data):
    """ Applica StandardScaler alle feature numeriche (senza toccare la variabile target) """
    features_to_scale = [col for col in data.columns if col not in ["Obesity", "Weight"]]

    scaler = StandardScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    joblib.dump(scaler, "scaler.pkl")

    # Standardizzazione separata del peso
    scaler_weight = StandardScaler()
    data["Weight"] = scaler_weight.fit_transform(data[["Weight"]])
    joblib.dump(scaler_weight, "scaler_weight.pkl")

    return data

def preprocess_data(file_path):
    """ Esegue tutti i passaggi di preprocessing e salva il dataset pronto per il training """
    data = load_and_clean_data(file_path)
    data = encode_features(data)
    data = scale_features(data)

    # Salviamo il dataset preprocessato
    data.to_csv("preprocessed_obesity_data.csv", index=False)
    print("Dataset preprocessato e salvato come 'preprocessed_obesity_data.csv'.")

    return data

# Esegui il preprocessing
dataset_ready = preprocess_data("Obesity prediction.csv")
