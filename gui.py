import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pyswip import Prolog
import random


# **1. Caricamento dei modelli e scaler**
best_classification_model = joblib.load("best_classification_model.pkl")
best_regression_model = joblib.load("best_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
scaler_weight = joblib.load("scaler_weight.pkl")
le_obesity = joblib.load("obesity_encoder.pkl")
model_features = joblib.load("model_features.pkl")  # Feature corrette dal training

# **2. Interfaccia Streamlit**
st.title("Obesity Prediction System")

# **3. Input utente**
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=5, max_value=100, step=1)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, step=0.01)
family_history = st.selectbox("Family History of Obesity", ["yes", "no"])
favc = st.selectbox("Frequent High-Calorie Food Consumption", ["yes", "no"])
fcvc = st.slider("Frequency of Vegetable Consumption (1-3)", 1.0, 3.0, step=0.1)
ncp = st.slider("Number of Main Meals (1-4)", 1, 4, step=1)
caec = st.selectbox("Consumption of Food Between Meals", ["Always", "Frequently", "Sometimes", "no"])
smoke = st.selectbox("Do you Smoke?", ["yes", "no"])
ch2o = st.slider("Daily Water Consumption (liters)", 0.5, 3.0, step=0.1)
scc = st.selectbox("Calories Consumption Monitoring?", ["yes", "no"])
faf = st.slider("Physical Activity Frequency (0-3)", 0.0, 3.0, step=0.1)
sleep = st.slider("Sleep time", 0.0, 12.0, step=0.1)
tue = st.slider("Time Spent Using Technology (hours per day, 0-10)", 0.0, 10.0, step=0.1)
calc = st.selectbox("Alcohol Consumption Frequency", ["Sometimes", "Frequently", "no"])
mtrans = st.selectbox("Transportation Mode", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

def genera_fatti_prolog(sample):
    """ Genera fatti Prolog basati sulle scelte dell'utente """
    utente_id = f"p{random.randint(1,1000)}"
    row = sample.iloc[0]

    fatti = [
        f"persona({utente_id}).",
        f"eta_protetta({utente_id},si)." if row["Age"] <= 35 else f"eta_protetta({utente_id},no).",
        f"sedentario({utente_id},si)." if row["FAF"] <= 1.5 else f"sedentario({utente_id},no).",
        f"vita_sociale({utente_id},no)." if row["TUE"] > 5 else f"vita_sociale({utente_id},si).",
        f"tempo_social({utente_id}, alto)." if 4 < row["TUE"] <= 5 else f"tempo_social({utente_id}, basso).",
        f"pasti({utente_id},troppi)." if row["NCP"] >= 3 else f"pasti({utente_id},pochi).",
        f"beve({utente_id},poco)." if row["CH2O"] < 1.5 else f"beve({utente_id},molto).",
        f"dorme({utente_id},poco)." if row["Sleep"] < 7 else f"dorme({utente_id},molto).",
        f"consumo_calorico({utente_id},alto)." if row["FAVC"].lower() == "yes" else f"consumo_calorico({utente_id},basso).",
        f"fumare({utente_id},si)." if row["SMOKE"].lower() == "yes" else f"fumare({utente_id},no)."
    ]

    with open("kb.pl", "a") as f:
        f.write("\n" + "\n".join(fatti))
    return utente_id

def interroga_prolog(utente_id):
    """ Interroga Prolog per valutare i rischi basati sulle abitudini dell'utente """
    try:
        prolog = Prolog()
        prolog.consult("kb.pl")
        
        query_stile = f"malattia({utente_id}, cardiovascolare)."
        query_apnea = f"malattia({utente_id}, apnea_notturna)."
        
        risultato_stile = list(prolog.query(query_stile))
        risultato_apnea = list(prolog.query(query_apnea))
        
        output_stile = "Soggetto a rischio di malattie cardiovascolari" if risultato_stile else "Soggetto non a rischio"
        output_apnea = "Il Soggetto potrebbe soffrire di apnea notturna" if risultato_apnea else "Sonno normale"
        
        return f"{output_stile} | {output_apnea}"
    except Exception as e:
        return f"Errore durante l'interrogazione di Prolog: {str(e)}"
    
# **4. Predizione al click**
if st.button("Predict"):
    # **Creazione dataframe utente**
    sample = pd.DataFrame([{ 
        'Gender': gender,
        'Age': age,
        'Height': height,
        'family_history': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'Sleep': sleep,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }])
    utente_id = genera_fatti_prolog(sample)
    risultato_prolog = interroga_prolog(utente_id)
    # **5. One-Hot Encoding**
    sample = pd.get_dummies(sample)

    # **6. Aggiunta delle feature mancanti con valore 0**
    for col in model_features:
        if col not in sample.columns:
            sample[col] = 0  

    # **7. Rimozione delle feature extra**
    sample = sample[model_features]  

    # **8. Verifica che il numero di colonne combaci**
    if sample.shape[1] != len(model_features):
        st.error(f"Feature mismatch! Il modello si aspetta {len(model_features)} feature, ma ne ha ricevute {sample.shape[1]}")
        st.stop()

    # **9. Riordinare le colonne per combaciare con quelle viste dallo `StandardScaler`**
    sample = sample[scaler.feature_names_in_]

    # **10. Standardizzazione**
    sample_scaled = scaler.transform(sample)

    # **11. Predizione Obesity**
    obesity_prediction = best_classification_model.predict(sample_scaled)[0]
    obesity_category = le_obesity.inverse_transform([obesity_prediction])[0]

    # **12. Predizione del Peso**
    weight_prediction_scaled = best_regression_model.predict(sample_scaled).reshape(-1, 1)

# **Correzione della denormalizzazione**
    weight_prediction = scaler_weight.inverse_transform(weight_prediction_scaled)
    weight_prediction = float(weight_prediction[0, 0])  # Assicuriamoci che sia un singolo valore scalare

    # **13. Output**
    st.subheader("Predicted Results")
    st.write(f"Predicted Obesity Class: {obesity_category}")
    st.write(f"Predicted Weight: {round(weight_prediction, 2)*100} kg")
    st.write(f"**Prolog Analysis:** {risultato_prolog}")
