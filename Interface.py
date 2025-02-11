import streamlit as st
import pandas as pd
import joblib
import random
from pyswip import Prolog

def genera_fatti_prolog(sample):
    """ Genera fatti Prolog basati sulle scelte dell'utente """
    utente_id = f"p{random.randint(1000, 9999)}"
    
    fatti = [
        f"persona({utente_id}).",
        f"eta_protetta({utente_id},si)." if sample["Age"] <= 35 else f"eta_protetta({utente_id},no).",
        f"sedentario({utente_id},si)." if sample["FAF"] <= 1.5 else f"sedentario({utente_id},no).",
        f"vita_sociale({utente_id},no)." if sample["TUE"] > 5 else f"vita_sociale({utente_id},si).",
        f"tempo_social({utente_id},alto)." if 4 < sample["TUE"] <= 5 else f"tempo_social({utente_id},basso).",
        f"pasti({utente_id},troppi)." if sample["NCP"] >= 3 else f"pasti({utente_id},pochi).",
        f"beve({utente_id},poco)." if sample["CH2O"] < 1.5 else f"beve({utente_id},molto).",
        f"dorme({utente_id},poco)." if sample["Sleep"] < 7 else f"dorme({utente_id},molto).",
        f"consumo_calorico({utente_id},alto)." if sample["FAVC"] == "yes" else f"consumo_calorico({utente_id},basso).",
        f"fumare({utente_id},si)." if sample["SMOKE"] == "yes" else f"fumare({utente_id},no)."
    ]
    
    with open("kb.pl", "a") as f:
        f.write("\n" + "\n".join(fatti))
    return utente_id

def interroga_prolog(utente_id):
    """ Interroga Prolog per valutare i rischi basati sulle abitudini dell'utente """
    try:
        prolog = Prolog()
        prolog.consult("kb.pl")
        
        query_risultati = {
            "Malattie cardiovascolari": f"malattia({utente_id}, cardiovascolare).",
            "Apnea notturna": f"malattia({utente_id}, apnea_notturna).",
            "Rischio obesità": f"rischio_obesita({utente_id}).",
            "Errori": f"errore({utente_id}, Messaggio)."
        }
        
        risultati = {}
        for key, query in query_risultati.items():
            risultato = list(prolog.query(query))
            if key == "Errori":
                risultati[key] = " | ".join([f"Errore: {erro['Messaggio']}" for erro in risultato]) if risultato else "Nessun vincolo violato"
            else:
                risultati[key] = f"Soggetto a rischio di {key}" if risultato else f"Soggetto non a rischio di {key}"
        
        return risultati
    except Exception as e:
        return {"Errore": f"Errore durante l'interrogazione di Prolog: {str(e)}"}

def main():
    st.title("Predizione Obesità e Inferenza Prolog")
    st.write("Compila i campi per ricevere una valutazione del rischio di obesità e delle malattie associate.")
    
    clf = joblib.load("best_classification_model.pkl")
    reg = joblib.load("best_regression_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    
    age = st.number_input("Età", min_value=10, max_value=100, value=30)
    height = st.number_input("Altezza (m)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
    gender = st.selectbox("Genere", ["Male", "Female"])
    family_history = st.selectbox("Storia familiare di obesità?", ["yes", "no"])
    scc = st.selectbox("Controllo della calorie (SCC)?", ["yes", "no"])
    caec = st.selectbox("Consumo di cibo tra i pasti (CAEC)", ["no", "sometimes", "frequently", "always"])
    faf = st.number_input("Attività fisica (FAF)", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
    tue = st.number_input("Tempo su dispositivi elettronici (TUE)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    ncp = st.number_input("Numero pasti giornalieri (NCP)", min_value=1, max_value=6, value=3)
    ch2o = st.number_input("Litri d'acqua al giorno (CH2O)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
    sleep = st.number_input("Ore di sonno (Sleep)", min_value=0, max_value=12, value=7)
    favc = st.selectbox("Consumo cibi calorici?", ["yes", "no"])
    smoke = st.selectbox("Fumatore?", ["yes", "no"])
    fcvc = st.number_input("Consumo di verdure (FCVC)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    calc = st.selectbox("Consumo di alcool (CALC)", ["no", "sometimes", "frequently", "always"])
    mtrans = st.selectbox("Mezzo di trasporto (MTRANS)", ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"])
    
    if st.button("Esegui Predizione e Inferenza Prolog"):
        user_data = pd.DataFrame([[age, height, gender, family_history, scc, caec, faf, tue, ncp, ch2o, favc, smoke, fcvc, calc, mtrans, sleep]],
                                columns=["Age", "Height", "Gender", "family_history", "SCC", "CAEC", "FAF", "TUE", "NCP", "CH2O", "FAVC", "SMOKE", "FCVC", "CALC", "MTRANS", "Sleep"])
        
        utente_id = genera_fatti_prolog(user_data.iloc[0])
        user_data = user_data.drop(columns=["Sleep"])
        user_data_processed = preprocessor.transform(user_data)
        
        prediction_obesity = clf.predict(user_data_processed)[0]
        prediction_weight = reg.predict(user_data_processed)[0]
        prediction_label = label_encoder.inverse_transform([prediction_obesity])[0]
        
        risultati_prolog = interroga_prolog(utente_id)
        
        st.subheader("Risultati della Predizione")
        st.write(f"**Predizione Obesità:** {prediction_label}")
        st.write(f"**Predizione Peso:** {prediction_weight:.2f} kg")
        
        st.subheader("Inferenza Prolog")
        for key, value in risultati_prolog.items():
            st.write(f"**{key}:** {value}")
            
if __name__ == "__main__":
    main()