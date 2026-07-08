import streamlit as st
import pandas as pd
import joblib

model_mean = joblib.load("model_mean_fevfvc.pkl")
model_lower = joblib.load("model_lower_fevfvc.pkl")
model_upper = joblib.load("model_upper_fevfvc.pkl")

st.set_page_config(page_title="Prédiction FEV/FVC", layout="centered")
st.title("Prédiction du rapport FEV/FVC chez les femmes")
st.write("Modèle Gradient Boosting utilisant l’âge et la taille.")

st.sidebar.header("Caractéristiques du sujet")
age = st.sidebar.number_input("Âge en années", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
height = st.sidebar.number_input("Taille en cm", min_value=80.0, max_value=220.0, value=160.0, step=1.0)

if st.sidebar.button("Prédire"):
    new_data = pd.DataFrame({"age": [age], "height": [height]})
    pred_mean = model_mean.predict(new_data)[0]
    pred_lower = model_lower.predict(new_data)[0]
    pred_upper = model_upper.predict(new_data)[0]

    st.subheader("Résultats")
    st.write(f"**Prédiction moyenne FEV/FVC :** {pred_mean:.3f}")
    st.write(f"**Limite inférieure IC 90 % :** {pred_lower:.3f}")
    st.write(f"**Limite supérieure IC 90 % :** {pred_upper:.3f}")

    resultats = pd.DataFrame({
        "Âge": [age],
        "Taille": [height],
        "FEV/FVC prédit": [round(pred_mean, 3)],
        "Limite inférieure 90%": [round(pred_lower, 3)],
        "Limite supérieure 90%": [round(pred_upper, 3)]
    })
    st.dataframe(resultats)

st.caption("Gradient Boosting Regressor + régression quantile pour l’intervalle à 90 %.")
