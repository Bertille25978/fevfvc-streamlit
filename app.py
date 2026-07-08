import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Prédiction FEV/FVC",
    page_icon="🫁",
    layout="centered"
)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/pefura/IFPERA/main/Cameroon_lung_function.csv"
    data = pd.read_csv(url)

    dataset = data.loc[data["sex"] == 2, ["age", "height", "fev", "fvc"]].copy()
    dataset = dataset.dropna()
    dataset = dataset[dataset["fvc"] > 0]
    dataset["fevfvc"] = dataset["fev"] / dataset["fvc"]

    return dataset

@st.cache_resource
def train_models(dataset):
    X = dataset[["age", "height"]]
    y = dataset["fevfvc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=123
    )

    model_mean = GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        n_estimators=100,
        random_state=123
    )

    model_lower = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.05,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        n_estimators=100,
        random_state=123
    )

    model_upper = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.95,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        n_estimators=100,
        random_state=123
    )

    model_mean.fit(X_train, y_train)
    model_lower.fit(X_train, y_train)
    model_upper.fit(X_train, y_train)

    return model_mean, model_lower, model_upper

dataset = load_data()
model_mean, model_lower, model_upper = train_models(dataset)

st.title("Application de prédiction du rapport FEV/FVC")
st.write(
    "Cette application prédit le rapport FEV/FVC chez les sujets de sexe féminin "
    "à partir de l’âge et de la taille."
)

st.sidebar.header("Caractéristiques du sujet")

age = st.sidebar.number_input(
    "Âge en années",
    min_value=5,
    max_value=100,
    value=35,
    step=1
)

height = st.sidebar.number_input(
    "Taille en cm",
    min_value=100,
    max_value=220,
    value=165,
    step=1
)

input_data = pd.DataFrame({
    "age": [age],
    "height": [height]
})

prediction_mean = model_mean.predict(input_data)[0]
prediction_lower = model_lower.predict(input_data)[0]
prediction_upper = model_upper.predict(input_data)[0]

st.subheader("Résultat de la prédiction")

st.metric(
    label="Prédiction moyenne du rapport FEV/FVC",
    value=f"{prediction_mean:.3f}"
)

st.write("Intervalle prédictif approximatif à 90 % :")

col1, col2 = st.columns(2)

with col1:
    st.metric("Limite inférieure", f"{prediction_lower:.3f}")

with col2:
    st.metric("Limite supérieure", f"{prediction_upper:.3f}")

st.subheader("Données saisies")

st.dataframe(input_data)

st.info(
    "Les limites inférieure et supérieure correspondent à une approximation "
    "d’un intervalle prédictif à 90 %, obtenue par régression quantile."
)
