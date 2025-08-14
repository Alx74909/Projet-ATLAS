import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Chargement du pipeline, sélecteur et modèle ---

# --- Chargement des artefacts depuis une Release GitHub (cache /tmp) ---
import urllib.request
from pathlib import Path

# ⚠️ REMPLACE ces URLs par celles de TA release
ARTIFACT_URLS = {
    "final_preprocessing_pipeline.pkl": "https://github.com/Alx74909/Projet-ATLAS/releases/download/models-v1/final_preprocessing_pipeline.pkl",
    "final_feature_selector.pkl":       "https://github.com/Alx74909/Projet-ATLAS/releases/download/models-v1/final_feature_selector.pkl",
    "final_model_reduced.pkl":          "https://github.com/Alx74909/Projet-ATLAS/releases/download/models-v1/final_model_reduced.pkl",
}

def _fetch(path: Path, url: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(url, path)

@st.cache_resource(show_spinner="Chargement des modèles…")
def load_artifacts():
    cache_dir = Path("/tmp/artifacts")
    objs = []
    for name, url in ARTIFACT_URLS.items():
        dest = cache_dir / name
        _fetch(dest, url)
        with open(dest, "rb") as f:
            objs.append(pickle.load(f))
    return tuple(objs)

pipeline, selector, model = load_artifacts()


st.title("📦 Prédiction des Retards de Livraison")
st.markdown("Entrez les informations de commande pour prédire un risque de retard.")

# --- Interface utilisateur ---
with st.form("form"):
    st.header("Informations Commande")

    order_status = st.selectbox("Statut de commande", ["validated", "not validated", "delivered"])
    order_line_status = st.selectbox("Statut ligne", ["partially delivered", "fully delivered"])
    quantity = st.number_input("Quantité commandée", min_value=1, step=1)
    weight = st.number_input("Poids du produit (kg)", min_value=0.0)
    height = st.number_input("Hauteur produit (cm)", min_value=0.0)
    width = st.number_input("Largeur produit (cm)", min_value=0.0)
    length = st.number_input("Longueur produit (cm)", min_value=0.0)
    category = st.text_input("Catégorie produit")
    container = st.text_input("Type de contenant")
    seller_region = st.text_input("Région vendeur")
    customer_region = st.text_input("Région client")
    weather_level = st.selectbox("Niveau intempérie", ["Non renseigné", "Faible", "Moyenne", "Forte"])
    date_validated = st.date_input("Date de validation")

    submit = st.form_submit_button("Prédire le retard")

if submit:
    # ---- Construction du DataFrame utilisateur
    input_data = pd.DataFrame([{
        "Order Status": order_status,
        "Order Line Status": order_line_status,
        "Order Quantity": quantity,
        "ProductWeight": weight,
        "ProductHeight": height,
        "ProductWidth": width,
        "ProductLength": length,
        "ProductCategory": category,
        "ProductContainerType": container,
        "SellerRegion": seller_region,
        "CustomerRegion": customer_region,
        "Niveau_intempérie": weather_level,
        "Order Validated Date": pd.to_datetime(date_validated)
    }])

    # ---- Features temporelles
    input_data["OrderMonth"] = input_data["Order Validated Date"].dt.month
    input_data["OrderWeekday"] = input_data["Order Validated Date"].dt.weekday
    input_data["OrderYear"] = input_data["Order Validated Date"].dt.year
    input_data.drop(columns=["Order Validated Date"], inplace=True)

    # ---- Colonnes attendues par le pipeline (d'après messages d'erreur)
    required_missing = [
        "Payment Type", "SellerName", "ProductName", "Order Price Amount", "Region",
        "SellerID", "CustomerZipCode", "ProductID", "HasWeatherIssue",
        "Order Delayed Cause", "SellerCity", "ProductPricelist", "CustomerCity",
        "CustomerName", "Order Estimated Delivery Date", "Order Creation Date",
        "SellerCountry", "SellerZipCode", "CustomerID", "SellerCommune",
        "CustomerCommuneCode", "CustomerCountry"
    ]

    # ---- Mappings depuis l'UI
    # Region générique attendue
    if "Region" not in input_data.columns:
        input_data["Region"] = input_data.get("SellerRegion", np.nan)

    # HasWeatherIssue = 1 si intempérie (faible/moyenne/forte) sinon 0
    input_data["HasWeatherIssue"] = input_data["Niveau_intempérie"].isin(["Faible", "Moyenne", "Forte"]).astype(int)

    # ---- Valeurs par défaut
    text_defaults = {
        "Payment Type": "UNKNOWN",
        "SellerName": "UNKNOWN",
        "ProductName": "UNKNOWN",
        "SellerCity": "UNKNOWN",
        "CustomerCity": "UNKNOWN",
        "CustomerName": "UNKNOWN",
        "SellerCountry": "UNKNOWN",
        "SellerCommune": "UNKNOWN",
        "CustomerCountry": "UNKNOWN",
        "Order Delayed Cause": "unknown",
        "Region": "UNKNOWN",
    }
    # Colonnes numériques (ou considérées comme telles par le pipeline)
    numeric_defaults = {
        "Order Price Amount": np.nan,
        "ProductPricelist": np.nan,          # IMPORTANT: pas de 'STANDARD' ici
        "CustomerZipCode": np.nan,
        "SellerZipCode": np.nan,
        "CustomerCommuneCode": np.nan,
        "CustomerID": np.nan,
        "SellerID": np.nan,
        "ProductID": np.nan,
        "HasWeatherIssue": 0,
    }
    # Dates
    date_defaults = {
        "Order Estimated Delivery Date": pd.NaT,
        "Order Creation Date": pd.NaT,
    }

    # Injection si manquantes
    for col in required_missing:
        if col not in input_data.columns:
            if col in text_defaults:
                input_data[col] = text_defaults[col]
            elif col in numeric_defaults:
                input_data[col] = numeric_defaults[col]
            elif col in date_defaults:
                input_data[col] = date_defaults[col]
            else:
                input_data[col] = np.nan

    # Forcer les dates en datetime
    for dcol in ["Order Estimated Delivery Date", "Order Creation Date"]:
        input_data[dcol] = pd.to_datetime(input_data[dcol], errors="coerce")

    # ---- Nettoyage anti "median with non-numeric data"
    # Remplacer chaînes vides / espaces par NaN
    input_data = input_data.replace(r"^\s*$", np.nan, regex=True)

    # Colonnes supposées numériques par le pipeline
    numeric_like_cols = [
        "Order Quantity", "ProductWeight", "ProductHeight", "ProductWidth", "ProductLength",
        "Order Price Amount", "HasWeatherIssue",
        "OrderMonth", "OrderWeekday", "OrderYear",
        "CustomerZipCode", "SellerZipCode", "CustomerCommuneCode",
        "CustomerID", "SellerID", "ProductID",
        "ProductPricelist"
    ]
    for col in numeric_like_cols:
        if col not in input_data.columns:
            input_data[col] = np.nan
        input_data[col] = pd.to_numeric(input_data[col], errors="coerce")

    # ---- Transformations + sélection de features
    transformed = pipeline.transform(input_data)
    reduced = selector.transform(transformed)

    # ---- Prédiction
    prediction = model.predict(reduced)[0]
    prob = model.predict_proba(reduced)[0][1]

    # ---- Affichage
    st.subheader("Résultat de la prédiction :")
    if prediction == 1:
        st.error(f"🚨 Livraison en retard probable (confiance : {prob:.2%})")
    else:
        st.success(f"✅ Livraison à l'heure probable (confiance : {1 - prob:.2%})")
