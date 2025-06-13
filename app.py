import streamlit as st
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ------------ CONFIGURACIÓN DE LA PÁGINA ------------
st.set_page_config(page_title="Score Crediticio", layout="wide")
st.title("🔍 Calcula tu Score Crediticio")
st.markdown("Ingresa tus datos y descubre tu nivel de riesgo comparado con la población real del dataset.")

# ------------ CARGA DE MODELO Y UTILIDADES ------------
class CreditRiskNNV2(torch.nn.Module):
    def __init__(self, input_dim):
        super(CreditRiskNNV2, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_artifacts():
    with open("Outputs/features.pkl", "rb") as f:
        feature_names = pickle.load(f)

    with open("Outputs/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = CreditRiskNNV2(input_dim=len(feature_names))
    model.load_state_dict(torch.load("Outputs/best_credit_risk_model_v2.pth", map_location=torch.device("cpu")))
    model.eval()

    scorecard_df = pd.read_csv("Outputs/credit_scorecard.csv")

    return model, scaler, feature_names, scorecard_df

model, scaler, feature_names, scorecard_df = load_artifacts()

# ------------ SCORE FUNCTION ACTUALIZADA (SIN REESCALADO) ------------
def prob_to_score(prob, base_score=600, base_odds=1/20, pdo=50, min_score=300, max_score=850):
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    odds = (1 - prob) / prob
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    raw_score = offset + factor * np.log(odds)
    return np.clip(raw_score, min_score, max_score)

# ------------ VARIABLES Y DESCRIPCIONES ------------
categorical_features = {
    "home_ownership": ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"],
    "verification_status": ["Source Verified", "Verified"],
    "purpose": [
        "credit_card", "debt_consolidation", "educational", "home_improvement", "house",
        "major_purchase", "medical", "moving", "other", "renewable_energy", "small_business",
        "vacation", "wedding"
    ],
    "addr_state": [
        "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN",
        "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
        "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA",
        "VT", "WA", "WI", "WV", "WY"
    ],
    "application_type": ["INDIVIDUAL"],
}

numerical_features = [
    "loan_amnt", "int_rate", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_util", "total_acc", "out_prncp",
    "total_rec_prncp", "total_rec_late_fee", "collections_12_mths_ex_med", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim"
]

etiquetas_es = {
    "loan_amnt": "Monto del crédito solicitado",
    "int_rate": "Tasa de interés",
    "annual_inc": "Ingreso anual",
    "dti": "Relación deuda/ingreso",
    "delinq_2yrs": "Moras en los últimos 2 años",
    "inq_last_6mths": "Consultas de crédito (últimos 6 meses)",
    "mths_since_last_delinq": "Meses desde última morosidad",
    "open_acc": "Cuentas de crédito abiertas",
    "pub_rec": "Registros públicos negativos",
    "revol_util": "Uso de crédito rotativo (%)",
    "total_acc": "Total de cuentas en historial",
    "out_prncp": "Principal pendiente",
    "total_rec_prncp": "Principal ya pagado",
    "total_rec_late_fee": "Pagos por mora acumulados",
    "collections_12_mths_ex_med": "Cobros (últimos 12 meses, sin médicos)",
    "acc_now_delinq": "Cuentas en mora actuales",
    "tot_coll_amt": "Total en cobranza",
    "tot_cur_bal": "Saldo actual en cuentas",
    "total_rev_hi_lim": "Límite total de crédito rotativo",
    "home_ownership": "Tipo de vivienda",
    "verification_status": "Verificación de ingresos",
    "purpose": "Propósito del crédito",
    "addr_state": "Estado de residencia",
    "application_type": "Tipo de solicitud"
}

traducciones_categoricas = {
    "home_ownership": {
        "Hipoteca": "MORTGAGE", "Ninguna": "NONE", "Otra": "OTHER", "Propia": "OWN", "Arriendo": "RENT"
    },
    "verification_status": {
        "Fuente verificada": "Source Verified", "Verificado": "Verified"
    },
    "purpose": {
        "Tarjeta de crédito": "credit_card", "Consolidación de deudas": "debt_consolidation",
        "Educación": "educational", "Mejoras del hogar": "home_improvement", "Casa": "house",
        "Compra mayor": "major_purchase", "Gastos médicos": "medical", "Mudanza": "moving",
        "Otro": "other", "Energía renovable": "renewable_energy", "Pequeño negocio": "small_business",
        "Vacaciones": "vacation", "Boda": "wedding"
    },
    "application_type": {
        "Individual": "INDIVIDUAL"
    },
    "addr_state": {state: state for state in categorical_features["addr_state"]}
}

# ------------ GESTIÓN DE PÁGINA Y FORMULARIO ------------
all_features = numerical_features + list(categorical_features.keys())
total_pages = (len(all_features) - 1) // 6 + 1

if "page" not in st.session_state:
    st.session_state.page = 0
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}

start_idx = st.session_state.page * 6
end_idx = min(start_idx + 6, len(all_features))
current_features = all_features[start_idx:end_idx]

st.subheader(f"📝 Datos del usuario – Bloque {st.session_state.page + 1} de {total_pages}")

with st.form(key="formulario"):
    for feature in current_features:
        st.markdown(f"**{etiquetas_es.get(feature, feature)}**")
        
        if feature in numerical_features:
            st.session_state.user_inputs[feature] = st.number_input(
                "", value=st.session_state.user_inputs.get(feature, 0.0), key=feature
            )
        else:
            opciones_es = list(traducciones_categoricas[feature].keys())
            seleccion_es = st.selectbox(
                "", opciones_es,
                index=opciones_es.index(
                    next((k for k, v in traducciones_categoricas[feature].items() if v == st.session_state.user_inputs.get(feature, None)), 0)
                ) if feature in st.session_state.user_inputs else 0,
                key=f"{feature}_select"
            )
            st.session_state.user_inputs[feature] = traducciones_categoricas[feature][seleccion_es]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.session_state.page > 0:
            if st.form_submit_button("⬅️ Anterior"):
                st.session_state.page -= 1
                st.rerun()
    with col2:
        if st.session_state.page < total_pages - 1:
            if st.form_submit_button("➡️ Siguiente"):
                st.session_state.page += 1
                st.rerun()
    with col3:
        if st.session_state.page == total_pages - 1:
            submit = st.form_submit_button("💡 Calcular mi score")

# ------------ RESULTADO ------------
if st.session_state.page == total_pages - 1 and "submit" in locals() and submit:
    input_df = pd.DataFrame([st.session_state.user_inputs])
    input_encoded = pd.get_dummies(input_df)
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_names]
    X_user_scaled = scaler.transform(input_encoded)
    X_tensor = torch.tensor(X_user_scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = model(X_tensor).item()
    score = prob_to_score(prob)

    st.markdown("### 🎯 Resultado del análisis")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.metric(label="Probabilidad de incumplimiento", value=f"{prob:.2%}")
        st.metric(label="Score Crediticio", value=int(score))

        fig, ax = plt.subplots()
        ax.hist(scorecard_df['Score Crediticio'], bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        ax.axvline(score, color="red", linestyle="--", label="Tu score")
        ax.set_title("Distribución de Scores en la Población")
        ax.set_xlabel("Score Crediticio")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        st.pyplot(fig)
