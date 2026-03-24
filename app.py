
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Gradient boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Explicabilité
import shap
shap.initjs()


# ========================== THEME & CSS GLOBAL ==============================
def set_custom_style():
    st.markdown("""
        <style>

        /* ------- Fonts: Times New Roman ------- */
        @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman:wght@300;400;600;700&display=swap');
        html, body, div, span, p, label {
            font-family: 'Times New Roman', serif !important;
        }

        /* ------- Background ------- */
        .main {
            background-color: #f5f7fa !important;
        }

        /* ------- Sidebar ------- */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #dcdde1;
        }

        /* ------- Titles ------- */
        h1 {
            font-size: 38px !important;
            font-weight: 700 !important;
            text-align: center !important;
            color: #2c3e50 !important;
        }
        h2 {
            font-size: 28px !important;
            font-weight: 600 !important;
            color: #34495e !important;
        }
        h3 {
            font-size: 22px !important;
            font-weight: 600 !important;
            color: #34495e !important;
        }

        /* ------- Modern Tabs ------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: #ecf0f1 !important;
            padding: 10px 25px !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 18px !important;
            color: #2c3e50 !important;
            border: 1px solid #dfe6e9 !important;
            transition: 0.25s !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: #e2e6eb !important;
            transform: translateY(-2px);
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg,#3498db,#2980b9) !important;
            color: white !important;
            font-weight: 700 !important;
            border: 1px solid #2980b9 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        }

        /* ------- Plot container ------- */
        .plot-container {
            background: white !important;
            border-radius: 15px !important;
            padding: 15px !important;
            box-shadow: 0 3px 12px rgba(0,0,0,0.08) !important;
        }

        /* ------- Metrics cards ------- */
        .stMetric {
            background-color: #ffffff !important;
            border-radius: 12px !important;
            padding: 15px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
        }

        </style>
    """, unsafe_allow_html=True)

set_custom_style()

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def preprocess_data(df):
    """Encodage, outliers, normalisation"""
    df = df.copy()

    numerical = ['person_age','person_income','person_emp_length',
                 'loan_amnt','loan_int_rate','loan_percent_income']

    categorical = ['person_home_ownership','loan_intent','person_prev_default']

    # Encodage one-hot
    df = pd.get_dummies(df, columns=categorical, drop_first=True)

    # Traitement des outliers (IQR + clipping)
    for col in numerical:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

    # Normalisation
    scaler = StandardScaler()
    df[numerical] = scaler.fit_transform(df[numerical])

    return df, scaler


def train_models(X_train, y_train):
    """Entraîne tous les modèles"""

    models = {
        "Régression Logistique": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss'
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


def evaluate_model(model, X_test, y_test):
    """Retourne toutes les métriques d’un modèle"""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "AUC": roc_auc_score(y_test, proba)
    }, preds, proba

# =============================================================================
# UI STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Application ML - Risque de Crédit",
    layout="wide"
)

#st.title("📊 Application Machine Learning - Analyse du Risque de Crédit")
#st.markdown("Développé par **Cheikh SENE** 🎯")


# ====================== HEADER MODERNE ======================
st.markdown("""
<div style="
    padding: 18px;
    background-color:#3498db;
    border-radius: 12px;
    text-align:center;
    margin-bottom:25px;
">
    <h1 style="color:white; margin:0;">
        📊 Application Machine Learning – Analyse du Risque Crédit
    </h1>
    <p style="color:white; margin-top:8px; font-size:20px;">
        Développée par <strong>Cheikh SENE</strong> 🚀
    </p>
</div>
""", unsafe_allow_html=True)


menu = st.sidebar.selectbox(
    "Navigation",
    [
        "🏁 Importation des données",
        "📈 EDA interactive",
        "📊 Analyse univariée",
        "📊 Analyse bivariée",
        "🌐 Analyse multivariée avancée",
        "🤖 Modèles & Performances",
        "🧩 Matrices de confusion",
        "⚙️ Ajustement du seuil",
        "🎯 Seuil optimal",
        "🔮 Prédiction individuelle",
        "🧠 Explicabilité (SHAP)"
    ]
)


# =============================================================================
# 1. IMPORT DES DONNÉES (VERSION AUTOMATIQUE)
# =============================================================================

if menu == "🏁 Importation des données":

    st.header("📊 Chargement automatique du dataset")

    @st.cache_data
    def load_data():
        df = pd.read_csv("credit_risk_dataset.csv", sep=";")
        return df

    df = load_data()
    st.session_state["raw_data"] = df

    st.success("✅ Dataset chargé automatiquement depuis GitHub.")

    st.subheader("Aperçu des données")
    st.dataframe(df.head())

# =============================================================================
# 2. EDA INTERACTIVE
# =============================================================================

elif menu == "📈 EDA interactive":

    if "raw_data" not in st.session_state:
        st.error("Veuillez d'abord importer un fichier CSV.")
        st.stop()

    df = st.session_state["raw_data"]

    st.header("📈 EDA - Analyse des données")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution d'une variable numérique")
        num_var = st.selectbox("Variable", df.select_dtypes('number').columns)
        fig = px.histogram(df, x=num_var, nbins=40)
        st.plotly_chart(fig)

    with col2:
        st.subheader("Variable cible : loan_status")
        fig = px.pie(df, names="loan_status")
        st.plotly_chart(fig)


# =============================================================================
# 3. MODELISATION
# =============================================================================

elif menu == "🤖 Modèles & Performances":

    if "raw_data" not in st.session_state:
        st.error("Veuillez d'abord importer un fichier CSV.")
        st.stop()

    df = st.session_state["raw_data"]

    st.header("🤖 Entraînement & comparaison des modèles")

    df_processed, scaler = preprocess_data(df.copy())

    X = df_processed.drop("loan_status", axis=1)
    y = df_processed["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)

    st.subheader("📊 Résultats des modèles")

    results = {}
    for name, model in models.items():
        metrics, preds, proba = evaluate_model(model, X_test, y_test)
        results[name] = metrics

    results_df = pd.DataFrame(results).T
    st.dataframe(results_df.style.highlight_max(color="lightgreen"))

    st.session_state["models"] = models
    st.session_state["scaler"] = scaler
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test


# =============================================================================
# 4. PREDICTION INDIVIDUELLE
# =============================================================================

elif menu == "🔮 Prédiction individuelle":

    if "models" not in st.session_state:
        st.error("Veuillez entraîner les modèles d'abord.")
        st.stop()

    st.header("🔮 Prédiction individuelle")

    model_choice = st.selectbox("Modèle", list(st.session_state["models"].keys()))
    model = st.session_state["models"][model_choice]

    X_train = st.session_state["X_train"]  # Pour récupérer les colonnes finales
    scaler = st.session_state["scaler"]

    st.subheader("✅ Entrez les informations du client")

    # ---- 1. Variables numériques ----
    num_values = {}
    num_values["person_age"] = st.number_input("Âge", 18, 100, 30)
    num_values["person_income"] = st.number_input("Revenu annuel", 0, 500000, 50000)
    num_values["person_emp_length"] = st.number_input("Ancienneté (années)", 0, 50, 2)
    num_values["loan_amnt"] = st.number_input("Montant du prêt", 0, 50000, 5000)
    num_values["loan_int_rate"] = st.number_input("Taux d'intérêt (%)", 0.0, 40.0, 10.0)
    num_values["loan_percent_income"] = st.number_input("Ratio montant/revenu", 0.0, 1.0, 0.2)

    # ---- 2. Variables catégorielles ----
    home = st.selectbox("Statut logement", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    intent = st.selectbox("Intent du prêt", 
                          ["EDUCATION","MEDICAL","VENTURE","PERSONAL","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])
    prev_default = st.selectbox("Historique de défaut", ["No", "Yes"])

    # ---- 3. Construction d'un DF avec EXACTEMENT les colonnes du modèle ----
    df_input = pd.DataFrame(columns=X_train.columns)
    df_input.loc[0] = 0  # Remplir toutes les colonnes à 0

    # Ajouter les valeurs numériques
    for col, val in num_values.items():
        df_input[col] = val

    # Ajouter les colonnes catégorielles encodées
    if f"person_home_ownership_{home}" in df_input.columns:
        df_input[f"person_home_ownership_{home}"] = 1

    if f"loan_intent_{intent}" in df_input.columns:
        df_input[f"loan_intent_{intent}"] = 1

    if f"person_prev_default_{prev_default}" in df_input.columns:
        df_input[f"person_prev_default_{prev_default}"] = 1

    # ---- 4. Normalisation des colonnes numériques ----
    numeric_cols = ["person_age","person_income","person_emp_length",
                    "loan_amnt","loan_int_rate","loan_percent_income"]

    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

    # ---- 5. Prédiction ----
    if st.button("Prédire"):
        proba = model.predict_proba(df_input)[0][1]

        st.metric("Probabilité de défaut", f"{proba*100:.2f}%")

        if proba > 0.5:
            st.error("⚠️ Risque élevé de défaut")
        else:
            st.success("✅ Risque faible")

#==============================================================================
#Matrices de confusion
#==============================================================================

elif menu == "🧩 Matrices de confusion":

    if "models" not in st.session_state:
        st.error("Veuillez entraîner les modèles d'abord.")
        st.stop()

    st.header("🧩 Matrices de confusion & Interprétation")

    # Selection du modèle
    model_choice = st.selectbox(
        "Choisir un modèle",
        list(st.session_state["models"].keys())
    )

    model = st.session_state["models"][model_choice]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    # Prédictions
    preds = model.predict(X_test)

    # Matrice de confusion
    cm = confusion_matrix(y_test, preds)
    TN, FP, FN, TP = cm.ravel()

    st.subheader(f"Matrice de confusion – {model_choice}")
    st.write(cm)

    # Heatmap graphique
    fig = px.imshow(
        cm,
        labels=dict(x="Prédiction", y="Vérité terrain", color="Nombre"),
        x=["Non Défaut", "Défaut"],
        y=["Non Défaut", "Défaut"],
        text_auto=True,
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig)

    st.markdown("---")
    st.subheader("📘 Interprétation détaillée")

    # Explication automatique des cases
    st.markdown(f"""
    ### ✅ TN — Vrais Négatifs : `{TN}`
    Le modèle a correctement prédit **pas de défaut**.

    ### ✅ TP — Vrais Positifs : `{TP}`
    Le modèle a correctement identifié les clients **à risque (défaut)**.

    ### ⚠️ FP — Faux Positifs : `{FP}`
    Le modèle pense que le client va faire défaut alors que **non**.  
    👉 Conséquence en scoring : **refuser un bon client** (perte commerciale).

    ### ⚠️ FN — Faux Négatifs : `{FN}`
    Le modèle prédit “pas de défaut” alors que le client **va faire défaut**.  
    👉 Conséquence : **risque financier direct pour la banque**.
    """)

    st.markdown("---")
    st.subheader("🔎 Analyse du modèle")

    # Analyse automatique du comportement du modèle
    if FN > FP:
        st.error("⚠️ Trop de Faux Négatifs : le modèle laisse passer des clients risqués.")
        st.markdown("""
        **Impact :** pertes financières plus importantes.  
        **Solutions proposées :**  
        - Augmenter le recall (réglage du seuil > 0.5)  
        - Utiliser un modèle plus sensible : XGBoost, LGBM  
        - Rebalancer les classes (SMOTE, class_weight='balanced')
        """)
    elif FP > FN:
        st.warning("⚠️ Le modèle génère beaucoup de Faux Positifs.")
        st.markdown("""
        **Impact :** refus injustifiés pour de bons clients.  
        **Solutions proposées :**  
        - Optimiser le seuil vers 0.45–0.40  
        - Utiliser un modèle avec meilleure calibration (Logistic Regression)  
        """)
    else:
        st.success("✅ Les FP et FN sont équilibrés : bon compromis général.")

    st.markdown("---")
    st.subheader("🎯 Conseils généraux pour améliorer le modèle")

    st.markdown("""
    - Ajuster le **seuil de classification** (ex: 0.35 → augmente le recall)  
    - Tester XGBoost/LightGBM pour meilleure séparation des classes  
    - Rééquilibrer les classes (SMOTE, undersampling, class weights)  
    - Optimiser les hyperparamètres (GridSearchCV, RandomSearchCV)  
    """)

#==============================================================================
#Ajustement du seuil
#==============================================================================

elif menu == "⚙️ Ajustement du seuil":

    if "models" not in st.session_state:
        st.error("Veuillez entraîner les modèles d'abord.")
        st.stop()

    st.header("⚙️ Ajustement dynamique du seuil de classification")

    model_choice = st.selectbox(
        "Choisir un modèle",
        list(st.session_state["models"].keys())
    )

    model = st.session_state["models"][model_choice]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    # Probabilités
    if hasattr(model, "predict_proba"):
        proba_test = model.predict_proba(X_test)[:, 1]
    else:
        st.error("Ce modèle ne fournit pas de probabilités.")
        st.stop()

    st.subheader("🔧 Ajuster le seuil")

    threshold = st.slider("Seuil de classification", 0.0, 1.0, 0.50, 0.01)

    preds_adjusted = (proba_test >= threshold).astype(int)

    # Matrice recalculée
    cm_adj = confusion_matrix(y_test, preds_adjusted)
    TN, FP, FN, TP = cm_adj.ravel()

    st.subheader("📊 Matrice de confusion recalculée")
    st.write(cm_adj)

    # Heatmap Plotly
    fig = px.imshow(
        cm_adj,
        labels=dict(x="Prédiction", y="Vérité", color="Nombre"),
        x=["Non Défaut", "Défaut"],
        y=["Non Défaut", "Défaut"],
        text_auto=True,
        color_continuous_scale="Inferno"
    )
    st.plotly_chart(fig)

    st.markdown("---")
    st.subheader("📘 Interprétation du modèle avec seuil personnalisé")

    st.markdown(f"""
    ### ✅ Vrais Négatifs (TN) : `{TN}`
    Clients correctement identifiés comme **non risqués**.

    ### ✅ Vrais Positifs (TP) : `{TP}`
    Clients risqués correctement détectés.

    ### ⚠️ Faux Positifs (FP) : `{FP}`
    **Bons clients** classés comme risqués.  
    → Augmentation du taux de refus injustifiés.

    ### ⚠️ Faux Négatifs (FN) : `{FN}`
    **Mauvais clients** classés comme non risqués.  
    → Danger : pertes financières potentielles.
    """)

    st.markdown("---")
    st.subheader("🧠 Analyse du compromis FP / FN")

    if FN > FP:
        st.error("⚠️ Trop de Faux Négatifs : le modèle laisse passer des clients risqués.")
        st.markdown("""
        👉 Solutions :  
        - Augmenter le seuil (par ex. > 0.50)  
        - Utiliser XGBoost / LightGBM plus sensibles  
        - Rebalancer les classes (class_weight='balanced')
        """)

    elif FP > FN:
        st.warning("⚠️ Trop de Faux Positifs : le modèle refuse trop de bons clients.")
        st.markdown("""
        👉 Solutions :  
        - Baisser légèrement le seuil (0.45 – 0.35)  
        - Modèles plus stables : Logistic Regression  
        - Calibration (Isotonic Regression, Platt Scaling)
        """)

    else:
        st.success("✅ Bon équilibre FP / FN. Excellent compromis.")

    st.markdown("---")
    st.subheader("📈 Courbe ROC avec visualisation du seuil")

    fpr, tpr, thr = roc_curve(y_test, proba_test)

    fig2 = px.area(
        x=fpr, y=tpr,
        title="Courbe ROC",
        labels=dict(x="Taux de Faux Positifs", y="Taux de Vrais Positifs"),
        width=700, height=500
    )
    fig2.add_shape(
        type='line',
        x0=0, x1=1, y0=0, y1=1,
        line=dict(color="gray", dash="dash")
    )

    st.plotly_chart(fig2)

    st.info("""
    ✅ Ajuster le seuil permet de mieux contrôler le compromis entre :
    - **FN (risque financier)**  
    - **FP (perte commerciale)**  
    """)
#================================================================
#Seuil optimal
#================================================================

elif menu == "🎯 Seuil optimal":

    if "models" not in st.session_state:
        st.error("Veuillez entraîner les modèles d'abord.")
        st.stop()

    st.header("🎯 Calcul automatique du meilleur seuil")

    model_choice = st.selectbox(
        "Choisir un modèle",
        list(st.session_state["models"].keys())
    )

    model = st.session_state["models"][model_choice]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    # Probabilités
    if not hasattr(model, "predict_proba"):
        st.error("❌ Ce modèle ne fournit pas de probabilités.")
        st.stop()

    proba_test = model.predict_proba(X_test)[:, 1]

    # Calcul ROC
    fpr, tpr, thresholds = roc_curve(y_test, proba_test)

    # ========================================================================================
    # 1. THRESHOLD OPTIMAL SELON YOUDEN J (TPR - FPR)
    # ========================================================================================
    J = tpr - fpr
    idx_best_J = np.argmax(J)
    best_thresh_J = thresholds[idx_best_J]

    # ========================================================================================
    # 2. THRESHOLD OPTIMAL POUR MAX F1-SCORE
    # ========================================================================================
    f1_scores = []
    for thr in thresholds:
        preds = (proba_test >= thr).astype(int)
        f1_scores.append(f1_score(y_test, preds))
    idx_best_F1 = np.argmax(f1_scores)
    best_thresh_F1 = thresholds[idx_best_F1]

    # ========================================================================================
    # 3. THRESHOLD MAXIMUM POUR LE RECALL
    # ========================================================================================
    recall_scores = []
    for thr in thresholds:
        preds = (proba_test >= thr).astype(int)
        recall_scores.append(recall_score(y_test, preds))
    idx_best_recall = np.argmax(recall_scores)
    best_thresh_recall = thresholds[idx_best_recall]

    # ========================================================================================
    # 4. THRESHOLD MAXIMUM POUR LA PRECISION
    # ========================================================================================
    precision_scores = []
    for thr in thresholds:
        preds = (proba_test >= thr).astype(int)
        precision_scores.append(precision_score(y_test, preds))
    idx_best_precision = np.argmax(precision_scores)
    best_thresh_precision = thresholds[idx_best_precision]

    # ========================================================================================
    # AFFICHAGE DES RESULTATS
    # ========================================================================================
    st.subheader("📌 Meilleurs seuils calculés automatiquement")

    results = pd.DataFrame({
        "Méthode": [
            "Indice de Youden J (TPR-FPR)",
            "F1-Score maximal",
            "Recall maximal",
            "Précision maximale"
        ],
        "Seuil optimal": [
            best_thresh_J,
            best_thresh_F1,
            best_thresh_recall,
            best_thresh_precision
        ]
    })

    st.dataframe(results.style.highlight_max(axis=0, color="lightgreen"))

    st.markdown("---")
    st.subheader("📈 Visualisation : Courbe ROC + meilleur seuil Youden J")

    fig = px.area(
        x=fpr, y=tpr,
        title=f"Courbe ROC – {model_choice}",
        labels=dict(x="Faux Positifs (FPR)", y="Vrais Positifs (TPR)"),
        width=700, height=500
    )
    fig.add_shape(
        type='line', x0=0, x1=1, y0=0, y1=1,
        line=dict(color="gray", dash="dash")
    )
    fig.add_annotation(
        x=fpr[idx_best_J],
        y=tpr[idx_best_J],
        text=f"Seuil optimal : {best_thresh_J:.3f}",
        showarrow=True,
        arrowhead=2
    )

    st.plotly_chart(fig)

    st.markdown("✅ **Le seuil optimal selon Youden J maximise la séparation entre classes.**")

# =============================================================================
# 📊 ANALYSE UNIVARIÉE
# =============================================================================
elif menu == "📊 Analyse univariée":

    st.header("📊 Analyse univariée")

    if "raw_data" not in st.session_state:
        st.error("Veuillez importer un dataset d'abord.")
        st.stop()

    df = st.session_state["raw_data"]

    cat_vars = df.select_dtypes(include=["object"]).columns.tolist()
    num_vars = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    tab_num, tab_cat = st.tabs(["📈 Numériques", "📊 Catégorielles"])

    # ---------------------------- NUMÉRIQUES ------------------------------
    with tab_num:
        st.subheader("📈 Variables numériques")

        var = st.selectbox(
            "Sélectionner :", 
            num_vars, 
            key="univar_num_select"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Histogramme")
            fig = px.histogram(df, x=var, nbins=40)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Boxplot")
            fig2 = px.box(df, y=var)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Statistiques descriptives")
        st.dataframe(df[var].describe())

    # ---------------------------- CATÉGORIELLES ---------------------------
    with tab_cat:
        st.subheader("📊 Variables catégorielles")

        var = st.selectbox(
            "Sélectionner :", 
            cat_vars, 
            key="univar_cat_select"
        )

        st.markdown("### Fréquences")
        st.dataframe(df[var].value_counts())

        st.markdown("### Barplot")
        fig3 = px.bar(df[var].value_counts())
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Pie chart")
        fig4 = px.pie(df, names=var)
        st.plotly_chart(fig4, use_container_width=True)


# =============================================================================
# 📊 ANALYSE BIVARIÉE — VERSION AVANCÉE
# =============================================================================
elif menu == "📊 Analyse bivariée":

    st.header("📊 Analyse bivariée — relation variable / loan_status")

    if "raw_data" not in st.session_state:
        st.error("Veuillez importer un dataset d'abord.")
        st.stop()

    df = st.session_state["raw_data"]
    target = "loan_status"

    cat_vars = df.select_dtypes(include=["object"]).columns.tolist()
    num_vars = df.select_dtypes(include=["int64","float64"]).columns.tolist()

    if target in cat_vars:
        cat_vars.remove(target)
    if target in num_vars:
        num_vars.remove(target)

    tab_num, tab_cat, tab_cross, tab_multi = st.tabs([
        "📈 Numérique vs cible",
        "📊 Catégoriel vs cible",
        "🔀 Analyse croisée",
        "🌐 Corrélations"
    ])


    # -------------------------------------------------------------------------
    # 📈 NUMÉRIQUES VS CIBLE — AVEC TESTS STATISTIQUES (ANOVA, TUKEY, KRUSKAL)
    # -------------------------------------------------------------------------
    with tab_num:

        st.subheader("📈 Variable numérique vs loan_status")

        var = st.selectbox(
            "Sélectionner une variable numérique :", 
            num_vars,
            key="bivar_num_vs_target"
        )

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.box(df, x=target, y=var, points="all")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.violin(df, x=target, y=var, box=True)
            st.plotly_chart(fig2, use_container_width=True)


        # ===============================================================
        # ✅ TESTS : ANOVA + TUKEY HSD + KRUSKAL
        # ===============================================================
        import scipy.stats as stats
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        st.markdown("## 🔬 Tests statistiques")

        groups = [df[df[target] == grp][var].dropna() for grp in df[target].unique()]


        # ✅ ANOVA
        if len(groups) >= 2:
            f_stat, p_anova = stats.f_oneway(*groups)

            st.markdown("### ✅ ANOVA")
            st.write(f"F = {f_stat:.4f}, p = {p_anova:.6f}")

            if p_anova < 0.05:
                st.success("✅ Différences significatives entre groupes (ANOVA).")
            else:
                st.info("ℹ️ Pas de différence significative (ANOVA).")


            # ✅ TUKEY HSD si ANOVA significatif
            if p_anova < 0.05:
                st.markdown("### ✅ Test post‑hoc : Tukey HSD")

                tukey = pairwise_tukeyhsd(
                    endog=df[var].astype(float),
                    groups=df[target].astype(str),
                    alpha=0.05
                )

                tukey_df = pd.DataFrame(
                    data=tukey._results_table.data[1:],  
                    columns=tukey._results_table.data[0]
                )

                st.dataframe(tukey_df)


        # ✅ KRUSKAL-WALLIS
        stat_kruskal, p_kruskal = stats.kruskal(*groups)

        st.markdown("### ✅ Test de Kruskal‑Wallis")
        st.write(f"Kruskal H = {stat_kruskal:.4f}, p = {p_kruskal:.6f}")

        if p_kruskal < 0.05:
            st.success("✅ Différence significative (Kruskal‑Wallis).")
        else:
            st.info("ℹ️ Pas de différence significative (Kruskal‑Wallis).")



    # -------------------------------------------------------------------------
    # 📊 CATÉGORIELLES VS CIBLE
    # -------------------------------------------------------------------------
    with tab_cat:

        st.subheader("📊 Variables catégorielles vs loan_status")

        var = st.selectbox(
            "Choisir une variable catégorielle :", 
            cat_vars,
            key="bivar_cat_vs_target"
        )

        cont = pd.crosstab(df[var], df[target], normalize="index") * 100
        st.dataframe(cont.round(2))

        fig3 = px.bar(cont, barmode="stack")
        st.plotly_chart(fig3, use_container_width=True)

        from scipy.stats import chi2_contingency
        chi2, p, _, _ = chi2_contingency(pd.crosstab(df[var], df[target]))

        st.markdown("### ✅ Test du Chi‑2")
        st.write(f"Chi² = {chi2:.3f}, p = {p:.4f}")

        if p < 0.05:
            st.success("✅ Relation significative")
        else:
            st.info("ℹ️ Aucun lien significatif")



    # -------------------------------------------------------------------------
    # 🔀 ANALYSE CROISÉE
    # -------------------------------------------------------------------------
    with tab_cross:

        st.subheader("🔀 Analyse croisée Quanti × Quali")

        var_cat = st.selectbox(
            "Choisir une variable catégorielle :", 
            cat_vars,
            key="bivar_cross_cat"
        )

        var_num = st.selectbox(
            "Choisir une variable numérique :", 
            num_vars,
            key="bivar_cross_num"
        )

        fig4 = px.box(df, x=var_cat, y=var_num, points="all")
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### 🔹 Moyennes par catégorie")
        st.dataframe(df.groupby(var_cat)[var_num].mean().round(2))



    # -------------------------------------------------------------------------
    # 🌐 CORRÉLATIONS
    # -------------------------------------------------------------------------
    with tab_multi:

        st.subheader("🌐 Heatmap des corrélations")

        num_df = df.select_dtypes(include=["int64","float64"])
        corr = num_df.corr()

        fig5 = px.imshow(corr, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig5, use_container_width=True)

        if target in corr.columns:
            st.markdown("### 🔹 Corrélations avec la cible")
            st.dataframe(corr[target].sort_values(ascending=False))



# =============================================================================
# 🌐 ANALYSE MULTIVARIÉE AVANCÉE — PCA & KMeans
# =============================================================================
elif menu == "🌐 Analyse multivariée avancée":

    st.header("🌐 Analyse multivariée avancée — PCA & Clustering")

    if "raw_data" not in st.session_state:
        st.error("Veuillez importer un dataset d'abord.")
        st.stop()

    df = st.session_state["raw_data"]

    num_df = df.select_dtypes(include=["int64","float64"]).dropna()

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_df)

    tab_pca, tab_cluster = st.tabs(["🧮 PCA", "🎯 Clustering"])

    # ---------------------------- PCA ----------------------------
    with tab_pca:

        st.subheader("🧮 Analyse en Composantes Principales")

        pca = PCA(n_components=3)
        components = pca.fit_transform(X_scaled)

        st.markdown("### Variance expliquée (%)")
        st.dataframe(pd.DataFrame({
            "PC": ["PC1","PC2","PC3"],
            "Var %": pca.explained_variance_ratio_*100
        }))

        st.markdown("### Projection 2D")
        fig2d = px.scatter(
            x=components[:,0], 
            y=components[:,1],
            title="PCA 2D (PC1 vs PC2)"
        )
        st.plotly_chart(fig2d, use_container_width=True)

        st.markdown("### Projection 3D")
        fig3d = px.scatter_3d(
            x=components[:,0], 
            y=components[:,1], 
            z=components[:,2],
            title="PCA 3D"
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # ---------------------------- CLUSTERING ----------------------------
    with tab_cluster:

        st.subheader("🎯 Clustering KMeans")

        nb_clusters = st.slider("Nombre de clusters :", 2, 10, 3)

        kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        fig_cluster = px.scatter(
            x=components[:,0], 
            y=components[:,1],
            color=labels.astype(str),
            title="Clusters (PC1-PC2)"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        df_clust = pd.DataFrame(X_scaled, columns=num_df.columns)
        df_clust["Cluster"] = labels

        st.markdown("### Profil moyen des clusters")
        st.dataframe(df_clust.groupby("Cluster").mean().round(3))

# =============================================================================
# 5. SHAP - Version simplifiée (3 onglets) — Compatible Streamlit
# =============================================================================
elif menu == "🧠 Explicabilité (SHAP)":

    import shap
    import matplotlib.pyplot as plt
    shap.initjs()

    if "models" not in st.session_state:
        st.error("Veuillez entraîner les modèles d'abord.")
        st.stop()

    st.header("🧠 Explicabilité des modèles — Analyse SHAP")

    # Sélection du modèle
    model_choice = st.selectbox(
        "Choisir un modèle",
        list(st.session_state["models"].keys())
    )
    model = st.session_state["models"][model_choice]

    # Données train & test
    X_train = st.session_state["X_train"].astype("float64")
    X_test = st.session_state["X_test"].astype("float64")

    # ---------------------------------------------------------
    # ⚙️ Sélection automatique de l'explainer
    # ---------------------------------------------------------
    try:
        if model_choice in ["Random Forest", "XGBoost", "LightGBM"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            if isinstance(shap_values, list):  # XGB/LGBM renvoient une liste
                shap_values = shap_values[1]

        elif model_choice == "Régression Logistique":
            explainer = shap.LinearExplainer(
                model,
                X_train,
                feature_perturbation="interventional"
            )
            shap_values = explainer.shap_values(X_train)

        else:
            st.error("❌ SHAP ne supporte pas ce modèle (SVM, KNN).")
            st.stop()

    except Exception as e:
        st.error(f"Erreur SHAP : {e}")
        st.stop()

    # ======================================================
    # ✅ Onglets : GLOBAL — LOCAL — DECISION
    # ======================================================
    tab_global, tab_local, tab_decision = st.tabs([
        "🌍 Importance globale",
        "🎯 Explication individuelle",
        "🧭 Decision Plot"
    ])

    # ======================================================
    # 🌍 1. IMPORTANCE GLOBALE
    # ======================================================
    with tab_global:
        st.subheader("📊 Importance globale des caractéristiques")

        # Summary bar plot
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(fig)

        # Summary scatter plot
        fig2 = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, show=False)
        st.pyplot(fig2)

    # ======================================================
    # 🎯 2. EXPLICATION INDIVIDUELLE (SANS FORCE PLOT)
    # ======================================================
    with tab_local:

        st.subheader("🎯 Explication locale")

        index = st.slider("Sélectionner l'observation :", 0, len(X_test) - 1, 0)
        instance = X_test.iloc[index:index+1]

        st.write("🔎 Observation sélectionnée :")
        st.dataframe(instance)

        # Probabilité prédite
        proba = model.predict_proba(instance)[0][1]
        st.metric("Probabilité prédite", f"{proba*100:.2f}%")

        # 💧 WATERFALL PLOT
        st.markdown("### 💧 Waterfall Plot (local)")

        try:
            fig_w = plt.figure(figsize=(10, 6))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values[index],
                feature_names=instance.columns
            )
            st.pyplot(fig_w)

        except Exception as e:
            st.warning("Waterfall non supporté pour ce modèle.")
            st.code(str(e))

    # ======================================================
    # 🧭 3. DECISION PLOT
    # ======================================================
    with tab_decision:

        st.subheader("🧭 Decision Plot")

        try:
            fig_d = plt.figure(figsize=(12, 6))
            shap.decision_plot(
                explainer.expected_value,
                shap_values[:100],   # Limité pour lisibilité
                X_train.columns,
                show=False
            )
            st.pyplot(fig_d)

        except Exception as e:
            st.warning("Decision plot non supporté.")
            st.code(str(e))
