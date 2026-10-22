
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Churn Insight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "notebooks" / "models" / "churn_model.pkl"


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            padding: 18px;
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 16px;
            padding: 24px;
        }

        div[data-testid="stFormSubmitButton"] button {
            border-radius: 10px;
            min-height: 48px;
            font-weight: 700;
        }

        div[data-testid="stDownloadButton"] button {
            border-radius: 10px;
            min-height: 44px;
            font-weight: 600;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def charger_modele():
    artefacts = joblib.load(MODEL_PATH)

    return (
        artefacts["model"],
        artefacts["preprocessor"],
        artefacts["threshold"]
    )


if not MODEL_PATH.exists():
    st.error(f"Le fichier du modèle est introuvable : {MODEL_PATH}")
    st.stop()

try:
    model, preprocessor, seuil = charger_modele()

except Exception as erreur:
    st.error(f"Erreur lors du chargement du modèle : {erreur}")
    st.stop()


if "historique_predictions" not in st.session_state:
    st.session_state.historique_predictions = []

if "derniere_soumission" not in st.session_state:
    st.session_state.derniere_soumission = None


with st.sidebar:
    st.title("📊 Churn Insight AI")
    st.caption("Customer Retention Analytics")

    st.divider()

    page = st.radio(
        "MENU PRINCIPAL",
        ["Prédiction", "Historique"]
    )

    st.divider()

    st.markdown("### Configuration")
    st.write("Modèle de Machine Learning")

    st.metric(
        "Seuil de décision",
        f"{seuil:.0%}"
    )

    st.caption(
        "Les prédictions sont des estimations "
        "et ne garantissent pas le comportement du client."
    )


if page == "Prédiction":

    st.title("Analyse du risque client")

    st.markdown(
        "Estimez le risque de résiliation d'un client "
        "à partir de ses informations et services souscrits."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Modèle", "Churn ML")

    with col2:
        st.metric("Seuil", f"{seuil:.0%}")

    with col3:
        st.metric(
            "Analyses réalisées",
            len(st.session_state.historique_predictions)
        )

    with st.form("formulaire_client"):

        st.subheader("1. Informations du client")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Genre",
                ["Female", "Male"]
            )

            SeniorCitizen = st.selectbox(
                "Client senior",
                [0, 1],
                format_func=lambda x: "Oui" if x == 1 else "Non"
            )

            Partner = st.selectbox(
                "En couple",
                ["Yes", "No"],
                format_func=lambda x: "Oui" if x == "Yes" else "Non"
            )

        with col2:
            Dependents = st.selectbox(
                "Personnes à charge",
                ["Yes", "No"],
                format_func=lambda x: "Oui" if x == "Yes" else "Non"
            )

            tenure = st.slider(
                "Ancienneté du client",
                0,
                72,
                12,
                format="%d mois"
            )

        st.divider()

        st.subheader("2. Services souscrits")

        col1, col2 = st.columns(2)

        with col1:
            PhoneService = st.selectbox(
                "Téléphone",
                ["Yes", "No"]
            )

            MultipleLines = st.selectbox(
                "Lignes multiples",
                ["No", "Yes", "No phone service"]
            )

            InternetService = st.selectbox(
                "Service Internet",
                ["DSL", "Fiber optic", "No"]
            )

            OnlineSecurity = st.selectbox(
                "Sécurité en ligne",
                ["Yes", "No", "No internet service"]
            )

            OnlineBackup = st.selectbox(
                "Sauvegarde en ligne",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            DeviceProtection = st.selectbox(
                "Protection des appareils",
                ["Yes", "No", "No internet service"]
            )

            TechSupport = st.selectbox(
                "Support technique",
                ["Yes", "No", "No internet service"]
            )

            StreamingTV = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            StreamingMovies = st.selectbox(
                "Streaming Films",
                ["Yes", "No", "No internet service"]
            )

        st.divider()

        st.subheader("3. Contrat et facturation")

        col1, col2 = st.columns(2)

        with col1:
            Contract = st.selectbox(
                "Type de contrat",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ],
                format_func=lambda x: {
                    "Month-to-month": "Mensuel",
                    "One year": "1 an",
                    "Two year": "2 ans"
                }[x]
            )

            PaperlessBilling = st.selectbox(
                "Facturation électronique",
                ["Yes", "No"]
            )

            PaymentMethod = st.selectbox(
                "Mode de paiement",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col2:
            MonthlyCharges = st.number_input(
                "Montant mensuel (€)",
                min_value=0.0,
                value=70.0,
                step=5.0
            )

            TotalCharges = st.number_input(
                "Montant total facturé (€)",
                min_value=0.0,
                value=840.0,
                step=50.0
            )

        submitted = st.form_submit_button(
            "Analyser le risque de résiliation",
            use_container_width=True
        )

    if submitted:

        client = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])

        try:
            X_client = preprocessor.transform(client)

            probabilite = float(
                model.predict_proba(X_client)[0][1]
            )

            prediction = int(probabilite >= seuil)

        except Exception as erreur:
            st.error(f"Erreur lors de la prédiction : {erreur}")
            st.stop()

        signature = tuple(client.iloc[0].tolist())

        if signature != st.session_state.derniere_soumission:

            resultat = {
                "Ancienneté (mois)": tenure,
                "Contrat": Contract,
                "Montant mensuel (€)": MonthlyCharges,
                "Montant total (€)": TotalCharges,
                "Probabilité de résiliation": round(probabilite, 4),
                "Prédiction": (
                    "Résiliation probable"
                    if prediction == 1
                    else "Client susceptible de rester"
                )
            }

            st.session_state.historique_predictions.append(resultat)
            st.session_state.derniere_soumission = signature

        st.divider()
        st.subheader("Résultat de l'analyse")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Probabilité de résiliation",
                f"{probabilite:.1%}"
            )

            st.progress(probabilite)
            st.caption(f"Seuil de décision : {seuil:.0%}")

        with col2:
            if prediction == 1:
                st.error("Risque de résiliation détecté")

                st.write(
                    "La probabilité estimée atteint ou dépasse "
                    "le seuil de décision."
                )

            else:
                st.success("Client susceptible de rester")

                st.write(
                    "La probabilité estimée est inférieure "
                    "au seuil de décision."
                )

        st.subheader("Résumé du client")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Ancienneté", f"{tenure} mois")

        with col2:
            st.metric(
                "Contrat",
                {
                    "Month-to-month": "Mensuel",
                    "One year": "1 an",
                    "Two year": "2 ans"
                }[Contract]
            )

        with col3:
            st.metric(
                "Montant mensuel",
                f"{MonthlyCharges:.2f} €"
            )

        st.info(
            "Cette prédiction est une estimation du modèle "
            "et ne garantit pas le comportement futur du client."
        )


elif page == "Historique":

    st.title("Historique des analyses")

    st.markdown(
        "Consultez les profils analysés et les résultats "
        "des prédictions réalisées pendant cette session."
    )

    historique = st.session_state.historique_predictions

    if historique:

        df_historique = pd.DataFrame(historique)

        nb_predictions = len(df_historique)

        nb_resiliations = (
            df_historique["Prédiction"] == "Résiliation probable"
        ).sum()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total des analyses", nb_predictions)

        with col2:
            st.metric(
                "Résiliations prédites",
                int(nb_resiliations)
            )

        with col3:
            st.metric(
                "Clients susceptibles de rester",
                nb_predictions - int(nb_resiliations)
            )

        st.subheader("Détail des prédictions")

        st.dataframe(
            df_historique,
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            csv = df_historique.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "Télécharger l'historique CSV",
                data=csv,
                file_name="historique_churn.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            if st.button(
                "Effacer l'historique",
                use_container_width=True
            ):
                st.session_state.historique_predictions = []
                st.session_state.derniere_soumission = None
                st.rerun()

    else:
        st.info(
            "Aucune analyse enregistrée. "
            "Rendez-vous dans la page Prédiction pour commencer."
        )
