import warnings
import matplotlib
matplotlib.use('Agg')  # Configuration du backend non interactif pour Matplotlib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, jaccard_score, hamming_loss, classification_report, auc
from ngboost import NGBClassifier
import shap
import requests
from io import StringIO
import matplotlib.pyplot as plt

# Ignorer les avertissements PyplotGlobalUseWarning
warnings.filterwarnings("ignore", category=UserWarning, message="Matplotlib is currently using agg")

# URL du fichier CSV sur GitHub
github_url = 'https://raw.githubusercontent.com/Githubsamson88/Credit-analysis/main/data_M.csv'

# Téléchargement du fichier CSV depuis GitHub
response = requests.get(github_url)

# Lecture du contenu du fichier CSV
data = pd.read_csv(StringIO(response.text))

# Suppression de la colonne 'last_pymnt_d'
#data.drop('last_pymnt_d', axis=1, inplace=True)

# Affichage des premières lignes et des statistiques descriptives
st.write("Aperçu des données :")
st.write(data.head(5))
st.write("Statistiques descriptives :")
st.write(data.describe())

# Encodage des variables booléennes
bool_columns = data.select_dtypes(include=['bool']).columns
data[bool_columns] = data[bool_columns].astype(int)

# Suppression des pourcentages des colonnes int_rate et revol_util
if data['int_rate'].dtype == 'object':
    data['int_rate'] = data['int_rate'].str.replace('%', '')
    data['int_rate'] = pd.to_numeric(data['int_rate'], errors='coerce')

if data['revol_util'].dtype == 'object':
    data['revol_util'] = data['revol_util'].str.replace('%', '')
    data['revol_util'] = pd.to_numeric(data['revol_util'], errors='coerce')

# Conversion des colonnes catégorielles en type 'category'
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = data[col].astype('category')

# Suppression des valeurs manquantes
data = data.dropna()

# Visualisation
# Visualisation 1 : data['annual_inc'].plot()
st.write("Visualisation 1 : Revenu annuel")
st.line_chart(data['annual_inc'])

# Visualisation 2 : Répartition des modalités de loan_status
st.write("Visualisation 2 : Répartition des modalités de loan_status")
total_samples = data.shape[0]
loan_status_percentages = data['loan_status'].value_counts(normalize=True) * 100
plt.figure(figsize=(8, 6))
loan_status_percentages.plot(kind='bar', color=['blue', 'orange'])
plt.title('Répartition des modalités de loan_status')
plt.xlabel('loan_status')
plt.ylabel('Pourcentage')
plt.xticks(rotation=0)
for i, percentage in enumerate(loan_status_percentages):
    plt.text(i, percentage + 1, f'{percentage:.2f}%', ha='center')

plt.show()
st.pyplot()

# Encodage de loan_status
label_encoder = LabelEncoder()
data['loan_status'] = label_encoder.fit_transform(data['loan_status'])

# Définition des caractéristiques et de la cible
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des colonnes numériques
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['float64', 'int32', 'float32']).columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Création du modèle NGBoost
model_ngboost = NGBClassifier()

# Entraînement du modèle NGBoost
model_ngboost.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred_ngboost = model_ngboost.predict(X_test)
y_pred_proba_ngboost = model_ngboost.predict_proba(X_test)[:, 1]

# Calcul des métriques
accuracy_ngboost = accuracy_score(y_test, y_pred_ngboost)
jaccard_ngboost = jaccard_score(y_test, y_pred_ngboost, average='weighted')
hamming_ngboost = hamming_loss(y_test, y_pred_ngboost)

# Affichage des résultats
st.write("Résultats du modèle NGBoost :")
st.write("Accuracy :", accuracy_ngboost)
st.write("Jaccard Score :", jaccard_ngboost)
st.write("Hamming Loss :", hamming_ngboost)

# Évaluation du modèle
st.write("Évaluation du modèle :")
st.write(classification_report(y_test, y_pred_ngboost))
roc_auc = roc_auc_score(y_test, y_pred_proba_ngboost)
st.write(f'ROC AUC Score: {roc_auc}')

# Obtention des importances des caractéristiques
feature_importances = model_ngboost.feature_importances_
feature_importances = feature_importances.ravel()  # ou feature_importances.flatten()
features = X_train.columns
coefficients_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
intercept_df = pd.DataFrame({'Feature': ['Intercept'], 'Importance': [model_ngboost.score(X_train, y_train)]})
coefficients_df = pd.concat([coefficients_df, intercept_df], ignore_index=True)
st.write(coefficients_df)

# Visualisation avec SHAP
explainer = shap.TreeExplainer(model_ngboost)
shap_values = explainer.shap_values(X_test)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0, :])


###
# Visualisation avec SHAP
explainer = shap.TreeExplainer(model_ngboost)
shap_values = explainer.shap_values(X_test)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0, :])

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_ngboost)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
st.pyplot()

# Liste déroulante pour les codes postaux
st.write("Probabilités pour un client selon le code postal")
unique_zip_codes = data['zip_code'].unique()
selected_zip_code = st.selectbox('Sélectionnez un code postal', unique_zip_codes)

# Filtrer les données pour le code postal sélectionné
client_data = data[data['zip_code'] == selected_zip_code]

if not client_data.empty:
    # Normalisation des données du client sélectionné
    client_data[numerical_cols] = scaler.transform(client_data[numerical_cols])
    
    # Créer une liste d'identifiants de clients uniques
    client_ids = client_data.index.tolist()

    # Prédiction des probabilités pour le client sélectionné
    client_proba = model_ngboost.predict_proba(client_data.drop('loan_status', axis=1))[:, 1]

    # Créer une liste de dictionnaires contenant les résultats
    results = []
    for client_id, client_proba, client_data_row in zip(client_ids, client_proba, client_data.iterrows()):
        client_variables = {col: val for col, val in client_data_row[1].items()}
        client_variables.update({'Client ID': client_id, 'Probabilité de défaut de paiement': client_proba})
        results.append(client_variables)

    # Créer un DataFrame à partir des résultats
    results_df = pd.DataFrame(results)

    # Afficher le DataFrame
    st.write("Résultats pour les clients avec le code postal sélectionné :")
    st.write(results_df)

# Déploiement sur Streamlit Sharing
if __name__ == '__main__':
    st.write("Modèle NGBoost prêt à l'emploi !")
