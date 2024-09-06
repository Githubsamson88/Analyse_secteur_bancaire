import warnings
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from catboost import CatBoostClassifier, Pool
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
data_m = pd.read_csv(StringIO(response.text))
    
# Encodage des variables booléennes
bool_columns = data_m.select_dtypes(include=['bool']).columns
data_m[bool_columns] = data_m[bool_columns].astype(int)

# Suppression des pourcentages des colonnes int_rate et revol_util
if data_m['int_rate'].dtype == 'object':
    data_m['int_rate'] = data_m['int_rate'].str.replace('%', '')
    data_m['int_rate'] = pd.to_numeric(data_m['int_rate'], errors='coerce')

if data_m['revol_util'].dtype == 'object':
    data_m['revol_util'] = data_m['revol_util'].str.replace('%', '')
    data_m['revol_util'] = pd.to_numeric(data_m['revol_util'], errors='coerce')

# Conversion des colonnes catégorielles en type 'category'
categorical_columns = data_m.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data_m[col] = data_m[col].astype('category')

# Suppression des valeurs manquantes
data_m = data_m.dropna()

# Encodage de loan_status
label_encoder = LabelEncoder()
data_m['loan_status'] = label_encoder.fit_transform(data_m['loan_status'])

# Définition des caractéristiques et de la cible
X = data_m.drop('loan_status', axis=1)
y = data_m['loan_status']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des colonnes numériques
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['float64', 'int32', 'float32']).columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Définition des indices des colonnes catégorielles
categorical_features_indices = [X.columns.get_loc(col) for col in X.select_dtypes(include=['category']).columns]

# Création du modèle CatBoost
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, cat_features=categorical_features_indices, verbose=0)

# Entraînement du modèle
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100, plot=True)

# Prédiction
y_probs = model.predict_proba(X_test)[:, 1]

# Évaluation des performances du modèle
auc = roc_auc_score(y_test, y_probs)
st.write(f"Score AUC : {auc:.4f}")


# Calcul des valeurs SHAP pour le modèle CatBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap_values_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X_train)

# Affichage du plot SHAP Summary
st.write("## SHAP Summary Plot")
fig_summary, ax_summary = plt.subplots()
shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)
st.pyplot(fig_summary)

# Nettoyage de la figure
plt.close(fig_summary)

# Affichage du plot SHAP Waterfall
st.write("## SHAP Waterfall Plot")
shap.waterfall_plot(shap_values_exp[0])

st.pyplot()



# Tracer la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
st.pyplot(bbox_inches='tight')
plt.clf()