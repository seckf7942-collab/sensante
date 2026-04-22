import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ============================================================
# ETAPE 2 : Charger et préparer les données
# ============================================================
df = pd.read_csv("data/patients_dakar.csv")
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded']   = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'frissons', 'nausee', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible    : {y.shape}")

# ============================================================
# ETAPE 3 : Séparer train / test
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"\nEntrainement : {X_train.shape[0]} patients")
print(f"Test         : {X_test.shape[0]} patients")

# ============================================================
# ETAPE 4 : Entraîner le modèle
# ============================================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"\nModele entraine !")
print(f"Nombre d'arbres  : {model.n_estimators}")
print(f"Nombre features  : {model.n_features_in_}")
print(f"Classes          : {list(model.classes_)}")

# ============================================================
# ETAPE 5 : Evaluer le modèle
# ============================================================
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy:.2%}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)

# Visualisation
os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
print("Figure sauvegardee : figures/confusion_matrix.png")

# ============================================================
# ETAPE 6 : Sérialiser le modèle
# ============================================================
os.makedirs("models", exist_ok=True)

joblib.dump(model,        "models/model.pkl")
joblib.dump(le_sexe,      "models/encoder_sexe.pkl")
joblib.dump(le_region,    "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

size = os.path.getsize("models/model.pkl")
print(f"\nModele sauvegarde : models/model.pkl ({size/1024:.1f} Ko)")
print("Encodeurs et metadata sauvegardes.")

# ============================================================
# ETAPE 7 : Tester le modèle rechargé
# ============================================================
model_loaded     = joblib.load("models/model.pkl")
le_sexe_loaded   = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"\nModele recharge : {type(model_loaded).__name__}")

nouveau_patient = {
    'age': 28, 'sexe': 'F', 'temperature': 39.5,
    'tension_sys': 110, 'toux': True, 'fatigue': True,
    'maux_tete': True, 'frissons': True, 'nausee': False,
    'region': 'Dakar'
}

sexe_enc   = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# On utilise un DataFrame pour eviter le warning feature names
features_df = pd.DataFrame([[
    nouveau_patient['age'], sexe_enc,
    nouveau_patient['temperature'], nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']), int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']), int(nouveau_patient['frissons']),
    int(nouveau_patient['nausee']), region_enc
]], columns=feature_cols)

diagnostic = model_loaded.predict(features_df)[0]
probas     = model_loaded.predict_proba(features_df)[0]
proba_max  = probas.max()

print(f"\n--- Resultat du pre-diagnostic ---")
print(f"Patient    : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite: {proba_max:.1%}")
print(f"\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:12s} : {proba:.1%} {bar}")