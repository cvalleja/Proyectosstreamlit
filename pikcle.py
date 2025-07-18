import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

df = pd.read_csv('creditcard.csv')

# Preparar datos
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Configuración óptima del modelo (ajustar según necesidad)
best_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'class_weight': 'balanced_subsample',
    'random_state': 42,
    'n_jobs': -1
}

# Crear y entrenar pipeline con SMOTE
model = make_pipeline(
    StandardScaler(),
    SMOTE(random_state=42),
    RandomForestClassifier(**best_params)
)

print("Entrenando modelo...")
model.fit(X_train, y_train)

# Guardar modelo entrenado
with open('creditcard_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo entrenado y guardado correctamente en creditcard_model.pkl")