import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV,  RepeatedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score,make_scorer)
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from itertools import cycle

def plot_learning_curves(model, X_train, y_train, title, scoring, cv=5):
    """Genera e visualizza le curve di apprendimento con metriche aggiuntive."""
    plt.figure(figsize=(12, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring=scoring,
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 6)
    )
    
    # Identifica se la metrica usata è da classificazione o regressione
    is_classification = scoring in ["accuracy", "precision", "recall", "f1"]
    
    if is_classification:
        # Per classificazione -> convertiamo accuracy in errore (1 - accuracy), ecc.
        train_mean = 1 - np.mean(train_scores, axis=1)
        test_mean = 1 - np.mean(test_scores, axis=1)
        ylabel = "Error"
    else:
        # Per regressione -> plottiamo direttamente la metrica (es. MSE)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        ylabel = scoring.upper()
    
    plt.plot(train_sizes, train_mean, label='Train', color='green')
    plt.plot(train_sizes, test_mean, label='Test', color='red')
    
    plt.title(f'Learning Curve - {title}')
    plt.xlabel("Training Set Size")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
    
    # Metriche aggiuntive di classificazione
    if is_classification:
        metrics = {
            "Precision": make_scorer(precision_score, average="weighted"),
            "Recall": make_scorer(recall_score, average="weighted"),
            "F1-score": make_scorer(f1_score, average="weighted"),
        }
        metric_values = {}
        for metric_name, metric_func in metrics.items():
            score = np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring=metric_func))
            metric_values[metric_name] = score

        plt.figure(figsize=(10, 5))
        plt.bar(metric_values.keys(), metric_values.values(), color=["blue", "orange", "purple"])
        plt.title("Metriche di Classificazione (CV)")
        plt.ylabel("Score")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
    else:
        # Metriche aggiuntive di regressione
        metrics = {
            "MSE": make_scorer(mean_squared_error, greater_is_better=False),
            "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
            "R²": make_scorer(r2_score)
        }
        metric_values = {}
        
        for metric_name, metric_func in metrics.items():
            score = np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring=metric_func))
            # Se la metrica è neg_mean_squared_error, la usiamo in valore assoluto
            if "neg_" in metric_func._score_func.__name__:
                score = abs(score)
            metric_values[metric_name] = score
        
        plt.figure(figsize=(10, 5))
        plt.bar(metric_values.keys(), metric_values.values(), color=["blue", "red", "green"])
        plt.title("Metriche di Regressione (CV)")
        plt.ylabel("Valore")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

# Definizione della validazione incrociata con più suddivisioni
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# Caricamento dataset
data = pd.read_csv("Obesity prediction.csv")

# Gestione dei valori mancanti
for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype == "object":
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].mean(), inplace=True)

# Codifica delle classi per la classificazione
label_encoder = LabelEncoder()
data['Obesity'] = label_encoder.fit_transform(data['Obesity'])
joblib.dump(label_encoder, "label_encoder.pkl")

y_classification = data['Obesity']
y_regression = data['Weight'].values.reshape(-1, 1)
X = data.drop(columns=['Obesity', 'Weight'])

# Identificazione delle feature categoriche e numeriche
categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

# Pipeline di preprocessing
ohe = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', num_imputer), ('scaler', scaler)]), numerical_features),
    ('cat', Pipeline([('imputer', cat_imputer), ('encoder', ohe)]), categorical_features)
])

# Suddivisione train-test
X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

n_classes = len(np.unique(y_test_cls))
class_labels = label_encoder.classes_
# Applichiamo il preprocessor
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
joblib.dump(preprocessor, "preprocessor.pkl")

# Assicuriamoci che non ci siano valori NaN residui
X_train_scaled = np.nan_to_num(X_train_scaled)
X_test_scaled = np.nan_to_num(X_test_scaled)

# Modelli per classificazione e regressione
classification_models = {
    'RandomForest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 150],
        'max_depth': [4, 5],
        'min_samples_split': [15],
        'min_samples_leaf': [10]
    }),
    'LogisticRegression': (LogisticRegression(solver='newton-cg', max_iter=500, random_state=42), {
        'C': [0.1, 1, 10],  # Regolarizzazione L2
        'penalty': ['l2']
    }) 

}

regression_models = {
    'RandomForest': (RandomForestRegressor(random_state=42), {
        'n_estimators': [100,200],
        'max_depth': [5,7],
        'min_samples_split': [9,10],
        'min_samples_leaf': [9,10]
    }),
    'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    })
}

# GridSearch per la classificazione
best_classification_model, best_score_cls = None, 0
for model_name, (model, params) in classification_models.items():
    grid_search = GridSearchCV(model, params, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_cls)
    best_model = grid_search.best_estimator_
    scores_cls = cross_val_score(best_model, X_train_scaled, y_train_cls, cv=cv, scoring='accuracy')
    
    print(f"\n[Classificazione] Modello: {model_name}")
    print("Punteggi cross-validation:", scores_cls)
    print(f"Deviazione Standard: {np.std(scores_cls):.4f}, Varianza: {np.var(scores_cls):.4f}")
    
    if max(scores_cls) > best_score_cls:
        best_classification_model, best_score_cls = best_model, max(scores_cls)

plot_learning_curves(best_classification_model,X_train_scaled,y_train_cls,title="Miglior Modello di Classificazione",scoring="accuracy",cv=cv)
joblib.dump(best_classification_model, "best_classification_model.pkl")

# GridSearch per la regressione
best_regression_model, best_score_reg = None, float('inf')  # Corretto per minimizzare MSE
for model_name, (model, params) in regression_models.items():
    grid_search = GridSearchCV(model, params, cv=cv, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_reg.ravel())
    
    best_model = grid_search.best_estimator_  # Ora è definito correttamente
    mse_scores = -cross_val_score(best_model, X_train_scaled, y_train_reg.ravel(), cv=cv, scoring='neg_mean_squared_error')
    
    print(f"\n[Regressione] Modello: {model_name}")
    print("Punteggi MSE (cross-validation):", mse_scores)
    print(f"Deviazione Standard MSE: {np.std(mse_scores):.4f}, Varianza MSE: {np.var(mse_scores):.4f}")
    if min(mse_scores) < best_score_reg:
        best_regression_model, best_score_reg = best_model, min(mse_scores)

# Valutazione del miglior modello di regressione
plot_learning_curves(best_regression_model,X_train_scaled, y_train_reg.ravel(),title="Miglior Modello di Regressione",scoring="neg_mean_squared_error",cv=cv)

joblib.dump(best_regression_model, "best_regression_model.pkl")

print("\n--- Risultati finali ---")  
print(f"Classificazione - Migliore accuracy (massimo cross-validation): {best_score_cls:.4f}")
print(f"Regressione - Migliore MSE (minimo cross-validation): {-best_score_reg:.4f}")
