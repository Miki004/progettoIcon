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

from sklearn.exceptions import UndefinedMetricWarning
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from itertools import cycle

def plot_regression_metrics(model, X_test, y_test):
    """Plotta le metriche MSE, MAE, R² come grafico a barre."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = ["MSE", "MAE", "R²"]
    values = [mse, mae, r2]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, values, color=["blue", "red", "green"])
    plt.ylabel("Valore")
    plt.title("Metriche di Regressione")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotazioni sulle barre
    for i, v in enumerate(values):
        plt.text(i, v + 0.01 * max(values), f"{v:.3f}", ha="center", fontsize=12)

    plt.show()

def plot_predicted_vs_actual(model, X_test, y_test):
    """Plotta i valori predetti vs i valori reali."""
    y_pred = model.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue", label="Predetto vs Reale")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="dashed", linewidth=2, label="Perfetta Predizione")
    
    plt.xlabel("Valori Reali")
    plt.ylabel("Valori Predetti")
    plt.title("Valori Predetti vs Reali")
    plt.legend()
    plt.grid()
    plt.show()

def plot_multiclass_roc(model, X_test, y_test, n_classes):
    """Plotta la curva ROC per un problema di classificazione multi-classe."""
    # Controlla se il modello ha il metodo predict_proba()
    if not hasattr(model, "predict_proba"):
        print("Il modello non supporta predict_proba(), impossibile disegnare la curva ROC.")
        return
    
    y_score = model.predict_proba(X_test)  # Probabilità previste per ogni classe
    
    # Binarizziamo le etichette per il calcolo della curva ROC
    from sklearn.preprocessing import label_binarize
    y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))

    # Calcoliamo ROC e AUC per ogni classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plottiamo le curve ROC per ogni classe
    plt.figure(figsize=(10, 6))
    colors = cycle(["blue", "red", "green", "purple", "orange", "cyan"])  # Colori per le curve
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,label=f"Classe {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)  # Linea diagonale (random classifier)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC multi-classe")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def plot_classification_metrics(model, X_test, y_test, class_labels):
    """Plotta un grafico a barre con Precision, Recall, F1-score per ogni classe."""
    y_pred = model.predict(X_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    x = np.arange(len(class_labels))
    width = 0.25  # Larghezza delle barre

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width=width, label="Precision", color="blue")
    plt.bar(x, recall, width=width, label="Recall", color="red")
    plt.bar(x + width, f1, width=width, label="F1-score", color="green")

    plt.xticks(x, class_labels, rotation=45)
    plt.xlabel("Classi")
    plt.ylabel("Valore")
    plt.title("Metriche di Classificazione per Classe")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def evaluate_classification_model(model, X_test, y_test):
    """Calcola e stampa le metriche di classificazione."""
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
    
    # Verifica se il modello supporta predict_proba() o decision_function()
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            print("ROC-AUC (OvR):", roc_auc)
        except ValueError as e:
            print(f"Errore nel calcolo ROC-AUC: {e}")
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        try:
            roc_auc = roc_auc_score(y_test, y_scores, multi_class='ovr')
            print("ROC-AUC (OvR):", roc_auc)
        except ValueError as e:
            print(f"Errore nel calcolo ROC-AUC: {e}")
    else:
        print("ROC-AUC non disponibile: il modello non supporta predict_proba() o decision_function().")


def evaluate_regression_model(model, X_test, y_test):
    """Calcola e stampa le metriche di regressione."""
    y_pred = model.predict(X_test)
    
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2-score:", r2_score(y_test, y_pred))

def plot_learning_curves(model, X_train, y_train, title, scoring):
    """Genera e visualizza le curve di apprendimento."""
    plt.figure(figsize=(12, 6))
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_var = np.var(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_var = np.var(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, label=f'{model.__class__.__name__} - Training', linestyle='--')
    plt.fill_between(train_sizes, train_mean - train_var, train_mean + train_var, alpha=0.1)
    plt.plot(train_sizes, test_mean, label=f'{model.__class__.__name__} - Validation')
    plt.fill_between(train_sizes, test_mean - test_var, test_mean + test_var, alpha=0.1)

    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel(scoring.capitalize())
    plt.legend()
    plt.grid()
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

# Valutazione del miglior modello di classificazione
evaluate_classification_model(best_classification_model, X_test_scaled, y_test_cls)
plot_classification_metrics(best_classification_model, X_test_scaled, y_test_cls, class_labels)
plot_multiclass_roc(best_classification_model, X_test_scaled, y_test_cls, n_classes)
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
evaluate_regression_model(best_regression_model, X_test_scaled, y_test_reg)
plot_predicted_vs_actual(best_regression_model, X_test_scaled, y_test_reg)
plot_regression_metrics(best_regression_model, X_test_scaled, y_test_reg)

joblib.dump(best_regression_model, "best_regression_model.pkl")

print("\n--- Risultati finali ---")  
print(f"Classificazione - Migliore accuracy (massimo cross-validation): {best_score_cls:.4f}")
print(f"Regressione - Migliore MSE (minimo cross-validation): {-best_score_reg:.4f}")
