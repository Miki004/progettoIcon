import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sup
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV, KFold, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RepeatedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

def plot_learning_curves(model, X_train, y_train, title, scoring):
    """ Genera e visualizza le curve di apprendimento. """
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


# **1. Caricamento del dataset preprocessato**
data = pd.read_csv("preprocessed_obesity_data.csv")

# **2. Definizione delle feature e della variabile target**
X = data.drop(columns=['Obesity', 'Weight'])
y_classification = data['Obesity']
y_regression = data['Weight'].values.reshape(-1, 1) 

# **3. Divisione train-test**
X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# **4. Caricamento dello scaler salvato nel preprocessing**
scaler = joblib.load("scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_weight = joblib.load("scaler_weight.pkl")
y_train_reg_scaled = scaler_weight.transform(y_train_reg)
y_test_reg_scaled = scaler_weight.transform(y_test_reg)

# **5. Definizione della strategia di cross-validation**
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

# **6. Definizione dei modelli con GridSearchCV**
classification_models = {
    'RandomForestClassifier': (RandomForestClassifier(random_state=42), {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 6, 7],
        'min_samples_split': [9, 10],
        'min_samples_leaf': [9, 10],
        'bootstrap': [True]
    })
}

regression_models = {
    'RandomForestRegressor': (RandomForestRegressor(random_state=42), {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 6, 7],
        'min_samples_split': [9, 10],
        'min_samples_leaf': [9, 10],
        'bootstrap': [True]
    })
}

# **7. GridSearch per il modello di classificazione**
best_classification_model = None
best_score_cls = 0

for model_name, (model, params) in classification_models.items():
    grid_search = GridSearchCV(model, params, cv=kf, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_cls)
    best_model = grid_search.best_estimator_
    mean_score = cross_val_score(best_model, X_train_scaled, y_train_cls, cv=cv, scoring='accuracy').mean()
    
    if mean_score > best_score_cls:
        best_classification_model = best_model
        best_score_cls = mean_score

plot_learning_curves(best_classification_model, X_train_scaled, y_train_cls, "Model Valuation", "accuracy")
joblib.dump(best_classification_model, "best_classification_model.pkl")

# **8. GridSearch per il modello di regressione**
best_regression_model = None
best_score_reg = float('-inf')

for model_name, (model, params) in regression_models.items():
    grid_search = GridSearchCV(model, params, cv=kf, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_reg_scaled.ravel())  # .ravel() per evitare warning su array 2D
    best_model = grid_search.best_estimator_
    mean_score = cross_val_score(best_model, X_train_scaled, y_train_reg_scaled.ravel(), cv=cv, scoring='neg_mean_squared_error').mean()
    
    if mean_score > best_score_reg:
        best_regression_model = best_model
        best_score_reg = mean_score

joblib.dump(best_regression_model, "best_regression_model.pkl")

# **9. Valutazione del modello di classificazione**
y_pred_cls = best_classification_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test_cls, y_pred_cls)
precision = precision_score(y_test_cls, y_pred_cls, average='weighted')
recall = recall_score(y_test_cls, y_pred_cls, average='weighted')
f1 = f1_score(y_test_cls, y_pred_cls, average='weighted')

print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# **10. Valutazione del modello di regressione**
y_pred_reg_scaled = best_regression_model.predict(X_test_scaled)
y_pred_reg = scaler_weight.inverse_transform(y_pred_reg_scaled.reshape(-1, 1))  # Riconvertiamo il peso alla scala originale

r2 = r2_score(y_test_reg, y_pred_reg)

print(f"R2 Score per il modello di regressione: {r2:.4f}")

# **11. Creazione grafico comparativo**
classification_metrics = {
    "Model": ["RandomForest"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1]
}

df_metrics = pd.DataFrame(classification_metrics)
df_metrics.set_index("Model").plot(kind="bar", figsize=(8, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Models")
plt.legend(title="Metrics")
plt.xticks(rotation=0)
plt.grid()
plt.show()
