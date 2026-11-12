"""Predicción de fallas en la red de distribución: comparación de modelos,
ajuste de XGBoost, SHAP, calibración y exportación de puntos para el GIS-MCDM.

Uso: uv run prediccion_fallas.py [ruta_csv]   (por defecto datasets/tabla.csv)
"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from scipy.stats import friedmanchisquare, randint, uniform, wilcoxon
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, learning_curve, train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATOS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "datasets" / "tabla.csv"
SALIDA = BASE / "resultados"
SALIDA.mkdir(exist_ok=True)

tiempos = {}


def cronometrar(etapa, inicio):
    tiempos[etapa] = time.perf_counter() - inicio
    print(f"\n[tiempo] {etapa}: {tiempos[etapa]:.1f} s")


def guardar_figura(nombre):
    plt.tight_layout()
    plt.savefig(SALIDA / nombre, dpi=300, bbox_inches="tight")
    plt.close("all")


# 1. Carga y preparación
df = pd.read_csv(DATOS)
df = df.drop(columns=["Fecha_de_I", "Hora_de_In", "Fecha_de_F", "Hora_de_Fi", "Duracion_F"])
df = df.dropna(subset=["RasterMantenimiento_1", "DensidadVegetacion"])
# Riesgo, Causa, Linea y Proteccion solo se conocen después de la falla (fuga de datos)
df = df.drop(columns=["Riesgo", "Causa", "Linea", "Proteccion"])
# Descargas atmosféricas casi no varía en la zona de estudio
print("Valores únicos DescargarAtmosKm2/ano:", df["DescargarAtmosKm2/ano"].nunique())
df = df.drop(columns=["DescargarAtmosKm2/ano"])

print(f"\nDataset final: {df.shape[0]} registros, {df.shape[1]} columnas")
print(f"Distribución target:\n{df['Falla'].value_counts()}")

X = df.drop(columns=["Falla"])
y = df["Falla"].map({"Sin Falla": 0, "Falla": 1})
numeric_cols = list(X.select_dtypes(include=["number"]).columns)
print(f"Variables predictivas ({len(numeric_cols)}): {numeric_cols}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
spw = (y_train == 0).sum() / (y_train == 1).sum()

preprocessor = ColumnTransformer(
    transformers=[("num", RobustScaler(), numeric_cols)], remainder="drop"
)

# 2. Comparación de 7 modelos x 5 estrategias de remuestreo
model_factories = {
    "LogReg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "SVM": lambda: SVC(class_weight="balanced", probability=True, random_state=42),
    "RandomForest": lambda: RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    "GradientBoosting": lambda: GradientBoostingClassifier(random_state=42),
    "XGBoost": lambda: XGBClassifier(
        eval_metric="logloss", scale_pos_weight=spw, random_state=42, n_jobs=-1
    ),
    "LightGBM": lambda: LGBMClassifier(is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    "MLP": lambda: MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42
    ),
}
sampler_factories = {
    "None": lambda: None,
    "SMOTE": lambda: SMOTE(random_state=42),
    "ADASYN": lambda: ADASYN(random_state=42),
    "RUS": lambda: RandomUnderSampler(random_state=42),
    "SMOTEENN": lambda: SMOTEENN(random_state=42),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scoring = {"f1": "f1", "roc_auc": "roc_auc", "pr_auc": "average_precision"}

inicio = time.perf_counter()
results, fold_scores = [], {}
for sampler_name, sampler_maker in sampler_factories.items():
    for model_name, model_maker in model_factories.items():
        sampler = sampler_maker()
        steps = [("preprocess", preprocessor)]
        if sampler is not None:
            steps.append(("sampler", sampler))
        steps.append(("clf", model_maker()))

        cv_res = cross_validate(Pipeline(steps), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        fold_scores[f"{sampler_name}_{model_name}"] = {"f1": cv_res["test_f1"]}
        results.append({
            "Sampler": sampler_name, "Model": model_name,
            "F1_mean": cv_res["test_f1"].mean(), "F1_std": cv_res["test_f1"].std(),
            "ROC_AUC_mean": cv_res["test_roc_auc"].mean(),
            "PR_AUC_mean": cv_res["test_pr_auc"].mean(),
        })
        r = results[-1]
        print(f"{sampler_name:10s} + {model_name:17s} | F1={r['F1_mean']:.3f}±{r['F1_std']:.3f} "
              f"| ROC={r['ROC_AUC_mean']:.3f} | PR={r['PR_AUC_mean']:.3f}")

results_df = pd.DataFrame(results).sort_values("F1_mean", ascending=False).reset_index(drop=True)
results_df.to_csv(SALIDA / "comparacion_modelos.csv", index=False)
print("\nTOP 10 CONFIGURACIONES POR F1:")
print(results_df.head(10).to_string(index=False))
cronometrar("Comparación de modelos (CV)", inicio)

# 3. Friedman + Wilcoxon sobre los 5 mejores modelos sin remuestreo
top_no_resample = results_df[results_df["Sampler"] == "None"].head(5)
top_keys = [f"None_{m}" for m in top_no_resample["Model"]]
stat, p_value = friedmanchisquare(*[fold_scores[k]["f1"] for k in top_keys])
print(f"\nFriedman (F1): chi2={stat:.2f}, p={p_value:.4g}")
if p_value < 0.05:
    for i in range(len(top_keys)):
        for j in range(i + 1, len(top_keys)):
            _, p_w = wilcoxon(fold_scores[top_keys[i]]["f1"], fold_scores[top_keys[j]]["f1"])
            sig = "***" if p_w < 0.001 else "**" if p_w < 0.01 else "*" if p_w < 0.05 else "ns"
            print(f"  {top_keys[i]:25s} vs {top_keys[j]:25s} | p={p_w:.4f} {sig}")

# 4. Ajuste de hiperparámetros (100 iteraciones, CV estratificada de 5 folds)
param_distributions = {
    "XGBoost": {
        "clf__max_depth": randint(3, 10),
        "clf__n_estimators": randint(50, 500),
        "clf__learning_rate": uniform(0.01, 0.3),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__min_child_weight": randint(1, 10),
        "clf__gamma": uniform(0, 1),
        "clf__reg_alpha": uniform(0, 1),
        "clf__reg_lambda": uniform(0.5, 2),
    },
    "LightGBM": {
        "clf__max_depth": randint(3, 15),
        "clf__n_estimators": randint(50, 500),
        "clf__learning_rate": uniform(0.01, 0.3),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__num_leaves": randint(15, 100),
        "clf__min_child_samples": randint(5, 50),
        "clf__reg_alpha": uniform(0, 1),
        "clf__reg_lambda": uniform(0, 1),
    },
    "RandomForest": {
        "clf__n_estimators": randint(50, 500),
        "clf__max_depth": [None] + list(range(5, 30, 5)),
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ["sqrt", "log2", 0.5, 0.7, 0.9],
    },
}
cv_tuning = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tuning_results = {}
for model_name, distribucion in param_distributions.items():
    inicio = time.perf_counter()
    search = RandomizedSearchCV(
        Pipeline([("preprocess", preprocessor), ("clf", model_factories[model_name]())]),
        param_distributions=distribucion, n_iter=100, scoring="f1",
        cv=cv_tuning, random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuning_results[model_name] = search
    y_proba = search.predict_proba(X_test)[:, 1]
    print(f"\n{model_name}: F1 CV={search.best_score_:.4f} | ROC test={roc_auc_score(y_test, y_proba):.4f} "
          f"| PR test={average_precision_score(y_test, y_proba):.4f}")
    print(f"Mejores hiperparámetros: {search.best_params_}")
    cronometrar(f"Ajuste {model_name}", inicio)

# 5. Evaluación final en test
inicio = time.perf_counter()
best_model_name = max(tuning_results, key=lambda k: tuning_results[k].best_score_)
final_pipe = tuning_results[best_model_name].best_estimator_
y_pred_final = final_pipe.predict(X_test)
y_proba_final = final_pipe.predict_proba(X_test)[:, 1]
print(f"\nModelo final: {best_model_name}")
print(classification_report(y_test, y_pred_final, digits=4))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_final):.4f}")
print(f"PR AUC:  {average_precision_score(y_test, y_proba_final):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_final), display_labels=["No Fault", "Fault"]).plot(
    ax=axes[0], cmap="Blues", values_format="d")
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_final, normalize="true"),
                       display_labels=["No Fault", "Fault"]).plot(ax=axes[1], cmap="Blues", values_format=".2%")
guardar_figura("confusion_matrix_final.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fpr, tpr, _ = roc_curve(y_test, y_proba_final)
axes[0].plot(fpr, tpr, "b-", lw=2, label=f"{best_model_name} (AUC={roc_auc_score(y_test, y_proba_final):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba_final)
axes[1].plot(recall_vals, precision_vals, "r-", lw=2,
             label=f"{best_model_name} (AP={average_precision_score(y_test, y_proba_final):.3f})")
axes[1].axhline(y=y_test.mean(), color="k", linestyle="--", alpha=0.5, label=f"Baseline ({y_test.mean():.2f})")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)
guardar_figura("roc_pr_curves.png")

# 6. SHAP sobre el conjunto de test
trained_model = final_pipe.named_steps["clf"]
X_test_processed = final_pipe.named_steps["preprocess"].transform(X_test)
shap_values = shap.TreeExplainer(trained_model).shap_values(X_test_processed)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap.summary_plot(shap_values, X_test_processed, feature_names=numeric_cols, show=False)
guardar_figura("shap_summary.png")
shap.summary_plot(shap_values, X_test_processed, feature_names=numeric_cols, plot_type="bar", show=False)
guardar_figura("shap_bar.png")

shap_importance = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({"Feature": numeric_cols, "Mean_abs_SHAP": shap_importance})
shap_df = shap_df.sort_values("Mean_abs_SHAP", ascending=False)
shap_df.to_csv(SALIDA / "shap_importancia.csv", index=False)
print("\nImportancia SHAP:\n" + shap_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, feat_idx in enumerate(np.argsort(shap_importance)[-3:][::-1]):
    shap.dependence_plot(feat_idx, shap_values, X_test_processed,
                         feature_names=numeric_cols, ax=axes[i], show=False)
guardar_figura("shap_dependence.png")
cronometrar("Evaluación final y SHAP", inicio)

# 7. Calibración isotónica de las probabilidades
inicio = time.perf_counter()
calibrated_pipe = CalibratedClassifierCV(final_pipe, method="isotonic", cv=5)
calibrated_pipe.fit(X_train, y_train)
y_proba_calibrated = calibrated_pipe.predict_proba(X_test)[:, 1]
print(f"\nBrier sin calibrar: {brier_score_loss(y_test, y_proba_final):.4f}")
print(f"Brier calibrado:    {brier_score_loss(y_test, y_proba_calibrated):.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
for proba, etiqueta, marca, color in [(y_proba_final, "uncalibrated", "s-", "blue"),
                                      (y_proba_calibrated, "calibrated", "o-", "red")]:
    frac_pos, prob_media = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
    ax.plot(prob_media, frac_pos, marca, color=color, label=f"{best_model_name} ({etiqueta})")
ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives", title="Calibration Curve")
ax.legend()
ax.grid(alpha=0.3)
guardar_figura("calibration_curve.png")
cronometrar("Calibración isotónica", inicio)

# 8. Puntos de falla predichos para el GIS-MCDM (incluye coordenadas: no se publica)
salida_gis = X_test.reset_index(drop=True).copy()
salida_gis["falla_true"] = y_test.reset_index(drop=True)
salida_gis["falla_pred"] = y_pred_final
salida_gis["fault_probability"] = y_proba_calibrated
salida_gis.to_csv(SALIDA / "predicted_failure_points_with_probabilities.csv", index=False)
print(f"\nPuntos predichos como falla: {int(salida_gis['falla_pred'].sum())} "
      f"(verdaderos positivos: {int(((salida_gis['falla_pred'] == 1) & (salida_gis['falla_true'] == 1)).sum())})")

# 9. Curva de aprendizaje
inicio = time.perf_counter()
train_sizes, train_scores, val_scores = learning_curve(
    final_pipe, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1", n_jobs=-1, random_state=42,
)
fig, ax = plt.subplots(figsize=(8, 5))
for scores, etiqueta, color in [(train_scores, "Training F1", "blue"), (val_scores, "Validation F1", "red")]:
    media, desv = scores.mean(axis=1), scores.std(axis=1)
    ax.plot(train_sizes, media, "o-", color=color, label=etiqueta)
    ax.fill_between(train_sizes, media - desv, media + desv, alpha=0.15, color=color)
ax.set(xlabel="Training Set Size", ylabel="F1 Score", title=f"Learning Curve - {best_model_name}")
ax.legend()
ax.grid(alpha=0.3)
guardar_figura("learning_curve.png")
cronometrar("Curva de aprendizaje", inicio)

tabla_tiempos = pd.Series(tiempos, name="segundos").round(1)
tabla_tiempos.to_csv(SALIDA / "tiempos.csv")
print("\nTIEMPOS POR ETAPA:\n" + tabla_tiempos.to_string())
