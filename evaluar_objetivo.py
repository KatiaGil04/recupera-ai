"""Experimento de viabilidad: predicción de castigo histórico en aceptados.

No reemplaza el modelo de cobranza ni recomienda aprobar créditos. Compara
predictores en las mismas cohortes: entrenamiento 2011-2013 y prueba 2014,
solo plazo de 36 meses y desenlaces Fully Paid / Charged Off. La exclusión de
no cerrados se cuantifica; no es una PD de 12 meses ni una validación actual.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from priorizacion_recobro import FEATURES, localizar_csv

NUMERICAS = [
    "loan_amnt", "annual_inc", "dti", "fico_range_low", "delinq_2yrs",
    "revol_util", "revol_bal", "inq_last_6mths", "open_acc", "total_acc",
    "pub_rec", "pub_rec_bankruptcies",
]
CATEGORICAS = ["home_ownership", "purpose", "emp_length", "verification_status"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ruta = localizar_csv()
    rng = np.random.default_rng(42)
    muestras = {"train": pd.DataFrame(), "test": pd.DataFrame()}
    limites = {"train": 100_000, "test": 50_000}
    conteos = {"train": Counter(), "test": Counter()}
    columnas = ["issue_d", "term", "loan_status", *NUMERICAS, *CATEGORICAS]
    with pd.read_csv(ruta, usecols=columnas, dtype="string", chunksize=100_000) as lector:
        for bloque in lector:
            fechas = pd.to_datetime(bloque["issue_d"], format="%b-%Y", errors="coerce")
            plazo = bloque["term"].str.strip().eq("36 months")
            mascaras = {"train": fechas.dt.year.between(2011, 2013), "test": fechas.dt.year.eq(2014)}
            for grupo, mascara in mascaras.items():
                cohorte = bloque.loc[mascara & plazo].copy()
                conteos[grupo].update(cohorte["loan_status"].fillna("SIN_ESTADO").value_counts().to_dict())
                cerrados = cohorte.loc[cohorte["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
                if cerrados.empty:
                    continue
                cerrados["_clave"] = rng.random(len(cerrados))
                muestras[grupo] = pd.concat([muestras[grupo], cerrados], ignore_index=True).nsmallest(limites[grupo], "_clave")

    for grupo, muestra in muestras.items():
        if muestra.empty:
            raise ValueError(f"Cohorte sin observaciones: {grupo}")
        for columna in NUMERICAS:
            valor = pd.to_numeric(muestra[columna], errors="coerce").astype("float64")
            muestra[columna] = valor.where(np.isfinite(valor) & valor.ge(0))
        for columna in CATEGORICAS:
            muestra[columna] = muestra[columna].astype(object).where(muestra[columna].notna(), np.nan)
        muestra["target_castigo"] = muestra["loan_status"].eq("Charged Off").astype(int)
        logging.info("%s: muestra de %s cuentas cerradas", grupo, len(muestra))

    train, test = muestras["train"], muestras["test"]
    y_train, y_test = train["target_castigo"], test["target_castigo"]
    resultados = {}
    for nombre, numericas, categoricas in [
        ("tres_variables_originales", list(FEATURES), []),
        ("variables_ampliadas", NUMERICAS, CATEGORICAS),
    ]:
        transformadores = [("numericas", SimpleImputer(strategy="median", add_indicator=True), numericas)]
        if categoricas:
            transformadores.append(("categoricas", Pipeline([
                ("imputacion", SimpleImputer(strategy="most_frequent")),
                ("codificacion", OneHotEncoder(handle_unknown="ignore")),
            ]), categoricas))
        modelo = Pipeline([
            ("preparacion", ColumnTransformer(transformadores)),
            ("bosque", RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=50,
                max_features=0.7, random_state=42, n_jobs=2,
            )),
        ])
        logging.info("Ajustando experimento: %s", nombre)
        modelo.fit(train[numericas + categoricas], y_train)
        prob = modelo.predict_proba(test[numericas + categoricas])[:, 1]
        resultados[nombre] = {
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "average_precision": float(average_precision_score(y_test, prob)),
            "brier_sin_calibrar": float(brier_score_loss(y_test, prob)),
            "predictores": numericas + categoricas,
        }
    prevalencia = float(y_train.mean())
    resultados["baseline_prevalencia"] = {
        "roc_auc": 0.5, "average_precision": float(y_test.mean()),
        "brier_sin_calibrar": float(brier_score_loss(y_test, np.full(len(y_test), prevalencia))),
    }
    informe = {
        "fuente": str(ruta), "target": "Charged Off=1, Fully Paid=0",
        "cohortes": {"train": "2011-2013, 36 meses", "test": "2014, 36 meses"},
        "estados_completos_cohortes": {k: dict(v) for k, v in conteos.items()},
        "muestra": {k: len(v) for k, v in muestras.items()},
        "prevalencia_castigo": {k: float(v["target_castigo"].mean()) for k, v in muestras.items()},
        "seed": 42, "resultados": resultados,
        "limitaciones": [
            "Experimento exploratorio sin ajuste de hiperparámetros ni calibración.",
            "Predice castigo histórico entre cerrados, no mora a 12 meses ni cura.",
            "No mide respuesta a llamadas ni sirve para aplicar a rechazados sin evaluar selección.",
            "Cohortes antiguas: no acredita rendimiento en cartera actual.",
            "Los AUC no son comparables directamente con el test aleatorio del prototipo original.",
        ],
    }
    salida = Path(__file__).resolve().parent / "outputs" / "auditoria"
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "viabilidad_prediccion.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(informe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
