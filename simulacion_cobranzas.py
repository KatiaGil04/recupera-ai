# %% [markdown]
# # Cohortes temporales y simulación mensual de cobranza
# Dos ejercicios separados: evaluación retrospectiva de castigo con datos reales
# y escenario mensual sintético con una matriz de transición SUPUESTA.
# No se reconstruye el historial real a partir de un último pago ni se evalúa
# el modelo contra etiquetas inventadas. No hay variables macroeconómicas aquí.

# %% 1. Dependencias y parámetros
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from evaluar_objetivo import NUMERICAS as NUM_REF, CATEGORICAS as CAT_REF
from priorizacion_recobro import localizar_csv

LOGGER = logging.getLogger(__name__)
# Referencia: las mismas 16 variables del experimento AUC=0.6484.
# Variante solicitada: sustituye pub_rec_bankruptcies por grade, manteniendo 16.
NUM_GRADE = [c for c in NUM_REF if c != "pub_rec_bankruptcies"]
CAT_GRADE = [*CAT_REF, "grade"]
ESTADOS = ("CORRIENTE", "1-30", "31-60", "61-90", "91-120", "121+", "PAGADO", "CASTIGADO")
MORA = ESTADOS[1:6]
COLUMNAS_PLAN = [
    "orden_llamada_mes", "id", "fecha_corte", "meses_en_libros", "estado_origen", "estado_simulado",
    "dias_mora_sinteticos", "saldo_origen", "out_prncp", "out_prncp_observado", "probabilidad_perdida",
    "tipo_dato", "origen_saldo", "valor_monetario_en_riesgo",
]
COLUMNAS_ROLL = [
    "estado_origen", "estado_simulado", "cuentas", "saldo_transicion", "cuentas_origen",
    "saldo_total_origen", "roll_rate_cuentas", "roll_rate_saldo", "mes_origen", "mes_destino", "tipo_dato",
]
DPD_SINTETICO = np.array([0, 15, 45, 75, 105, 150, 0, 180])
# Escenario didáctico editable; NO estimado con Lending Club.
MATRIZ_SUPUESTA = np.array([
    [.965, .025, 0, 0, 0, 0, .010, 0],
    [.450, .200, .350, 0, 0, 0, 0, 0],
    [.200, .050, .250, .500, 0, 0, 0, 0],
    [.100, 0, .050, .150, .700, 0, 0, 0],
    [.050, 0, 0, .050, .100, .800, 0, 0],
    [.020, 0, 0, 0, 0, .280, 0, .700],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
])


@dataclass(frozen=True)
class ConfigTemporal:
    csv_path: Path | None = None
    salida: Path = Path("outputs/simulacion")
    chunksize: int = 100_000
    max_train: int = 100_000
    muestra_evaluacion: int = 50_000
    arboles: int = 100
    jobs: int = 2
    seed: int = 42
    meses_colas: int = 6
    fecha_corte_observada: str | None = None
    guardar_panel: bool = False

    def __post_init__(self):
        if min(self.chunksize, self.max_train, self.muestra_evaluacion, self.arboles) < 1:
            raise ValueError("Tamaños de bloque, muestra y número de árboles deben ser positivos.")
        if self.meses_colas < 0 or self.jobs == 0:
            raise ValueError("meses_colas debe ser >=0 y jobs distinto de cero.")
        if self.fecha_corte_observada:
            pd.Timestamp(self.fecha_corte_observada)


# %% 2. Fechas reales: duración hasta el último pago, no DPD observado
def preparar_fechas(datos: pd.DataFrame, corte: str | None = None) -> pd.DataFrame:
    """Conserva filas con fechas/predictores nulos; marca inconsistencias."""
    datos = datos.copy()
    for campo in ("issue_d", "last_pymnt_d"):
        texto = datos[campo]
        fecha = (texto if pd.api.types.is_datetime64_any_dtype(texto)
                 else pd.to_datetime(texto, format="%b-%Y", errors="coerce"))
        datos[f"{campo}_invalida"] = texto.notna() & fecha.isna()
        datos[campo] = fecha
    meses = ((datos["last_pymnt_d"].dt.year - datos["issue_d"].dt.year) * 12
             + datos["last_pymnt_d"].dt.month - datos["issue_d"].dt.month)
    datos["fecha_pago_anterior_otorgamiento"] = meses.lt(0)
    datos["meses_hasta_ultimo_pago"] = meses.where(meses.ge(0)).astype("Int64")
    # Alias explícitamente aproximado: no acredita pagos continuos ni inicio de mora.
    datos["meses_activos_antes_del_atraso_aprox"] = datos["meses_hasta_ultimo_pago"]
    datos["dias_desde_ultimo_pago_aprox"] = pd.Series(pd.NA, index=datos.index, dtype="Int64")
    if corte:
        dias = (pd.Timestamp(corte) - datos["last_pymnt_d"]).dt.days
        datos["dias_desde_ultimo_pago_aprox"] = dias.where(dias.ge(0)).astype("Int64")
    return datos


def cargar_cohortes(config: ConfigTemporal) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Muestrea train y benchmark; conserva TODO 2014/36m, incluso con nulos."""
    ruta = localizar_csv(config.csv_path)
    columnas = list(dict.fromkeys([
        "id", "issue_d", "last_pymnt_d", "term", "loan_status", "grade",
        "int_rate", "installment", "total_pymnt", "out_prncp", *NUM_REF, *CAT_REF,
    ]))
    rng = np.random.default_rng(config.seed)
    muestras = {"train": pd.DataFrame(), "test": pd.DataFrame()}
    limites = {"train": config.max_train, "test": config.muestra_evaluacion}
    conteos = {"train": Counter(), "test": Counter()}
    test_completo = []
    sin_estado = 0
    with pd.read_csv(ruta, usecols=columnas, dtype="string", chunksize=config.chunksize) as lector:
        for numero, bloque in enumerate(lector, 1):
            sin_estado += int(bloque["loan_status"].isna().sum())
            fecha = pd.to_datetime(bloque["issue_d"], format="%b-%Y", errors="coerce")
            plazo = bloque["term"].str.strip().eq("36 months")
            mascaras = {"train": fecha.dt.year.between(2011, 2013), "test": fecha.dt.year.eq(2014)}
            for grupo, mascara in mascaras.items():
                cohorte = bloque.loc[mascara & plazo & bloque["loan_status"].notna()].copy()
                conteos[grupo].update(cohorte["loan_status"].value_counts().to_dict())
                if grupo == "test" and not cohorte.empty:
                    test_completo.append(cohorte)
                cerrados = cohorte.loc[cohorte["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
                if not cerrados.empty:
                    cerrados["_clave"] = rng.random(len(cerrados))
                    muestras[grupo] = pd.concat([muestras[grupo], cerrados], ignore_index=True).nsmallest(limites[grupo], "_clave")
            if numero == 1 or numero % 5 == 0:
                LOGGER.info("Bloques revisados: %s", numero)
    if muestras["train"].empty or not test_completo or muestras["test"].empty:
        raise ValueError("Faltan cohortes o desenlaces cerrados para evaluar.")
    completo = pd.concat(test_completo, ignore_index=True)

    def limpiar(datos):
        datos = preparar_fechas(datos, config.fecha_corte_observada)
        for c in [*NUM_REF, "int_rate", "installment", "total_pymnt", "out_prncp"]:
            numero = pd.to_numeric(datos[c].str.strip().str.removesuffix("%"), errors="coerce").astype("float64")
            datos[c] = numero.where(np.isfinite(numero) & numero.ge(0))
        for c in [*CAT_REF, "grade"]:
            datos[c] = datos[c].astype(object).where(datos[c].notna(), np.nan)
        return datos.reset_index(drop=True)

    train, benchmark, completo = [limpiar(d) for d in [muestras["train"], muestras["test"], completo]]
    if completo["id"].isna().any() or completo["id"].duplicated().any():
        raise ValueError("2014 necesita identificadores completos y únicos; no se hará una unión ambigua.")
    resumen = {"fuente": str(ruta), "filas_sin_estado_omitidas": sin_estado,
               "estados_cohortes": {k: dict(v) for k, v in conteos.items()},
               "n_train": len(train), "n_benchmark": len(benchmark), "n_2014": len(completo)}
    return train, benchmark, completo, resumen


# %% 3. Modelo de referencia de 16 variables y variante con grade
def construir_modelo(numericas, categoricas, config: ConfigTemporal) -> Pipeline:
    """Imputación aprendida solo en train; categorías nuevas admitidas en test."""
    return Pipeline([
        ("preparacion", ColumnTransformer([
            ("numericas", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numericas),
            ("categoricas", Pipeline([
                ("imputacion", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]), categoricas),
        ])),
        ("bosque", RandomForestClassifier(
            n_estimators=config.arboles, max_depth=12, min_samples_leaf=50,
            max_features=0.7, random_state=config.seed, n_jobs=config.jobs,
        )),
    ])


def metricas(y, probabilidades) -> dict:
    return {"roc_auc": float(roc_auc_score(y, probabilidades)),
            "average_precision": float(average_precision_score(y, probabilidades)),
            "brier": float(brier_score_loss(y, probabilidades))}


def evaluar_modelos(train, benchmark, test, config):
    """Test por año de otorgamiento; etiquetas finales observadas retrospectivamente."""
    import matplotlib.pyplot as plt

    y_train = train["loan_status"].eq("Charged Off").astype(int)
    y_bench = benchmark["loan_status"].eq("Charged Off").astype(int)
    cerrados = test["loan_status"].isin(["Fully Paid", "Charged Off"])
    if y_train.nunique() != 2 or y_bench.nunique() != 2:
        raise ValueError("Se requieren ambos desenlaces en train y evaluación.")
    modelos, informe = {}, {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, numericas, categoricas in [
        ("referencia_16", NUM_REF, CAT_REF), ("con_grade_16", NUM_GRADE, CAT_GRADE),
    ]:
        LOGGER.info("Entrenando %s con %s predictores", nombre, len(numericas + categoricas))
        modelo = construir_modelo(numericas, categoricas, config)
        modelo.fit(train[numericas + categoricas], y_train)
        prob = modelo.predict_proba(benchmark[numericas + categoricas])[:, 1]
        informe[nombre] = {"muestra_2014": metricas(y_bench, prob), "predictores": numericas + categoricas}
        fpr, tpr, _ = roc_curve(y_bench, prob)
        ax.plot(fpr, tpr, label=f"{nombre}: AUC={informe[nombre]['muestra_2014']['roc_auc']:.4f}")
        modelos[nombre] = modelo
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
           title=f"Castigo: muestra de {len(benchmark):,} cerrados de 2014", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    prob_test = modelos["con_grade_16"].predict_proba(test[NUM_GRADE + CAT_GRADE])[:, 1]
    if test.loc[cerrados, "loan_status"].nunique() == 2:
        informe["con_grade_16"]["todos_cerrados_2014"] = metricas(
            test.loc[cerrados, "loan_status"].eq("Charged Off").astype(int), prob_test[cerrados],
        )
    informe["filas_2014_puntuadas"] = len(test)
    informe["filas_2014_evaluables"] = int(cerrados.sum())
    informe["auc_referencia_previo"] = .6484280024753116
    informe["diferencia_auc_referencia"] = informe["referencia_16"]["muestra_2014"]["roc_auc"] - .6484280024753116
    return modelos["con_grade_16"], prob_test, informe, fig


# %% [markdown]
# ## 4. Escenario mensual sintético
# La trayectoria se sortea con la matriz supuesta y semilla fija, sin consultar
# loan_status final, last_pymnt_d ni pagos acumulados. La PD de originación se
# mantiene fija; cambian elegibilidad y saldo. No se estima efecto de llamadas.
# Saldo: amortización contractual simplificada. Meses en mora no pagan capital;
# al curar se asume regularización del capital vencido. PAGADO=saldo cero;
# CASTIGADO conserva la exposición y sale del plan de llamadas ordinario.

# %%
def saldo_contractual(principal, tasa_mensual, cuota, meses):
    """Capital programado, con tratamiento de tasa cero y sin imputar dinero."""
    factor = np.power(1 + tasa_mensual, meses)
    cociente = np.divide(factor - 1, tasa_mensual, out=meses.astype(float).copy(), where=tasa_mensual != 0)
    return np.clip(principal * factor - cuota * cociente, 0, principal)


def simular_meses(cartera, probabilidades, config, matriz=None):
    """Generador por mes: memoria proporcional a cuentas, no a cuentas x meses."""
    matriz = np.array(MATRIZ_SUPUESTA if matriz is None else matriz, dtype=float)
    if matriz.shape != (8, 8) or not np.isfinite(matriz).all() or (matriz < 0).any() or not np.allclose(matriz.sum(axis=1), 1):
        raise ValueError("La matriz debe ser 8x8, finita, no negativa y sumar 1 por fila.")
    if not np.array_equal(matriz[6:], np.eye(8)[6:]):
        raise ValueError("PAGADO y CASTIGADO deben ser estados absorbentes.")
    if len(probabilidades) != len(cartera) or not np.isfinite(probabilidades).all() or ((np.asarray(probabilidades) < 0) | (np.asarray(probabilidades) > 1)).any():
        raise ValueError("Cada cuenta necesita una probabilidad finita entre cero y uno.")
    requeridas = cartera[["loan_amnt", "int_rate", "installment"]].astype(float)
    if not np.isfinite(requeridas).all().all() or not (requeridas["loan_amnt"].gt(0) & requeridas["installment"].gt(0) & requeridas["int_rate"].ge(0)).all():
        raise ValueError("Para simular saldos se requieren importe/cuota positivos y tasa válida.")
    if cartera.empty:
        return
    issue = cartera["issue_d"].dt.to_period("M")
    if issue.isna().any() or cartera["id"].isna().any() or cartera["id"].duplicated().any():
        raise ValueError("La simulación requiere fechas e identificadores únicos válidos.")
    rng = np.random.default_rng(config.seed)
    ordinal_issue = issue.astype("int64").to_numpy()
    estados = np.zeros(len(cartera), dtype=int)
    principal = cartera["loan_amnt"].to_numpy(float)
    ids = cartera["id"].to_numpy()
    saldo_observado = cartera["out_prncp"].to_numpy()
    probabilidades = np.asarray(probabilidades)
    nombres_estados = np.array(ESTADOS)
    saldo = principal.copy()
    tasa = cartera["int_rate"].to_numpy(float) / 1200
    cuota = cartera["installment"].to_numpy(float)
    acumulada = matriz.cumsum(axis=1)
    acumulada[:, -1] = 1.0
    for mes in pd.period_range(issue.min() + 1, issue.max() + 36 + config.meses_colas, freq="M"):
        mob = mes.ordinal - ordinal_issue
        activa = (mob >= 1) & (estados < 6)
        indices = np.flatnonzero(activa)
        if not len(indices):
            continue
        origen = estados[indices].copy()
        saldo_origen = saldo[indices].copy()
        aleatorios = rng.random(len(indices))
        destino = (aleatorios[:, None] >= acumulada[origen]).sum(axis=1)
        # La cancelación por vencimiento contractual es otra hipótesis explícita.
        destino[(mob[indices] >= 36) & (destino == 0)] = 6
        estados[indices] = destino
        saldo_teorico = saldo_contractual(principal[indices], tasa[indices], cuota[indices], mob[indices])
        regular = destino == 0
        saldo[indices[regular]] = np.minimum(saldo_origen[regular], saldo_teorico[regular])
        saldo[indices[destino == 6]] = 0
        fila = pd.DataFrame({
            "id": ids[indices], "fecha_corte": mes.to_timestamp(how="end").normalize(),
            "meses_en_libros": mob[indices], "estado_origen": nombres_estados[origen],
            "estado_simulado": nombres_estados[destino], "dias_mora_sinteticos": DPD_SINTETICO[destino],
            "saldo_origen": saldo_origen, "out_prncp": saldo[indices],
            "out_prncp_observado": saldo_observado[indices],
            "probabilidad_perdida": probabilidades[indices],
            "tipo_dato": "SIMULADO", "origen_saldo": "AMORTIZACION_SUPUESTA",
        })
        fila["valor_monetario_en_riesgo"] = fila["probabilidad_perdida"] * fila["out_prncp"]
        yield fila


def generar_plan(test, probabilidades, config):
    """Escribe cada mes sin acumular el plan completo; publica al terminar.

    Solo retiene el mes, los IDs gestionados y agregados pequeños de roll rate.
    El CSV del plan/panel se reemplaza tras cerrar sus archivos temporales,
    evitando publicar una lista parcial si falla la simulación.
    """
    salida = Path(config.salida)
    salida.mkdir(parents=True, exist_ok=True)
    valido = (
        test["loan_amnt"].gt(0) & test["installment"].gt(0) & test["int_rate"].ge(0)
        & np.isfinite(test[["loan_amnt", "installment", "int_rate"]].to_numpy(float)).all(axis=1)
    )
    test.loc[~valido].to_csv(salida / "cuentas_revision_simulacion.csv", index=False, encoding="utf-8-sig")
    cartera = test.loc[valido].reset_index(drop=True)
    probabilidades = np.asarray(probabilidades, dtype=float)
    if probabilidades.shape != (len(test),) or not np.isfinite(probabilidades).all() or ((probabilidades < 0) | (probabilidades > 1)).any():
        raise ValueError("Cada cuenta necesita una probabilidad finita entre cero y uno.")
    scores = probabilidades[valido]
    tasas = []
    filas_plan = meses_plan = 0
    ids_plan = set()
    nombre_plan = "plan_llamadas_priorizado.csv"
    nombre_panel = "panel_mensual_simulado.csv"
    with TemporaryDirectory(prefix=".plan-", dir=salida) as temporal:
        temporal = Path(temporal)
        with ExitStack() as archivos:
            escritor_plan = archivos.enter_context((temporal / nombre_plan).open("w", encoding="utf-8-sig", newline=""))
            pd.DataFrame(columns=COLUMNAS_PLAN).to_csv(escritor_plan, index=False)
            escritor_panel = None
            if config.guardar_panel:
                escritor_panel = archivos.enter_context((temporal / nombre_panel).open("w", encoding="utf-8-sig", newline=""))
                pd.DataFrame(columns=COLUMNAS_PLAN[1:]).to_csv(escritor_panel, index=False)
            for mes in simular_meses(cartera, scores, config):
                if escritor_panel is not None:
                    mes.to_csv(escritor_panel, header=False, index=False)
                plan = mes.loc[mes["estado_simulado"].isin(MORA) & mes["out_prncp"].gt(0)].copy()
                plan = plan.sort_values(["valor_monetario_en_riesgo", "id"], ascending=[False, True])
                plan.insert(0, "orden_llamada_mes", np.arange(1, len(plan) + 1))
                if not plan.empty:
                    plan.to_csv(escritor_plan, header=False, index=False)
                    filas_plan += len(plan)
                    meses_plan += 1
                    ids_plan.update(plan["id"])
                agregado = mes.groupby(["estado_origen", "estado_simulado"], observed=True).agg(
                    cuentas=("id", "size"), saldo_transicion=("saldo_origen", "sum"),
                ).reset_index()
                rejilla = pd.MultiIndex.from_product([mes["estado_origen"].unique(), ESTADOS], names=["estado_origen", "estado_simulado"]).to_frame(index=False)
                agregado = rejilla.merge(agregado, how="left").fillna({"cuentas": 0, "saldo_transicion": 0})
                grupos = agregado.groupby("estado_origen", observed=True)
                agregado["cuentas_origen"] = grupos["cuentas"].transform("sum")
                agregado["saldo_total_origen"] = grupos["saldo_transicion"].transform("sum")
                agregado["roll_rate_cuentas"] = agregado["cuentas"] / agregado["cuentas_origen"]
                agregado["roll_rate_saldo"] = agregado["saldo_transicion"] / agregado["saldo_total_origen"].replace(0, np.nan)
                agregado["mes_origen"] = str(mes["fecha_corte"].iloc[0].to_period("M") - 1)
                agregado["mes_destino"] = str(mes["fecha_corte"].iloc[0].to_period("M"))
                agregado["tipo_dato"] = "SIMULADO"
                tasas.append(agregado)
        (temporal / nombre_plan).replace(salida / nombre_plan)
        if config.guardar_panel:
            (temporal / nombre_panel).replace(salida / nombre_panel)
    roll = pd.concat(tasas, ignore_index=True) if tasas else pd.DataFrame(columns=COLUMNAS_ROLL)
    roll.to_csv(salida / "roll_rates_simulados.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(MATRIZ_SUPUESTA, index=ESTADOS, columns=ESTADOS).to_csv(salida / "matriz_transicion_supuesta.csv")
    resumen = {"cuentas_simuladas": len(cartera), "cuentas_revision_simulacion": int((~valido).sum()),
               "filas_plan_cuenta_mes": filas_plan, "cuentas_distintas_plan": len(ids_plan),
               "meses_plan": meses_plan}
    return resumen, roll


def graficar_roll_simulado(roll, origen="31-60"):
    """Curvas mensuales del escenario supuesto, identificadas como simulación."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    datos = roll.loc[roll["estado_origen"].eq(origen)]
    if datos.empty:
        return None
    curvas = datos.pivot(index="mes_destino", columns="estado_simulado", values="roll_rate_cuentas").sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for destino in curvas:
        if curvas[destino].fillna(0).ne(0).any():
            ax.plot(pd.to_datetime(curvas.index), curvas[destino], label=destino)
    ax.set(title=f"SIMULACIÓN: transiciones desde {origen}", xlabel="Mes de destino", ylabel="Proporción de cuentas", ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(title="Destino", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


# %% 5. Ejecución completa y exportación de evidencia
def ejecutar(config: ConfigTemporal | None = None):
    config = config or ConfigTemporal()
    salida = Path(config.salida)
    salida.mkdir(parents=True, exist_ok=True)
    train, benchmark, test, datos = cargar_cohortes(config)
    modelo, probabilidades, evaluacion, figura = evaluar_modelos(train, benchmark, test, config)
    figura.savefig(salida / "curva_roc.png", dpi=180)
    test["probabilidad_perdida"] = probabilidades
    test["valor_monetario_en_riesgo_observado"] = probabilidades * test["out_prncp"]
    columnas = ["id", "issue_d", "last_pymnt_d", "loan_status", "meses_hasta_ultimo_pago",
                "meses_activos_antes_del_atraso_aprox", "dias_desde_ultimo_pago_aprox",
                "fecha_pago_anterior_otorgamiento", "last_pymnt_d_invalida", "total_pymnt", "out_prncp",
                "probabilidad_perdida", "valor_monetario_en_riesgo_observado"]
    test[columnas].to_csv(salida / "cohorte_2014_observada.csv", index=False, encoding="utf-8-sig")
    simulacion, roll = generar_plan(test, probabilidades, config)
    figura_roll = graficar_roll_simulado(roll) if not roll.empty else None
    if figura_roll is not None:
        figura_roll.savefig(salida / "curvas_roll_rate_simuladas.png", dpi=180)
    informe = {"datos": datos, "evaluacion": evaluacion, "simulacion": simulacion,
        "cuentas_2014_saldo_observado_positivo": int(test["out_prncp"].gt(0).sum()),
        "supuestos": {"seed": config.seed, "meses_colas": config.meses_colas, "PD_fija": True,
                      "matriz": "Manual, no estimada con Lending Club", "fecha_corte_observada": config.fecha_corte_observada},
        "limitaciones": [
            "Roll rates, días de mora y saldos del plan son sintéticos, no observados.",
            "last_pymnt_d es el último mes con pago; no identifica inicio de mora.",
            "total_pymnt, out_prncp, fechas de pago y target no son predictores.",
            "Los scores de Random Forest no se calibraron; PD es un proxy de castigo final.",
            "El producto PD por saldo omite LGD y no es pérdida esperada validada.",
            "La partición por otorgamiento es retrospectiva: etiquetas finales de 2013 pudieron conocerse después de 2014.",
            "No hay macroeconomía, efecto causal de llamadas ni validación con roll rates reales.",
            "No se evalúa el AUC contra trayectorias sintéticas ni se condicionan a desenlaces finales.",
        ]}
    (salida / "metricas_temporales.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(informe, ensure_ascii=False, indent=2))
    return {"modelo": modelo, "cohorte_2014": test, "metricas": informe, "figura_roc": figura,
            "roll_rates": roll, "figura_roll": figura_roll}


# %% CLI: en Jupyter importa ejecutar() o usa simulacion_cobranzas.ipynb
def main():
    parser = argparse.ArgumentParser(description="Escenario sintético mensual de cobranza y evaluación por cohortes.")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--salida", type=Path, default=Path("outputs/simulacion"))
    parser.add_argument("--fecha-corte-observada", default=None)
    parser.add_argument("--guardar-panel", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ejecutar(ConfigTemporal(csv_path=args.csv, salida=args.salida, jobs=args.jobs,
                           fecha_corte_observada=args.fecha_corte_observada, guardar_panel=args.guardar_panel))


if __name__ == "__main__":
    main()
