# %% [markdown]
# # Priorización monetaria de cobranza — Lending Club
# Prototipo reproducible: Fully Paid=1, Charged Off=0.
# Las probabilidades aproximan el desenlace de cuentas cerradas; no miden cura
# tras mora, respuesta a llamadas ni dinero recuperable validado.
# Los estados pertenecen al corte histórico del archivo, no a la fecha de hoy.

# %% 1. Importaciones y configuración
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)
COLUMNAS = (
    "loan_status", "delinq_2yrs", "revol_util", "annual_inc",
    "last_pymnt_amnt", "out_prncp",
)
FEATURES = ("delinq_2yrs", "revol_util", "annual_inc")
TARGET = {"Fully Paid": 1, "Charged Off": 0}
ESTADOS_MORA = ("Late (31-120 days)", "In Grace Period")
LIMITACIONES = [
    "prob_recobro aproxima Fully Paid entre cuentas cerradas, no cura tras mora.",
    "No mide efecto de llamadas, recuperaciones parciales ni costos de gestión.",
    "La mora pertenece al corte histórico del CSV, no necesariamente a hoy.",
    "El test aleatorio de cerrados no valida transferencia a morosos ni a futuro.",
    "No hay observaciones mensuales para calcular roll rates en este CSV.",
]


@dataclass(frozen=True)
class Config:
    """Parámetros editables desde Jupyter o desde la línea de comandos."""

    csv_path: Path | None = None
    chunksize: int = 100_000
    max_historicos: int = 200_000
    arboles: int = 120
    jobs: int = -1
    seed: int = 42
    id_col: str | None = None

    def __post_init__(self) -> None:
        if min(self.chunksize, self.max_historicos, self.arboles) < 1:
            raise ValueError("chunksize, max_historicos y arboles deben ser positivos.")
        if self.jobs == 0:
            raise ValueError("jobs no puede ser cero.")
        if self.id_col in (*COLUMNAS, "indice_origen", "target", "_clave_muestreo"):
            raise ValueError("id_col debe ser una columna de identificación independiente.")


@dataclass
class Resultado:
    """Tablas de salida y modelo ajustado accesibles desde Jupyter."""

    priorizadas: pd.DataFrame
    revision: pd.DataFrame
    metricas: dict
    calibracion: pd.DataFrame
    calidad: pd.DataFrame
    modelo: CalibratedClassifierCV


def localizar_csv(ruta: Path | None = None) -> Path:
    """Localiza CSV o gzip sin descomprimir ni cargar el archivo completo."""
    if ruta is not None:
        archivo = Path(ruta).expanduser().resolve()
        if not archivo.is_file():
            raise FileNotFoundError(f"No existe el CSV: {archivo}")
        return archivo
    proyecto = Path(__file__).resolve().parent
    carpetas = list(dict.fromkeys([
        Path.cwd(), Path.cwd() / "data", Path.cwd() / "data" / "archive", Path.cwd() / "archive",
        proyecto, proyecto / "data", proyecto / "data" / "archive", proyecto / "archive",
        Path.home() / "Downloads" / "archive",
    ]))
    # Primero las copias del proyecto; dentro de cada carpeta se prefiere el
    # CSV descomprimido, incluido el subdirectorio creado al extraer el gzip.
    nombres = [
        "accepted_2007_to_2018Q4.csv",
        "accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
        "accepted_2007_to_2018Q4.csv.gz",
    ]
    for carpeta in carpetas:
        for nombre in nombres:
            archivo = carpeta / nombre
            if archivo.is_file():
                return archivo.resolve()
    raise FileNotFoundError("No se encontró Lending Club. Indica su ruta con --csv.")


# %% [markdown]
# ## 2. Lectura y segmentación por bloques
# Cada histórico recibe una clave aleatoria; se retienen las N menores.
# Así todos tienen igual probabilidad de entrar, aun al final del archivo.
# La muestra limita memoria del entrenamiento. Se conservan TODOS los morosos.

# %%
def limpiar_bloque(bloque: pd.DataFrame) -> pd.DataFrame:
    """Normaliza números y porcentajes; no aprende imputaciones globales."""
    bloque = bloque.copy()
    bloque["loan_status"] = bloque["loan_status"].str.strip()
    for columna in COLUMNAS[1:]:
        texto = bloque[columna].astype("string").str.strip()
        if columna == "revol_util":
            texto = texto.str.removesuffix("%")
        valores = pd.to_numeric(texto, errors="coerce").astype("float64")
        bloque[columna] = valores.where(np.isfinite(valores) & valores.ge(0))
    return bloque


def cargar_segmentos(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Devuelve muestra histórica, todos los morosos y conteos del archivo."""
    ruta = localizar_csv(config.csv_path)
    seleccion = list(COLUMNAS) + ([config.id_col] if config.id_col else [])
    cabecera = pd.read_csv(ruta, nrows=0, compression="infer", encoding="utf-8-sig")
    faltantes = set(seleccion) - set(cabecera.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    rng = np.random.default_rng(config.seed)
    muestra = pd.DataFrame()
    morosos = []
    estados: Counter = Counter()
    leidas = 0
    with pd.read_csv(
        ruta, usecols=seleccion, dtype="string", chunksize=config.chunksize,
        compression="infer", encoding="utf-8-sig",
    ) as lector:
        for numero, bruto in enumerate(lector, start=1):
            # Registro según pandas, no línea física: los campos del CSV
            # pueden contener saltos de línea entrecomillados.
            bruto["indice_origen"] = np.arange(leidas, leidas + len(bruto))
            leidas += len(bruto)
            bruto["loan_status"] = bruto["loan_status"].str.strip()
            estados.update(bruto["loan_status"].fillna("SIN_ESTADO").value_counts().to_dict())
            segmento = limpiar_bloque(bruto.loc[
                bruto["loan_status"].isin([*TARGET, *ESTADOS_MORA])
            ])
            cerrados = segmento.loc[segmento["loan_status"].isin(TARGET)].copy()
            if not cerrados.empty:
                cerrados["target"] = cerrados["loan_status"].map(TARGET).astype("int8")
                cerrados["_clave_muestreo"] = rng.random(len(cerrados))
                muestra = pd.concat([
                    muestra, cerrados[[*FEATURES, "target", "_clave_muestreo"]]
                ], ignore_index=True)
                if len(muestra) > config.max_historicos:
                    muestra = muestra.nsmallest(config.max_historicos, "_clave_muestreo")
            mora = segmento.loc[segmento["loan_status"].isin(ESTADOS_MORA)]
            if not mora.empty:
                morosos.append(mora.copy())
            if numero == 1 or numero % 5 == 0:
                LOGGER.info("Registros leídos: %s; históricos retenidos: %s", leidas, len(muestra))

    if muestra.empty:
        raise ValueError("No se encontraron cuentas Fully Paid / Charged Off.")
    muestra = muestra.drop(columns="_clave_muestreo").reset_index(drop=True)
    mora_total = (
        pd.concat(morosos, ignore_index=True) if morosos
        else pd.DataFrame(columns=[*seleccion, "indice_origen"])
    )
    resumen = {
        "csv": str(ruta), "filas_leidas": leidas, "estados": dict(estados),
        "historicos_disponibles": sum(estados.get(estado, 0) for estado in TARGET),
        "historicos_muestra": len(muestra), "cuentas_mora": len(mora_total),
        "target_muestra": {str(k): int(v) for k, v in muestra["target"].value_counts().items()},
    }
    return muestra, mora_total, resumen


# %% [markdown]
# ## 3. Entrenamiento, imputación y calibración
# Último pago y saldo quedan fuera de X por observarse al cierre del histórico.
# Se reserva 20 % de test. En el 80 % restante, 3 folds generan predicciones
# fuera de muestra para calibrar. El test no interviene en ajuste ni imputación.
# ensemble=False conserva un único bosque final; no se rebalancean las clases.

# %%
def entrenar(
    historicos: pd.DataFrame, config: Config,
) -> tuple[CalibratedClassifierCV, dict, pd.DataFrame]:
    """Ajusta y evalúa el proxy de desenlace sobre históricos cerrados."""
    conteos = historicos["target"].value_counts()
    if len(conteos) != 2 or conteos.min() < 10:
        raise ValueError("Se requieren al menos 10 cuentas cerradas por clase en la muestra.")
    sin_datos = [c for c in FEATURES if historicos[c].notna().sum() == 0]
    if sin_datos:
        raise ValueError(f"Predictores completamente vacíos en históricos: {sin_datos}")
    X_train, X_test, y_train, y_test = train_test_split(
        historicos[list(FEATURES)], historicos["target"],
        test_size=0.20, stratify=historicos["target"], random_state=config.seed,
    )
    base = Pipeline([
        ("imputacion", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("bosque", RandomForestClassifier(
            n_estimators=config.arboles, max_depth=12, min_samples_leaf=50,
            max_features=1.0, random_state=config.seed, n_jobs=config.jobs,
        )),
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.seed)
    modelo = CalibratedClassifierCV(
        estimator=base, method="sigmoid", cv=cv, ensemble=False, n_jobs=1,
    )
    modelo.fit(X_train, y_train)
    positiva = int(np.flatnonzero(modelo.classes_ == 1)[0])
    prob = modelo.predict_proba(X_test)[:, positiva]
    prevalencia = float(y_train.mean())

    def evaluar(p: np.ndarray) -> dict:
        return {
            "roc_auc": float(roc_auc_score(y_test, p)),
            "average_precision": float(average_precision_score(y_test, p)),
            "brier": float(brier_score_loss(y_test, p)),
            "log_loss": float(log_loss(y_test, p, labels=[0, 1])),
        }

    metricas = {
        "filas_entrenamiento": len(y_train), "filas_test": len(y_test),
        "prevalencia_entrenamiento": prevalencia, "prevalencia_test": float(y_test.mean()),
        "modelo": evaluar(prob),
        "baseline_prevalencia": evaluar(np.full(len(y_test), prevalencia)),
        "features": list(FEATURES), "calibracion": "sigmoid, 3 folds, ensemble=False",
        "interpretacion": "Fully Paid vs Charged Off en cuentas cerradas.",
    }
    tabla = pd.DataFrame({"probabilidad": prob, "target": y_test.to_numpy()})
    tabla["tramo"] = pd.cut(tabla["probabilidad"], np.linspace(0, 1, 11), include_lowest=True)
    calibracion = tabla.groupby("tramo", observed=True).agg(
        cuentas=("target", "size"), probabilidad_media=("probabilidad", "mean"),
        frecuencia_fully_paid=("target", "mean"),
    ).reset_index()
    calibracion["tramo"] = calibracion["tramo"].astype(str)
    return modelo, metricas, calibracion


# %% [markdown]
# ## 4. Scoring y prioridad monetaria
# Nunca se imputa el saldo: hacerlo inventaría exposición financiera.
# Las cuentas con datos insuficientes pasan a revisión. La prioridad monetaria
# es un criterio experimental de orden, no el efecto causal de una llamada.

# %%
def priorizar(
    morosos: pd.DataFrame, modelo: CalibratedClassifierCV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula scores y ordena; conserva aparte las cuentas no puntuables."""
    sin_saldo = morosos["out_prncp"].isna() | morosos["out_prncp"].le(0)
    sin_features = morosos[list(FEATURES)].isna().all(axis=1)
    no_puntuable = sin_saldo | sin_features
    revision = morosos.loc[no_puntuable].copy()
    revision["motivo_revision"] = np.where(
        sin_saldo.loc[no_puntuable], "Saldo ausente, inválido o no positivo",
        "Todos los predictores están ausentes",
    )
    priorizadas = morosos.loc[~no_puntuable].copy()
    if not priorizadas.empty:
        positiva = int(np.flatnonzero(modelo.classes_ == 1)[0])
        priorizadas["prob_recobro"] = modelo.predict_proba(
            priorizadas[list(FEATURES)]
        )[:, positiva]
    else:
        priorizadas["prob_recobro"] = pd.Series(index=priorizadas.index, dtype="float64")
    priorizadas["prioridad_monetaria"] = (
        priorizadas["prob_recobro"] * priorizadas["out_prncp"].astype("float64")
    )
    priorizadas = priorizadas.sort_values(
        ["prioridad_monetaria", "indice_origen"], ascending=[False, True], kind="stable",
    ).reset_index(drop=True)
    priorizadas.insert(0, "orden_llamada", np.arange(1, len(priorizadas) + 1))
    return priorizadas, revision.reset_index(drop=True)


def ejecutar(config: Config | None = None) -> Resultado:
    """Ejecuta el pipeline; no exporta hasta llamar a exportar()."""
    config = config or Config()
    for limitacion in LIMITACIONES:
        LOGGER.warning(limitacion)
    historicos, morosos, resumen = cargar_segmentos(config)
    calidad = pd.DataFrame({
        "variable": FEATURES,
        "nulos_historicos_muestra": historicos[list(FEATURES)].isna().mean().to_numpy(),
        "nulos_morosos": morosos[list(FEATURES)].isna().mean().to_numpy(),
    })
    modelo, metricas, calibracion = entrenar(historicos, config)
    priorizadas, revision = priorizar(morosos, modelo)
    resumen.update({
        "cuentas_priorizadas": len(priorizadas), "cuentas_revision": len(revision),
        "saldo_cartera_priorizada": float(priorizadas["out_prncp"].sum()),
    })
    parametros = asdict(config)
    parametros["csv_path"] = str(config.csv_path) if config.csv_path else None
    return Resultado(
        priorizadas, revision,
        {"configuracion": parametros, "datos": resumen, "evaluacion": metricas,
         "limitaciones": LIMITACIONES},
        calibracion, calidad, modelo,
    )


# %% 5. Exportación CSV / Excel y ejecución desde terminal
def exportar(resultado: Resultado, carpeta: Path, excel: bool = False) -> None:
    """Guarda CSV UTF-8-BOM y opcionalmente Excel; sobrescribe esas salidas."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    tablas = {
        "priorizacion_cobranzas": resultado.priorizadas,
        "cuentas_revision": resultado.revision,
        "calibracion_test": resultado.calibracion,
        "calidad_variables": resultado.calidad,
    }
    for nombre, tabla in tablas.items():
        tabla.to_csv(carpeta / f"{nombre}.csv", index=False, encoding="utf-8-sig")
    (carpeta / "metricas.json").write_text(
        json.dumps(resultado.metricas, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if excel:
        if max(len(tabla) for tabla in tablas.values()) > 1_048_575:
            raise ValueError("Una tabla supera el límite de Excel. Los CSV ya se guardaron.")
        try:
            with pd.ExcelWriter(carpeta / "priorizacion_cobranzas.xlsx", engine="openpyxl") as escritor:
                for nombre, tabla in tablas.items():
                    tabla.to_excel(escritor, sheet_name=nombre, index=False, freeze_panes=(1, 0))
        except ImportError as error:
            raise ImportError("Instala openpyxl para Excel. Los CSV ya se guardaron.") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Priorización experimental de cobranza.")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=Path("outputs"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-historicos", type=int, default=200_000)
    parser.add_argument("--arboles", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--excel", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    resultado = ejecutar(Config(
        csv_path=args.csv, chunksize=args.chunksize, max_historicos=args.max_historicos,
        arboles=args.arboles, jobs=args.jobs, seed=args.seed, id_col=args.id_col,
    ))
    exportar(resultado, args.salida, excel=args.excel)
    print(json.dumps(resultado.metricas["datos"], ensure_ascii=False, indent=2))
    print(json.dumps(resultado.metricas["evaluacion"], ensure_ascii=False, indent=2))
    # Solo métricas en terminal; el detalle de cuentas permanece en el archivo.
    print(f"Archivos guardados en: {args.salida.resolve()}")


if __name__ == "__main__":
    main()
