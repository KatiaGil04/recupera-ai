"""Auditoría reproducible de CSV locales: integridad, esquema y calidad.

No modifica ni elimina los archivos de datos. Lee por bloques. Los conteos y
estadísticos agregados son exactos; no realiza imputaciones ni infiere labels.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
NUMERICAS_ACCEPTED = (
    "loan_amnt", "annual_inc", "dti", "fico_range_low", "fico_range_high",
    "delinq_2yrs", "revol_util", "revol_bal", "inq_last_6mths", "open_acc",
    "total_acc", "pub_rec", "out_prncp", "last_pymnt_amnt", "total_pymnt",
    "total_rec_prncp", "recoveries", "collection_recovery_fee", "settlement_amount",
)
CATEGORICAS = {
    "accepted": ("loan_status", "term", "grade", "home_ownership", "application_type",
                 "hardship_flag", "debt_settlement_flag", "verification_status"),
    "rejected": ("Employment Length", "State", "Policy Code"),
}
FECHAS = {
    "accepted": {"issue_d": "%b-%Y", "last_pymnt_d": "%b-%Y", "last_credit_pull_d": "%b-%Y"},
    "rejected": {"Application Date": "%Y-%m-%d"},
}


def comparar_con_gzip(csv: Path, gz: Path) -> dict:
    """SHA-256 de ambos contenidos completos; gzip también verifica su CRC."""
    def huella(archivo, comprimido=False):
        digest = hashlib.sha256()
        total = 0
        abrir = gzip.open if comprimido else open
        with abrir(archivo, "rb") as lector:
            while bloque := lector.read(16 * 1024 * 1024):
                digest.update(bloque)
                total += len(bloque)
        return {"sha256_contenido": digest.hexdigest(), "bytes_contenido": total}

    plano = huella(csv)
    resultado = {"csv": str(csv.resolve()), **plano}
    if gz.is_file():
        comprimido = huella(gz, comprimido=True)
        resultado.update({
            "gzip": str(gz.resolve()), "bytes_gzip": gz.stat().st_size,
            "gzip_descomprimido": comprimido,
            "contenidos_identicos": plano == comprimido,
        })
    else:
        resultado["gzip"] = None
    return resultado


def perfilar(csv: Path, tipo: str, salida: Path) -> dict:
    """Perfila todas las columnas y valida IDs de aceptados con índice en disco."""
    cabecera = pd.read_csv(csv, nrows=0).columns.tolist()
    nulos = pd.Series(0, index=cabecera, dtype="int64")
    categorias = {c: Counter() for c in CATEGORICAS[tipo] if c in cabecera}
    fechas = {c: {"min": None, "max": None, "invalidas": 0, "por_anio": Counter()}
              for c in FECHAS[tipo] if c in cabecera}
    numericas = (NUMERICAS_ACCEPTED if tipo == "accepted" else
                 ("Amount Requested", "Risk_Score", "Debt-To-Income Ratio"))
    estadisticos = {c: {"validos": 0, "invalidos_no_nulos": 0, "ceros": 0,
                        "negativos": 0, "suma": 0., "min": None, "max": None}
                    for c in numericas if c in cabecera}
    filas = 0
    filas_prestamos = 0
    ids_nulos = 0
    cohortes = Counter()
    castigos = Counter()
    with tempfile.TemporaryDirectory(prefix="auditoria-", dir=salida) as temporal:
        with closing(sqlite3.connect(Path(temporal) / "ids.sqlite")) as db:
            db.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
            db.execute("PRAGMA synchronous=OFF")
            with pd.read_csv(csv, dtype="string", chunksize=25_000 if tipo == "accepted" else 150_000) as lector:
                for numero, bloque in enumerate(lector, 1):
                    filas += len(bloque)
                    nulos += bloque.isna().sum()
                    for columna, contador in categorias.items():
                        contador.update(bloque[columna].str.strip().fillna("SIN_DATO").value_counts().to_dict())
                    for columna, formato in FECHAS[tipo].items():
                        if columna not in bloque:
                            continue
                        fechas_validas = pd.to_datetime(bloque[columna], format=formato, errors="coerce")
                        resumen = fechas[columna]
                        resumen["invalidas"] += int((bloque[columna].notna() & fechas_validas.isna()).sum())
                        resumen["por_anio"].update(fechas_validas.dt.year.dropna().astype(int).value_counts().to_dict())
                        if fechas_validas.notna().any():
                            minimo, maximo = fechas_validas.min().strftime("%Y-%m-%d"), fechas_validas.max().strftime("%Y-%m-%d")
                            resumen["min"] = min(resumen["min"] or minimo, minimo)
                            resumen["max"] = max(resumen["max"] or maximo, maximo)
                    for columna, resumen in estadisticos.items():
                        texto = bloque[columna].str.strip().str.removesuffix("%")
                        valores = pd.to_numeric(texto, errors="coerce").astype("float64")
                        validos = valores.loc[np.isfinite(valores)]
                        resumen["validos"] += len(validos)
                        resumen["invalidos_no_nulos"] += int((texto.notna() & ~np.isfinite(valores)).sum())
                        resumen["ceros"] += int(validos.eq(0).sum())
                        resumen["negativos"] += int(validos.lt(0).sum())
                        resumen["suma"] += float(validos.sum())
                        if len(validos):
                            minimo, maximo = float(validos.min()), float(validos.max())
                            resumen["min"] = min(resumen["min"] if resumen["min"] is not None else minimo, minimo)
                            resumen["max"] = max(resumen["max"] if resumen["max"] is not None else maximo, maximo)
                    if tipo == "accepted":
                        prestamos = bloque.loc[bloque["loan_status"].notna()]
                        filas_prestamos += len(prestamos)
                        ids = prestamos["id"].dropna()
                        ids_nulos += int(prestamos["id"].isna().sum())
                        db.executemany("INSERT OR IGNORE INTO ids VALUES (?)", ((str(x),) for x in ids))
                        agregado = bloque.groupby(["issue_d", "term", "loan_status"], dropna=True).size()
                        cohortes.update({tuple(k): int(v) for k, v in agregado.items()})
                        # Recuperación ex post: prevalencia y exposición original,
                        # sin confundirla con saldo al momento del castigo.
                        co = bloque.loc[bloque["loan_status"].eq("Charged Off")]
                        recuperado = pd.to_numeric(co["recoveries"], errors="coerce")
                        castigos.update({
                            "cuentas": len(co), "con_recuperacion_positiva": int(recuperado.gt(0).sum()),
                            "recoveries_nulos": int(recuperado.isna().sum()),
                            "recoveries_suma": float(recuperado.sum()),
                        })
                    if numero == 1 or numero % (10 if tipo == "accepted" else 20) == 0:
                        LOGGER.info("%s: %s registros revisados", tipo, f"{filas:,}")
            ids_unicos = db.execute("SELECT COUNT(*) FROM ids").fetchone()[0]
    for resumen in estadisticos.values():
        resumen["media"] = resumen.pop("suma") / resumen["validos"] if resumen["validos"] else None
    calidad = pd.DataFrame({"columna": cabecera, "nulos": nulos.values})
    calidad["porcentaje_nulos"] = calidad["nulos"] / filas * 100
    calidad.to_csv(salida / f"{tipo}_calidad_columnas.csv", index=False, encoding="utf-8-sig")
    if cohortes:
        pd.DataFrame([{"issue_d": k[0], "term": k[1], "loan_status": k[2], "cuentas": v}
                      for k, v in cohortes.items()]).to_csv(salida / "accepted_cohortes.csv", index=False)
    return {
        "filas": filas, "n_columnas": len(cabecera), "columnas": cabecera,
        "nulos_por_columna": {k: int(v) for k, v in nulos.items()},
        "categorias": {k: dict(v) for k, v in categorias.items()},
        "fechas": fechas, "numericas": estadisticos,
        **({"filas_con_estado": filas_prestamos, "filas_sin_estado": filas - filas_prestamos,
            "ids_unicos_prestamos": ids_unicos, "ids_nulos_prestamos": ids_nulos,
            "ids_duplicados_prestamos": filas_prestamos - ids_nulos - ids_unicos,
            "recuperaciones_castigados": dict(castigos)} if tipo == "accepted" else
           {"clave_identificacion": "No hay identificador de solicitud/cuenta; no se presume unicidad."}),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raiz = Path(__file__).resolve().parent
    salida = raiz / "outputs" / "auditoria"
    salida.mkdir(parents=True, exist_ok=True)
    informe = {}
    archivo_base = raiz / "data" / "archive"
    if not archivo_base.is_dir():
        archivo_base = raiz / "archive"
    for tipo in ("accepted", "rejected"):
        csv = archivo_base / f"{tipo}_2007_to_2018q4.csv" / f"{tipo}_2007_to_2018Q4.csv"
        gz = archivo_base / f"{tipo}_2007_to_2018Q4.csv.gz"
        LOGGER.info("Verificando integridad completa de %s", tipo)
        integridad = comparar_con_gzip(csv, gz)
        informe[tipo] = {"integridad": integridad}
        (salida / "auditoria.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("%s: CSV y gzip coinciden: %s", tipo, integridad.get("contenidos_identicos"))
        informe[tipo]["perfil"] = perfilar(csv, tipo, salida)
        (salida / "auditoria.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        LOGGER.info("%s: auditoría completa", tipo)
    print(f"Auditoría guardada en {salida}")


if __name__ == "__main__":
    main()
