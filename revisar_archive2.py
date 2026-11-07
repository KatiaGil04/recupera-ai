"""Inspección completa de la fuente extra; no concatena ni une datos sin clave."""
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd

from priorizacion_recobro import localizar_csv


def revisar():
    raiz = Path(__file__).resolve().parent
    extra = raiz / "data" / "archive2" / "loan.csv"
    accepted = localizar_csv()
    columnas = pd.read_csv(extra, nrows=0).columns.tolist()
    originales = pd.read_csv(accepted, nrows=0).columns.tolist()
    seleccion = [c for c in ["id", "member_id", "issue_d", "last_pymnt_d", "loan_status", "loan_amnt", "term"] if c in columnas]
    nulos = Counter()
    estados = Counter()
    filas = 0
    fechas = {}
    for bloque in pd.read_csv(extra, usecols=seleccion, dtype="string", chunksize=150_000):
        filas += len(bloque)
        nulos.update(bloque.isna().sum().to_dict())
        estados.update(bloque["loan_status"].fillna("SIN_ESTADO").value_counts().to_dict())
        for campo in ["issue_d", "last_pymnt_d"]:
            fecha = pd.to_datetime(bloque[campo], format="%b-%Y", errors="coerce")
            if fecha.notna().any():
                menor, mayor = fecha.min().isoformat(), fecha.max().isoformat()
                previo = fechas.get(campo, {"min": menor, "max": mayor})
                fechas[campo] = {"min": min(previo["min"], menor), "max": max(previo["max"], mayor)}
    # Lectura XML del diccionario: solo consulta, sin modificar el libro.
    definiciones = {}
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(raiz / "data" / "archive2" / "LCDataDictionary.xlsx") as z:
        cadenas = ["".join(si.itertext()) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", ns)]
        for archivo in z.namelist():
            if not (archivo.startswith("xl/worksheets/sheet") and archivo.endswith(".xml")):
                continue
            for fila in ET.fromstring(z.read(archivo)).findall(".//s:row", ns):
                valores = []
                for celda in fila.findall("s:c", ns):
                    valor = celda.find("s:v", ns)
                    if valor is not None:
                        valores.append(cadenas[int(valor.text)] if celda.attrib.get("t") == "s" else valor.text)
                    else:
                        valores.append("".join(celda.itertext()))
                if len(valores) > 1 and valores[0] in ["issue_d", "last_pymnt_d", "total_pymnt", "out_prncp", "grade", "recoveries", "loan_status"]:
                    definiciones[valores[0]] = valores[1]
    informe = {
        "fuente": str(extra), "filas": filas, "columnas": len(columnas),
        "campos_ausentes_frente_accepted": sorted(set(originales) - set(columnas)),
        "campos_adicionales": sorted(set(columnas) - set(originales)),
        "nulos": dict(nulos), "estados": dict(estados), "fechas": fechas,
        "definiciones_diccionario_local": definiciones,
        "decision": "No fusionar sin clave válida y procedencia temporal; semejanza de columnas no acredita un panel.",
    }
    salida = raiz / "outputs" / "simulacion"
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "revision_archive2.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(informe, ensure_ascii=False, indent=2))
    return informe


if __name__ == "__main__":
    revisar()
