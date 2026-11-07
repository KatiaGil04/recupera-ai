"""Roll rates observados; requiere un panel mensual, no una foto de Lending Club.

Referente: https://publishedguides.ncua.gov/examiner/Content/ExaminersGuide/AllowanceCreditLoss/Methodology.htm
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = ("CORRIENTE", "1-30", "31-60", "61-90", "91-120", "121+")
CIERRES = {"Fully Paid": "PAGADO", "Charged Off": "CASTIGADO"}
DESTINOS = (*BUCKETS, *CIERRES.values(), "SIN_OBSERVACION")


def calcular_roll_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """Calcula tasas mensuales por cuentas y saldo de origen.

    Columnas: id_cuenta, fecha_corte, dias_mora, saldo; estado_cierre opcional.
    Rechaza duplicados cuenta/mes. Los días de mora deben ser enteros >= 0
    salvo en estados terminales explícitos. El saldo debe ser finito y >= 0.
    No imputa cuentas desaparecidas como pagadas. No compara meses salteados.
    El último mes global carece de seguimiento y no actúa como origen.
    Excluye estados terminales como origen para evitar contar cierres repetidos.
    """
    requeridas = {"id_cuenta", "fecha_corte", "dias_mora", "saldo"}
    if requeridas - set(panel.columns):
        raise ValueError(f"Faltan columnas del panel: {sorted(requeridas - set(panel.columns))}")
    datos = panel.copy()
    if datos.empty:
        raise ValueError("El panel mensual está vacío.")
    if datos["id_cuenta"].isna().any() or datos["id_cuenta"].astype(str).str.strip().eq("").any():
        raise ValueError("Hay cuentas sin identificador.")
    fechas = pd.to_datetime(datos["fecha_corte"], errors="coerce")
    if fechas.isna().any():
        raise ValueError("Hay fechas de corte inválidas.")
    datos["mes"] = fechas.dt.to_period("M")
    if datos.duplicated(["id_cuenta", "mes"]).any():
        raise ValueError("Existe más de una observación por cuenta y mes.")
    if datos["mes"].nunique() < 2:
        raise ValueError("Se necesitan al menos dos meses de observaciones.")

    cierre = (
        datos["estado_cierre"].astype("string").fillna("").str.strip()
        if "estado_cierre" in datos else pd.Series("", index=datos.index, dtype="string")
    )
    if (~cierre.isin(["", *CIERRES])).any():
        raise ValueError("estado_cierre admite vacío, Fully Paid o Charged Off.")
    terminal = cierre.isin(CIERRES)
    dpd = pd.to_numeric(datos["dias_mora"], errors="coerce").astype("float64")
    dpd_valido = np.isfinite(dpd) & dpd.ge(0) & dpd.mod(1).eq(0)
    if ((~terminal) & ~dpd_valido).any():
        raise ValueError("dias_mora debe ser un entero no negativo en cuentas activas.")
    saldo = pd.to_numeric(datos["saldo"], errors="coerce").astype("float64")
    if not (np.isfinite(saldo) & saldo.ge(0)).all():
        raise ValueError("saldo debe ser finito y no negativo; no se imputa.")
    datos["saldo"] = saldo
    datos["bucket"] = pd.cut(
        dpd, [-np.inf, 0, 30, 60, 90, 120, np.inf], labels=BUCKETS,
    ).astype("string")
    datos.loc[terminal, "bucket"] = cierre.loc[terminal].map(CIERRES)

    origen = datos.loc[(datos["mes"] < datos["mes"].max()) & ~terminal,
                       ["id_cuenta", "mes", "bucket", "saldo"]].rename(
        columns={"mes": "mes_origen", "bucket": "bucket_origen", "saldo": "saldo_origen"}
    )
    if origen.empty:
        raise ValueError("No hay cuentas activas con un mes posterior observable.")
    origen["mes_destino"] = origen["mes_origen"] + 1
    destino = datos[["id_cuenta", "mes", "bucket"]].rename(
        columns={"mes": "mes_destino", "bucket": "bucket_destino"}
    )
    pares = origen.merge(destino, on=["id_cuenta", "mes_destino"], how="left", validate="one_to_one")
    pares["bucket_destino"] = pares["bucket_destino"].fillna("SIN_OBSERVACION")
    grupos = ["mes_origen", "bucket_origen", "bucket_destino"]
    agregados = pares.groupby(grupos, observed=True).agg(
        cuentas=("id_cuenta", "size"), saldo_transicion=("saldo_origen", "sum"),
    ).reset_index()
    # Materializa transiciones de frecuencia cero para que una curva no conecte
    # indebidamente dos meses omitiendo el cero intermedio.
    rejilla = pares[["mes_origen", "bucket_origen"]].drop_duplicates().merge(
        pd.DataFrame({"bucket_destino": DESTINOS}), how="cross",
    )
    tasas = rejilla.merge(agregados, on=grupos, how="left")
    tasas[["cuentas", "saldo_transicion"]] = tasas[["cuentas", "saldo_transicion"]].fillna(0)
    tasas["cuentas"] = tasas["cuentas"].astype("int64")
    por_origen = tasas.groupby(["mes_origen", "bucket_origen"], observed=True)
    tasas["cuentas_origen"] = por_origen["cuentas"].transform("sum")
    tasas["saldo_total_origen"] = por_origen["saldo_transicion"].transform("sum")
    tasas["roll_rate_cuentas"] = tasas["cuentas"] / tasas["cuentas_origen"]
    tasas["roll_rate_saldo"] = tasas["saldo_transicion"] / tasas["saldo_total_origen"].replace(0, np.nan)
    tasas["mes_destino"] = (tasas["mes_origen"] + 1).astype(str)
    tasas["mes_origen"] = tasas["mes_origen"].astype(str)
    return tasas.sort_values(grupos).reset_index(drop=True)


def graficar_roll_rates(
    tasas: pd.DataFrame, origen: str = "31-60", medida: str = "roll_rate_cuentas", ax=None,
):
    """Dibuja curvas mensuales de transición; devuelve un Axes de matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if medida not in ("roll_rate_cuentas", "roll_rate_saldo"):
        raise ValueError("medida debe ser roll_rate_cuentas o roll_rate_saldo.")
    datos = tasas.loc[tasas["bucket_origen"].eq(origen)]
    if datos.empty:
        raise ValueError(f"No hay observaciones para el bucket de origen {origen}.")
    curvas = datos.pivot(index="mes_origen", columns="bucket_destino", values=medida).sort_index()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    curvas.index = pd.to_datetime(curvas.index)
    for destino in curvas:
        if curvas[destino].fillna(0).ne(0).any():
            ax.plot(curvas.index, curvas[destino], marker="o", label=destino)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title=f"Transiciones desde {origen}", xlabel="Mes de origen", ylabel="Tasa mensual")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(title="Destino", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.figure.tight_layout()
    return ax
