"""Contratos temporales: no fuga del desenlace al escenario y dinero explícito."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from simulacion_cobranzas import (
    CAT_GRADE, CAT_REF, COLUMNAS_PLAN, COLUMNAS_ROLL, NUM_GRADE, NUM_REF, ConfigTemporal, cargar_cohortes,
    construir_modelo, generar_plan, preparar_fechas, saldo_contractual, simular_meses,
)


def cartera():
    return pd.DataFrame({
        "id": ["a", "b"], "issue_d": pd.to_datetime(["2014-01-01", "2014-02-01"]),
        "last_pymnt_d": pd.to_datetime(["2017-01-01", "2015-05-01"]),
        "loan_status": ["Fully Paid", "Charged Off"],
        "loan_amnt": [1000., 2000.], "int_rate": [0., 12.],
        "installment": [1000/36, 70.], "out_prncp": [0., 0.],
    })


class SimulacionTests(unittest.TestCase):
    def test_fechas_no_convierten_duracion_en_dpd(self):
        df = pd.DataFrame({"issue_d": ["Jan-2014"]*3, "last_pymnt_d": ["Jan-2015", None, "Dec-2013"]})
        limpio = preparar_fechas(df)
        self.assertEqual(limpio.loc[0, "meses_hasta_ultimo_pago"], 12)
        self.assertTrue(limpio["dias_desde_ultimo_pago_aprox"].isna().all())
        self.assertTrue(limpio.loc[1:, "meses_hasta_ultimo_pago"].isna().all())
        self.assertTrue(limpio.loc[2, "fecha_pago_anterior_otorgamiento"])
        con_corte = preparar_fechas(df, "2015-02-01")
        self.assertEqual(con_corte.loc[0, "dias_desde_ultimo_pago_aprox"], 31)
        self.assertEqual(len(limpio), 3)

    def test_saldo_tasa_cero_y_amortizacion(self):
        saldo = saldo_contractual(np.array([100., 100.]), np.array([0., .01]),
                                 np.array([10., 10.]), np.array([2, 1]))
        np.testing.assert_allclose(saldo, [80., 91.])

    def test_escenario_no_consulta_resultados_finales(self):
        original = cartera()
        alterado = original.copy()
        alterado["loan_status"] = ["Charged Off", "Fully Paid"]
        alterado["last_pymnt_d"] = pd.NaT
        alterado["out_prncp"] = 999999.
        config = ConfigTemporal(meses_colas=0)
        a = pd.concat(simular_meses(original, np.array([.2, .4]), config), ignore_index=True)
        b = pd.concat(simular_meses(alterado, np.array([.2, .4]), config), ignore_index=True)
        pd.testing.assert_frame_equal(a.drop(columns="out_prncp_observado"), b.drop(columns="out_prncp_observado"))
        self.assertTrue(a["tipo_dato"].eq("SIMULADO").all())
        self.assertTrue(a["out_prncp_observado"].eq(0).all())
        self.assertFalse(a.duplicated(["id", "fecha_corte"]).any())

    def test_bloques_y_reserva_temporal_con_nulos(self):
        with tempfile.TemporaryDirectory() as carpeta:
            path = Path(carpeta)/"accepted.csv"
            n = 120
            df = pd.DataFrame({c: np.full(n, 10.) for c in NUM_REF})
            for c in CAT_REF + ["grade"]:
                df[c] = "A"
            df["id"] = [str(i) for i in range(n)]
            df["issue_d"] = ["Jan-2012"]*40 + ["Jan-2013"]*40 + ["Jan-2014"]*40
            df["last_pymnt_d"] = "Jan-2017"
            df["loan_status"] = ["Fully Paid", "Charged Off"]*60
            df["term"] = " 36 months"
            df["int_rate"], df["installment"], df["total_pymnt"], df["out_prncp"] = 12., 1., 20., 0.
            df.loc[80:, "annual_inc"] = np.nan
            df.loc[80:, "grade"] = "CATEGORIA_NUEVA"
            df.to_csv(path, index=False)
            config = ConfigTemporal(csv_path=path, max_train=60, muestra_evaluacion=20, chunksize=31, arboles=2, jobs=1)
            train, _, test, resumen = cargar_cohortes(config)
            self.assertTrue(train["issue_d"].dt.year.between(2011, 2013).all())
            self.assertTrue(test["issue_d"].dt.year.eq(2014).all())
            self.assertEqual(resumen["n_2014"], 40)
            modelo = construir_modelo(NUM_GRADE, CAT_GRADE, config)
            modelo.fit(train[NUM_GRADE+CAT_GRADE], train["loan_status"].eq("Charged Off"))
            self.assertEqual(len(modelo.predict_proba(test[NUM_GRADE+CAT_GRADE])), 40)

    def test_plan_formula_orden_y_denominadores_roll(self):
        with tempfile.TemporaryDirectory() as carpeta:
            config = ConfigTemporal(salida=Path(carpeta), meses_colas=0)
            resumen, roll = generar_plan(cartera(), np.array([.2, .4]), config)
            plan = pd.read_csv(Path(carpeta)/"plan_llamadas_priorizado.csv")
            self.assertGreater(resumen["filas_plan_cuenta_mes"], 0)
            np.testing.assert_allclose(plan["valor_monetario_en_riesgo"], plan["probabilidad_perdida"]*plan["out_prncp"])
            for _, grupo in plan.groupby("fecha_corte"):
                self.assertTrue(grupo["valor_monetario_en_riesgo"].is_monotonic_decreasing)
                self.assertEqual(grupo["orden_llamada_mes"].tolist(), list(range(1, len(grupo)+1)))
            np.testing.assert_allclose(roll.groupby(["mes_destino", "estado_origen"])["roll_rate_cuentas"].sum(), 1.)

    def test_matriz_invalida_y_ausencia_de_datos_financieros(self):
        with self.assertRaisesRegex(ValueError, "matriz"):
            list(simular_meses(cartera(), np.array([.2, .4]), ConfigTemporal(), matriz=np.zeros((8,8))))
        invalida = cartera()
        invalida.loc[0, "loan_amnt"] = np.nan
        with self.assertRaisesRegex(ValueError, "importe"):
            list(simular_meses(invalida, np.array([.2, .4]), ConfigTemporal()))

    def test_sin_cuentas_validas_exporta_csv_legibles(self):
        with tempfile.TemporaryDirectory() as carpeta:
            config = ConfigTemporal(salida=Path(carpeta), guardar_panel=True)
            datos = cartera()
            datos["loan_amnt"] = np.inf
            resumen, roll = generar_plan(datos, np.array([.2, .4]), config)
            self.assertEqual(resumen["cuentas_revision_simulacion"], 2)
            self.assertEqual(resumen["filas_plan_cuenta_mes"], 0)
            self.assertTrue(roll.empty)
            for archivo, columnas in [
                ("plan_llamadas_priorizado.csv", COLUMNAS_PLAN),
                ("panel_mensual_simulado.csv", COLUMNAS_PLAN[1:]),
                ("roll_rates_simulados.csv", COLUMNAS_ROLL),
            ]:
                tabla = pd.read_csv(Path(carpeta)/archivo)
                self.assertTrue(tabla.empty)
                self.assertEqual(tabla.columns.tolist(), columnas)

    def test_panel_incremental_conserva_filas_orden_y_cabecera(self):
        with tempfile.TemporaryDirectory() as carpeta:
            config = ConfigTemporal(salida=Path(carpeta), guardar_panel=True, meses_colas=0)
            probabilidades = np.array([.2, .4])
            esperado = pd.concat(simular_meses(cartera(), probabilidades, config), ignore_index=True)
            generar_plan(cartera(), probabilidades, config)
            path = Path(carpeta)/"panel_mensual_simulado.csv"
            observado = pd.read_csv(path, dtype={"id": str}, parse_dates=["fecha_corte"])
            pd.testing.assert_frame_equal(observado, esperado, check_dtype=False, check_exact=False)
            self.assertEqual(path.read_bytes().count(b"\xef\xbb\xbf"), 1)

    def test_fallo_no_publica_plan_parcial(self):
        with tempfile.TemporaryDirectory() as carpeta:
            config = ConfigTemporal(salida=Path(carpeta))
            path = Path(carpeta)/"plan_llamadas_priorizado.csv"
            contenido_anterior = b"resultado anterior completo\n"
            path.write_bytes(contenido_anterior)
            primer_mes = next(simular_meses(cartera(), np.array([.2, .4]), config))

            def interrumpida(*args):
                yield primer_mes
                raise RuntimeError("fallo de simulacion")

            with patch("simulacion_cobranzas.simular_meses", side_effect=interrumpida):
                with self.assertRaisesRegex(RuntimeError, "fallo de simulacion"):
                    generar_plan(cartera(), np.array([.2, .4]), config)
            self.assertEqual(path.read_bytes(), contenido_anterior)
            self.assertFalse(any(Path(carpeta).glob(".plan-*")))


if __name__ == "__main__":
    unittest.main()
