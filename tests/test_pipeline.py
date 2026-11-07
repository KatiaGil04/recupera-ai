"""Pruebas de contratos financieros y de segmentación, con datos sintéticos."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from priorizacion_recobro import (
    COLUMNAS, FEATURES, Config, cargar_segmentos, ejecutar, exportar, limpiar_bloque,
    localizar_csv, priorizar,
)
from roll_rates import calcular_roll_rates


def fixture_csv(path):
    rng = np.random.default_rng(14)
    n = 300
    datos = pd.DataFrame({
        "id": [f"cuenta-{i}" for i in range(n)],
        "loan_status": ["Fully Paid", "Charged Off"] * (n // 2),
        "delinq_2yrs": rng.integers(0, 5, n),
        "revol_util": rng.uniform(0, 100, n),
        "annual_inc": rng.uniform(20000, 100000, n),
        "last_pymnt_amnt": rng.uniform(10, 1000, n),
        "out_prncp": np.zeros(n),
        "columna_ignorada": "texto voluminoso",
    })
    datos.loc[0, "annual_inc"] = np.nan
    mora = datos.iloc[:7].copy()
    mora["id"] = [f"mora-{i}" for i in range(7)]
    mora["loan_status"] = ["Late (31-120 days)", "In Grace Period"] * 3 + ["Current"]
    mora["out_prncp"] = [100, 1000, np.nan, -5, 0, 500, 100]
    mora.loc[mora.index[5], list(FEATURES)] = np.nan
    pd.concat([datos, mora], ignore_index=True).to_csv(path, index=False)


class PipelineTests(unittest.TestCase):
    def test_ruta_archive_prefiere_csv_extraido(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta)
            extraido = raiz / "archive" / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv"
            extraido.parent.mkdir(parents=True)
            extraido.write_text("loan_status\n", encoding="utf-8")
            (raiz / "archive" / "accepted_2007_to_2018Q4.csv.gz").write_bytes(b"respaldo")
            with patch("priorizacion_recobro.Path.cwd", return_value=raiz):
                self.assertEqual(localizar_csv(), extraido.resolve())

    def test_limpieza_porcentaje_y_saldo_sin_imputar(self):
        datos = pd.DataFrame({c: ["1", "2"] for c in COLUMNAS})
        datos["revol_util"] = ["29.7%", "125.5"]
        datos["out_prncp"] = ["no numerico", "-10"]
        datos["annual_inc"] = ["inf", "40000"]
        limpio = limpiar_bloque(datos)
        self.assertEqual(limpio["revol_util"].tolist(), [29.7, 125.5])
        self.assertTrue(limpio["out_prncp"].isna().all())
        self.assertTrue(pd.isna(limpio.loc[0, "annual_inc"]))

    def test_muestreo_reproducible_independiente_del_bloque(self):
        with tempfile.TemporaryDirectory() as carpeta:
            path = Path(carpeta) / "fixture.csv.gz"
            fixture_csv(path)
            a, mora, resumen = cargar_segmentos(Config(csv_path=path, chunksize=37, max_historicos=120))
            b, _, _ = cargar_segmentos(Config(csv_path=path, chunksize=80, max_historicos=120))
            pd.testing.assert_frame_equal(a, b)
            self.assertEqual(len(a), 120)
            self.assertEqual(len(mora), 6)
            self.assertEqual(resumen["historicos_disponibles"], 300)
            self.assertNotIn("columna_ignorada", mora.columns)
            self.assertNotIn("id", mora.columns)
            self.assertEqual(set(a.columns), {*FEATURES, "target"})

    def test_pipeline_exportacion_y_probabilidades(self):
        with tempfile.TemporaryDirectory() as carpeta:
            path = Path(carpeta) / "fixture.csv"
            fixture_csv(path)
            resultado = ejecutar(Config(csv_path=path, chunksize=71, arboles=5, jobs=1, id_col="id"))
            priorizadas = resultado.priorizadas
            self.assertEqual(len(priorizadas), 2)
            self.assertEqual(len(resultado.revision), 4)
            self.assertTrue(priorizadas["prob_recobro"].between(0, 1).all())
            self.assertTrue(priorizadas["prioridad_monetaria"].is_monotonic_decreasing)
            np.testing.assert_allclose(
                priorizadas["prioridad_monetaria"], priorizadas["out_prncp"] * priorizadas["prob_recobro"],
            )
            self.assertEqual(set(priorizadas["id"]), {"mora-0", "mora-1"})
            self.assertEqual(resultado.modelo.feature_names_in_.tolist(), list(FEATURES))
            self.assertEqual(resultado.metricas["evaluacion"]["filas_test"], 60)
            exportar(resultado, Path(carpeta) / "salidas")
            met = json.loads((Path(carpeta) / "salidas" / "metricas.json").read_text(encoding="utf-8"))
            self.assertEqual(met["datos"]["cuentas_priorizadas"], 2)
            self.assertEqual(len(pd.read_csv(Path(carpeta) / "salidas" / "priorizacion_cobranzas.csv")), 2)

    def test_ranking_monetario_no_es_ranking_probabilidad(self):
        class Modelo:
            classes_ = np.array([0, 1])

            def predict_proba(self, X):
                return np.array([[0.1, 0.9], [0.5, 0.5]])

        datos = pd.DataFrame({c: [1., 2.] for c in COLUMNAS})
        datos["out_prncp"] = [100., 1000.]
        datos["indice_origen"] = [0, 1]
        ranking, _ = priorizar(datos, Modelo())
        self.assertEqual(ranking["indice_origen"].tolist(), [1, 0])
        self.assertEqual(ranking["prioridad_monetaria"].tolist(), [500., 90.])

    def test_sin_morosos_devuelve_tabla_vacia(self):
        with tempfile.TemporaryDirectory() as carpeta:
            path = Path(carpeta) / "fixture.csv"
            fixture_csv(path)
            datos = pd.read_csv(path).iloc[:300]
            datos.to_csv(path, index=False)
            resultado = ejecutar(Config(csv_path=path, arboles=2, jobs=1))
            self.assertTrue(resultado.priorizadas.empty)
            self.assertIn("prioridad_monetaria", resultado.priorizadas)

    def test_columnas_faltantes_y_una_sola_clase(self):
        with tempfile.TemporaryDirectory() as carpeta:
            path = Path(carpeta) / "fixture.csv"
            pd.DataFrame({"loan_status": ["Fully Paid"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Faltan columnas"):
                cargar_segmentos(Config(csv_path=path))
            fixture_csv(path)
            datos = pd.read_csv(path)
            datos = datos.loc[datos["loan_status"].ne("Charged Off")]
            datos.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "por clase"):
                ejecutar(Config(csv_path=path, arboles=2, jobs=1))


class RollRateTests(unittest.TestCase):
    def panel(self):
        return pd.DataFrame({
            "id_cuenta": ["A", "A", "B", "B", "C", "C"],
            "fecha_corte": ["2025-01-31", "2025-02-28", "2025-01-31", "2025-02-28", "2025-01-31", "2025-03-31"],
            "dias_mora": [40, 0, 40, 70, 40, 0],
            "saldo": [100., 0., 300., 299., 600., 0.],
            "estado_cierre": ["", "Fully Paid", "", "", "", "Fully Paid"],
        })

    def test_saldos_origen_cierres_y_seguimiento_incompleto(self):
        tasas = calcular_roll_rates(self.panel())
        enero = tasas.loc[(tasas["mes_origen"] == "2025-01") & (tasas["bucket_origen"] == "31-60")].set_index("bucket_destino")
        self.assertAlmostEqual(enero.loc["PAGADO", "roll_rate_cuentas"], 1 / 3)
        self.assertAlmostEqual(enero.loc["PAGADO", "roll_rate_saldo"], .1)
        self.assertAlmostEqual(enero.loc["61-90", "roll_rate_saldo"], .3)
        self.assertAlmostEqual(enero.loc["SIN_OBSERVACION", "roll_rate_saldo"], .6)
        self.assertAlmostEqual(enero["roll_rate_cuentas"].sum(), 1.)
        self.assertAlmostEqual(enero["roll_rate_saldo"].sum(), 1.)
        self.assertNotIn("2025-03", tasas["mes_origen"].unique())
        self.assertNotIn("PAGADO", tasas["bucket_origen"].unique())
        self.assertEqual(enero.loc["CORRIENTE", "cuentas"], 0)

    def test_duplicado_por_mes_es_error(self):
        panel = self.panel()
        panel = pd.concat([panel, panel.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "por cuenta y mes"):
            calcular_roll_rates(panel)

    def test_saldo_cero_no_inventa_tasa_ponderada(self):
        panel = self.panel()
        panel["saldo"] = 0.
        tasas = calcular_roll_rates(panel)
        self.assertTrue(tasas["roll_rate_saldo"].isna().all())
        self.assertTrue(tasas["roll_rate_cuentas"].notna().all())

    def test_saldo_invalido_y_dpd_fraccionario(self):
        panel = self.panel()
        panel.loc[0, "saldo"] = -1.
        with self.assertRaisesRegex(ValueError, "saldo"):
            calcular_roll_rates(panel)
        panel = self.panel().astype({"dias_mora": "float64"})
        panel.loc[0, "dias_mora"] = 30.5
        with self.assertRaisesRegex(ValueError, "dias_mora"):
            calcular_roll_rates(panel)


if __name__ == "__main__":
    unittest.main()
