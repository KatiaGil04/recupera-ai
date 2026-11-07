"""Verifica integridad de copias y calidad sin tocar los datos del usuario."""
import gzip
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from auditar_datos import comparar_con_gzip, perfilar


class AuditoriaTests(unittest.TestCase):
    def test_comparacion_completa_detecta_cambio_al_final(self):
        with tempfile.TemporaryDirectory() as carpeta:
            csv = Path(carpeta) / "fuente.csv"
            gz = Path(carpeta) / "fuente.csv.gz"
            contenido = b"id,valor\n1,10\n2,20\n"
            csv.write_bytes(contenido)
            with gzip.open(gz, "wb") as salida:
                salida.write(contenido)
            self.assertTrue(comparar_con_gzip(csv, gz)["contenidos_identicos"])
            csv.write_bytes(contenido[:-3] + b"30\n")
            self.assertFalse(comparar_con_gzip(csv, gz)["contenidos_identicos"])

    def test_perfil_separa_notas_y_no_presume_unicidad(self):
        with tempfile.TemporaryDirectory() as carpeta:
            salida = Path(carpeta)
            csv = salida / "accepted.csv"
            pd.DataFrame({
                "id": ["1", "2", "2", "nota final"],
                "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", None],
                "issue_d": ["Jan-2014", "Feb-2014", "Feb-2014", None],
                "term": ["36 months", "36 months", "36 months", None],
                "recoveries": [0., 10., 0., None],
                "dti": ["10%", "inf", "-1", None],
            }).to_csv(csv, index=False)
            perfil = perfilar(csv, "accepted", salida)
            self.assertEqual(perfil["filas"], 4)
            self.assertEqual(perfil["filas_con_estado"], 3)
            self.assertEqual(perfil["filas_sin_estado"], 1)
            self.assertEqual(perfil["ids_duplicados_prestamos"], 1)
            self.assertEqual(perfil["numericas"]["dti"]["invalidos_no_nulos"], 1)
            self.assertEqual(perfil["numericas"]["dti"]["negativos"], 1)
            self.assertEqual(perfil["recuperaciones_castigados"]["con_recuperacion_positiva"], 1)
            # TemporaryDirectory puede cerrarse también en Windows: la conexión
            # SQLite debe haberse cerrado antes de eliminar su directorio.
            self.assertFalse(any(salida.glob("auditoria-*")))


if __name__ == "__main__":
    unittest.main()
