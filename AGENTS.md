# Instrucciones para asistentes que trabajen en recupera-ai

Antes de modificar el proyecto, leer
[INSTRUCCIONES_PROYECTO.md](INSTRUCCIONES_PROYECTO.md). Es la guía de contexto,
reproducción, metodología y mejoras pendientes para la responsable y su IA.
Consultar también README y el código afectado; los resultados históricos se
contrastan con los JSON de la ejecución, si están disponibles.

## Contexto y ruta principal

- Entrada actual: `simulacion_cobranzas.py` y `simulacion_cobranzas.ipynb`.
- Se evalúa castigo real retrospectivo y se simula por separado un escenario
  mensual. No presentar el escenario como historia observada o eficacia causal.
- El target actual es Charged Off=1 / Fully Paid=0. El prototipo
  `priorizacion_recobro.py` usa la codificación contraria.
- Los datos y outputs no están versionados: verificar su presencia y rutas.
  `data/archive2/loan.csv` tiene IDs vacíos; no fusionarlo por posición o semejanza.

## Criterios de implementación

- Respetar el alcance solicitado por la usuaria y el código existente. Esta
  guía no exige pedir autorización adicional para cambios rutinarios autorizados.
- Mantener fuentes intactas, lectura por columnas/bloques y trazabilidad de IDs.
- En originación, excluir pagos/saldos finales, último pago, FICO actualizado,
  recuperaciones y target de los predictores. Validar disponibilidad temporal
  antes de añadir otras variables.
- Aprender imputaciones, codificaciones y calibración dentro del entrenamiento;
  no usar test para ajustar o seleccionar repetidamente el modelo.
- `last_pymnt_d` indica un mes con pago, no el inicio de mora. No convertir la
  diferencia de fechas en DPD observado.
- Mantener las marcas de simulación. En el plan mensual `out_prncp` es sintético;
  `out_prncp_observado` conserva el original. PD por capital omite LGD y no es
  pérdida esperada ni recobro incremental validados.
- Conservar una evaluación real separada de los supuestos de transición.
  Registrar cambios de parámetros, matriz, población y significado de salidas.
- Verificar la lógica modificada y actualizar documentación/notebook si cambia
  el uso. No reentrenar ni repetir auditorías completas solo por editar textos.

Comando de pruebas desde la raíz, con el intérprete del entorno instalado:

```powershell
python -m unittest discover -s tests -v
```
