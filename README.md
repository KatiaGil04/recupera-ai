# recupera-ai

Si recibes el proyecto por primera vez, empieza por
[INSTRUCCIONES_PROYECTO.md](INSTRUCCIONES_PROYECTO.md): instalación, reproducción,
contexto metodológico y pautas para tu asistente de IA. Los asistentes que
trabajen en el código deben consultar también [AGENTS.md](AGENTS.md).

Para el escenario mensual solicitado, abre `simulacion_cobranzas.ipynb` o ejecuta
`python simulacion_cobranzas.py`. El flujo conserva una evaluación con datos
reales y genera por separado un escenario explícitamente sintético.

Priorización experimental de cobranza con Lending Club: lectura por bloques,
Random Forest calibrado y orden por `prob_recobro * out_prncp`.

## Ejecutar

Requiere Python 3.10 o posterior:

```powershell
python -m pip install -r requirements.txt
python priorizacion_recobro.py
```

Si `python` no está en el PATH, en este equipo se verificó con:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" priorizacion_recobro.py --id-col id --jobs 2
```

Busca `accepted_2007_to_2018Q4.csv` o `.csv.gz` en la carpeta actual, `data/`,
`data/archive/`, `archive/` del proyecto y `~/Downloads/archive/`. También reconoce las carpetas
creadas al extraer el gzip. En este proyecto se encuentran en:

```text
data/archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv
data/archive/rejected_2007_to_2018q4.csv/rejected_2007_to_2018Q4.csv
```

Prefiere el CSV descomprimido frente al gzip de la misma carpeta base.

Para conservar el identificador de préstamo y generar también Excel:

```powershell
python priorizacion_recobro.py --csv "data\archive\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv" --id-col id --excel
```

La carga predeterminada selecciona exactamente las seis columnas solicitadas.
`--id-col id` agrega solo esa séptima columna. Sin ella, `indice_origen` permite
rastrear el registro de datos según el lector CSV, pero no identifica al cliente
para llamar. Con ella, hay que cruzar la lista con el CRM para obtener contactos.

Para reducir memoria y tiempo:

```powershell
python priorizacion_recobro.py --chunksize 50000 --max-historicos 100000 --arboles 100 --jobs 2
```

Se recorre todo el archivo y se conserva una muestra aleatoria uniforme de hasta
200 000 cuentas cerradas por defecto. No se limita a las primeras filas ni se
balancean clases artificialmente. Se conservan todos los morosos. El consumo de
RAM depende del bloque, muestra, cartera por puntuar y árboles; no hay un límite
rígido de memoria. Los importes se mantienen en `float64`.

## Jupyter

Abre `priorizacion_recobro.ipynb` y ejecuta las celdas en orden.

El script contiene celdas `# %%` y explicaciones `# %% [markdown]` para VS Code
y Jupytext. También se puede importar desde un notebook:

```python
from pathlib import Path
from priorizacion_recobro import Config, ejecutar, exportar

resultado = ejecutar(Config(
    csv_path=Path("data/archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"),
    id_col="id",  # opcional; omitir para leer solo las seis columnas
))
df_priorizado = resultado.priorizadas
display(df_priorizado.head(20))
exportar(resultado, Path("outputs"), excel=False)
```

## Tratamiento y evaluación

1. Entrena con `Fully Paid` (1) y `Charged Off` (0). Puntúa únicamente
   `Late (31-120 days)` e `In Grace Period`. Los demás estados se contabilizan,
   pero no se entrenan ni puntúan. Puedes agregar `Late (16-30 days)` a
   `ESTADOS_MORA` si deseas ampliar la selección solicitada.
2. Usa como predictores `delinq_2yrs`, `revol_util` y `annual_inc`.
   `last_pymnt_amnt` puede contener información del desenlace y `out_prncp` al
   cierre no representa exposición al inicio de la mora. Se excluyen del modelo
   para evitar fuga de información; se conservan para consulta y priorización.
3. Números inválidos, infinitos y negativos pasan a nulos. `revol_util` admite
   `29.7` y `29.7%`, en puntos porcentuales; valores mayores que 100 no se
   recortan automáticamente. La imputación por mediana se ajusta dentro del
   pipeline, usando solo entrenamiento, con indicadores de ausencia.
4. Reserva un test estratificado del 20 %. En el 80 % restante calibra con tres
   particiones y método sigmoide. Reporta ROC-AUC, average precision, Brier y
   log loss frente a un baseline de prevalencia, más tabla de calibración.
5. No imputa el saldo: cuentas con saldo nulo, inválido, negativo o cero pasan a
   revisión. También se revisan las cuentas sin ningún predictor válido.

## Interpretación y limitaciones

`prob_recobro` es un **proxy de P(Fully Paid | variables de originación, cuenta
cerrada)** trasladado a morosos. `Fully Paid` no acredita mora previa ni pago
provocado por una llamada. Las métricas validan desenlaces de cerrados; la
partición aleatoria no valida transferencia a morosos ni rendimiento futuro.

Los estados reflejan el corte histórico del CSV, **no la cartera de hoy**. No se
inventa una fecha de corte a partir del nombre 2007–2018: existen pagos
posteriores al período de originación.

`prioridad_monetaria` mantiene la unidad del saldo (USD en Lending Club). Es un
criterio experimental de ordenamiento, no dinero recuperable validado. No
considera recuperaciones parciales, costos, contactabilidad, horizonte de cobro
ni efecto causal de la gestión.

Para un modelo operativo de cura se necesitan observaciones al inicio de la
mora, un horizonte (por ejemplo, volver a corriente en 90 días), información
disponible entonces y validación por cohortes futuras. Para optimizar llamadas
se necesitan gestión, contactos, pagos y un piloto con control que mida recobro
incremental.

## Archivos de salida

En `outputs/`:

- `priorizacion_cobranzas.csv`: `orden_llamada`, identificador opcional,
  `indice_origen`, seis columnas limpias, `prob_recobro` y `prioridad_monetaria`.
- `cuentas_revision.csv`: morosos que requieren corrección de datos.
- `metricas.json`: configuración, conteos, evaluación y limitaciones.
- `calibracion_test.csv`: tasa observada y probabilidad media por tramo.
- `calidad_variables.csv`: proporción de nulos en históricos y morosos.
- `priorizacion_cobranzas.xlsx`: opcional con `--excel`; hojas de prioridad,
  revisión, calibración y calidad. Requiere `openpyxl`.

CSV en UTF-8 con BOM para Excel. Reejecutar sobrescribe las salidas del mismo
directorio. No modifica el CSV fuente ni copia el histórico completo.

## Roll rates

El archivo de aceptados ofrece un estado por préstamo. No permite reconstruir
transiciones mensuales, días exactos de mora ni curaciones. `delinq_2yrs` no
sustituye ese historial y `Late (31-120 days)` no identifica buckets precisos.

`roll_rates.py` está preparado para un panel mensual real con `id_cuenta`,
`fecha_corte`, `dias_mora`, `saldo` y `estado_cierre` opcional
(`Fully Paid` / `Charged Off`). Acepta una fila por cuenta y mes:

```python
from roll_rates import calcular_roll_rates, graficar_roll_rates

# panel = ...  # historial mensual real, no el CSV de aceptados
tasas = calcular_roll_rates(panel)
display(tasas)
ax = graficar_roll_rates(tasas, origen="31-60", medida="roll_rate_cuentas")
```

Compara meses consecutivos de la misma cuenta; pondera por saldo de origen.
Las cuentas sin registro al mes siguiente se muestran como `SIN_OBSERVACION`,
sin suponer pago. El último mes global no tiene seguimiento y se excluye de
denominadores. Duplicados cuenta/mes y datos inválidos producen errores.
Las curvas son tasas mensuales observadas, no proyecciones.

Referencias: [calibración de scikit-learn](https://scikit-learn.org/stable/modules/calibration.html),
[fuga de información](https://scikit-learn.org/stable/common_pitfalls.html) y
[roll rates en NCUA](https://publishedguides.ncua.gov/examiner/Content/ExaminersGuide/AllowanceCreditLoss/Methodology.htm).

## Resultado de la verificación local

Se ejecutó el CSV completo con semilla 42, muestra de 200 000 históricos,
120 árboles, 2 procesos de trabajo e identificador `id`, usando Python 3.13.5,
pandas 2.3.3, numpy 2.3.3 y scikit-learn 1.7.2.

- 2 260 701 registros leídos; 1 345 310 cerrados elegibles.
- 29 903 cuentas en los estados de mora seleccionados.
- 29 770 priorizadas y 133 enviadas a revisión.
- Test de 40 000 cerrados: ROC-AUC **0,5619**, Brier 0,15814 frente a 0,15931
  del baseline. La mejora predictiva es pequeña; el ranking no está validado
  para una campaña real de cobranza.

Las salidas están en `outputs/`, excluido de Git. La ejecución verificada generó
CSV y JSON; la exportación Excel es opcional y requiere instalar `openpyxl`.

## Comprobaciones

```powershell
python -m unittest discover -s tests -v
```

Las pruebas verifican segmentación, limpieza, conservación de saldos, ranking,
probabilidades, exclusión de predictores con fuga, exportación y transiciones
mensuales con seguimiento incompleto.

## Auditoría y objetivo alternativo de predicción

`auditar_datos.py` revisa ambos CSV completos por bloques y compara su contenido
con el gzip mediante SHA-256 y tamaño descomprimido. No elimina archivos.
Guarda perfiles y evidencia en `outputs/auditoria/`:

```powershell
python auditar_datos.py
python evaluar_objetivo.py
```

Resultado de la auditoría completa:

| Archivo | Registros de datos | Columnas | Hallazgos |
| --- | ---: | ---: | --- |
| accepted | 2 260 668 préstamos con estado | 151 | IDs completos y únicos; otras 33 filas sin estado |
| rejected | 27 648 741 solicitudes | 9 | Sin ID de cuenta ni resultado de pagos |

Ambos CSV coinciden exactamente con el contenido descomprimido de su `.gz`,
incluidos SHA-256 y longitud. Puedes eliminar **los dos `.gz` de `archive/`**
manteniendo sus CSV: recuperarás 648 053 013 bytes (aproximadamente 648 MB).
La auditoría no borró esos archivos. La evidencia se conserva en
`outputs/auditoria/auditoria.json`.

En aceptados, las originaciones van de junio de 2007 a diciembre de 2018;
hay últimos pagos hasta marzo de 2019 y consultas de crédito hasta abril de
2019. Eso no establece una fecha común exacta de corte para todos los préstamos.
FICO inicial, importe e ingreso están casi completos; antigüedad laboral tiene
6,5 % de nulos. Deben revisarse extremos, como ingreso anual de 110 millones,
DTI hasta 999 y dos DTI negativos, sin eliminar casos solo por un umbral arbitrario.
Los campos de acuerdos y hardship tienen ausencia estructural elevada.

En rechazados falta `Risk_Score` al **66,9 %**; su rango observado es 0–990.
DTI contiene 1 203 063 valores negativos y un máximo de 50 000 031,49 puntos
porcentuales, que requieren interpretación del diccionario y limpieza antes
de modelar. Sin ID no se presume que todas las solicitudes sean de personas
distintas y no se deduplican por coincidencia de atributos.

`evaluar_objetivo.py` es un experimento separado; no reemplaza el prototipo de
cobranza. Evalúa **castigo histórico en préstamos otorgados**, con target
`Charged Off=1` y `Fully Paid=0` (inverso al target del recobro).
Usa plazo de 36 meses, entrenamiento 2011–2013 y prueba 2014. Las muestras son
uniformes, de 100 000 y 50 000 cuentas, respectivamente, con semilla 42.
En estas cohortes todos los estados observados fueron Fully Paid o Charged Off;
los conteos completos quedan en `viabilidad_prediccion.json`.

| Predictores en el mismo experimento temporal | ROC-AUC | Average precision |
| --- | ---: | ---: |
| Tres variables del prototipo | 0,5876 | 0,1784 |
| 16 variables de originación | 0,6484 | 0,2166 |
| Baseline de prevalencia | 0,5000 | 0,1404 |

El conjunto ampliado usa FICO inicial, importe, ingreso, DTI, utilización y saldo
revolvente, morosidad previa, consultas, cuentas, registros públicos,
bancarrotas, vivienda, finalidad, antigüedad laboral y verificación de ingreso.
No usa FICO actualizado, pagos, recuperaciones, saldo pendiente ni acuerdos como
predictores. Tampoco usa grade/sub_grade/int_rate, que contienen decisiones del
modelo original de Lending Club.

La mejora es moderada. No se ajustaron hiperparámetros ni se calibraron estos
scores exploratorios. No es una PD a 12 meses ni evidencia de desempeño en
clientes actuales o rechazados. Estos AUC no deben compararse directamente con
el 0,5619 del prototipo: cambian cohortes y partición.

`accepted` tiene 151 columnas y permite estudiar castigo y pérdida histórica.
`rejected` tiene solo 9 columnas de solicitud, sin desenlace de pago. Apoya un
análisis de selección histórica; **rechazo no equivale a incumplimiento**.
No existe una clave para unir sus filas a préstamos de aceptados.
`Application Date` e `issue_d` representan eventos distintos y no se debe
asumir que `Risk_Score` sea comparable con FICO sin el diccionario original.

Recomendación: orientar el proyecto académico hacia **predicción de castigo
histórico y análisis de pérdida en créditos otorgados**. Puede estudiarse
`recoveries` en castigados, pero `1 - recoveries/loan_amnt` no es una LGD
validada: faltan exposición y fecha al default y una ventana comparable de
recuperación. Recuperaciones, pagos y acuerdos son resultados posteriores,
no predictores válidos en originación.

Para call center y roll rates se necesita otro conjunto: identificador de
cuenta, cortes mensuales con días de atraso y saldo, pagos fechados, fechas de
cierre/castigo y gestiones con resultado y costo. El efecto incremental de
llamar requiere además una política base y un piloto con control.

El sesgo de selección se explica en
[Calabrese, Osmetti y Zanin (2024)](https://www.pure.ed.ac.uk/ws/portalfiles/portal/471998676/CalabreseEtal2024SEPSSampleSelectionBias.pdf).

## Escenario mensual y nueva fuente archive2

Se revisó por completo `data/archive2/loan.csv`: 2 260 668 filas y 145 columnas,
sin columnas adicionales frente a accepted. Todos los `id` y `member_id` están
vacíos. Faltan las seis columnas FICO presentes en accepted. Los estados difieren
y el último mes con pagos llega a febrero de 2019, frente a marzo en accepted.
Parece una versión de otro corte del mismo universo; sin clave ni fecha exacta
de extracción verificable, no se fusiona ni se concatenan préstamos que podrían
duplicarse. No aporta por sí sola un panel mensual. La revisión reproducible
está en `revisar_archive2.py` y `outputs/simulacion/revision_archive2.json`.

El diccionario `LCDataDictionary.xlsx` confirma que `last_pymnt_d` es el último
**mes** con pago, `total_pymnt` son pagos acumulados a la fecha y `out_prncp` es
capital pendiente al corte. No hay fecha de primera mora ni calendario de pagos.

### Ejecutar la simulación

```powershell
python simulacion_cobranzas.py
# Si python no está en PATH:
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" simulacion_cobranzas.py
```

En Jupyter/VS Code, abre `simulacion_cobranzas.ipynb` o importa:

```python
from simulacion_cobranzas import ConfigTemporal, ejecutar

resultado = ejecutar(ConfigTemporal(jobs=2))
display(resultado["figura_roc"])
display(resultado["figura_roll"])
```

### Qué calcula y qué supone

1. Convierte issue_d y last_pymnt_d a fechas, marca valores inválidos y conserva
   nulos. `meses_hasta_ultimo_pago` es la diferencia de meses entre ambas fechas.
   Su alias `meses_activos_antes_del_atraso_aprox` es descriptivo: **no prueba
   continuidad de pagos ni identifica el inicio del atraso**. Se calcula también
   para Fully Paid, que no acredita haber pasado por mora. Ninguno entra al modelo.
2. `dias_desde_ultimo_pago_aprox` queda nulo salvo que se indique explícitamente
   `--fecha-corte-observada YYYY-MM-DD`. Incluso entonces es tiempo desde un mes
   de pago aproximado al primer día, no DPD. No se usa la fecha de ejecución para
   evaluar préstamos históricos ni se deduce un corte común del máximo de fechas.
3. Entrena con cuentas cerradas otorgadas en 2011–2013 y evalúa 2014, con plazo
   normalizado a `36 months`. La muestra de train es uniforme, de hasta 100 000;
   conserva **todas** las cuentas de 2014 para scoring y el plan, sin descartar
   nulos en predictores. Una muestra de 50 000 cerrados reproduce la comparación
   anterior. Los no cerrados, si existen, reciben score sin inventarles target.
4. Compara referencia_16 (las 16 variables del experimento 0,6484) y con_grade_16
   (reemplaza pub_rec_bankruptcies por grade). One-Hot Encoding admite categorías
   nuevas y la mediana se aprende solo en train. Grade es la clasificación de LC
   en originación, no una variable macroeconómica ni una evaluación independiente
   de su política. No hay inflación, empleo agregado ni tasas macro en estos CSV.
5. Total_pymnt, out_prncp y fechas de último pago se excluyen de predictores: son
   posteriores al otorgamiento. Out_prncp sí se conserva para la comparación de
   exposición observada. El AUC se mide con desenlaces reales y no se fuerza a
   coincidir con 0,648. No se calibran los scores en este experimento.
6. El escenario comienza el mes posterior a cada otorgamiento de 2014. Sortea
   estados mensuales con una matriz **supuesta**, editable en MATRIZ_SUPUESTA,
   y semilla 42. No consulta el desenlace real, último pago ni pagos acumulados.
   Los estados PAGADO/CASTIGADO son absorbentes. Los roll rates resultan de esas
   hipótesis y se verifican por cuentas y por saldo de origen, sin validarlos
   contra una historia que no existe.
7. El saldo sintético parte de loan_amnt y usa tasa/cuota contractuales. No amortiza
   capital en meses de mora; al curar supone regularización del capital vencido
   hasta la curva contractual. Al llegar a 36 meses estando corriente supone
   liquidación; PAGADO lleva saldo cero y CASTIGADO sale del plan ordinario.
   Omite intereses de mora, comisiones y recuperaciones. Las filas sin importe,
   tasa o cuota válidos pasan a revisión, conservando su score en la cohorte.
8. El plan se ordena por mes y `probabilidad_perdida * out_prncp`, descendente,
   con posición por mes. **En ese archivo out_prncp es el saldo sintético** y
   `out_prncp_observado` conserva el del CSV. Se incluye `tipo_dato=SIMULADO` y
   `origen_saldo=AMORTIZACION_SUPUESTA`. La PD queda fija en originación; cambian
   elegibilidad y exposición. No estima el efecto incremental de llamadas.

La separación por años es una evaluación retrospectiva por cohortes, no un
backtest de un modelo disponible en enero de 2014: desenlaces de préstamos de
2013 pudieron conocerse más tarde. Tampoco es PD a 12 meses. El producto por
saldo omite LGD y no debe presentarse como pérdida esperada validada.

### Salidas en outputs/simulacion

- `plan_llamadas_priorizado.csv`: lista mensual sintética de cuentas en mora;
  excluye PAGADO y CASTIGADO de la gestión ordinaria.
- `cohorte_2014_observada.csv`: todas las cuentas 2014, fechas, duración aproximada,
  pagos/saldos observados y scores. No se inventa exposición al inicio de mora.
- `curva_roc.png`: comparación sobre la misma muestra real de cerrados de 2014.
- `roll_rates_simulados.csv` y `curvas_roll_rate_simuladas.png`: tasas y curvas
  del escenario, nunca tasas históricas reconstruidas.
- `matriz_transicion_supuesta.csv`: probabilidades usadas por el simulador.
- `metricas_temporales.json`: resultados, conteos, hipótesis y limitaciones.
- `cuentas_revision_simulacion.csv`: datos financieros insuficientes para simular.

`--guardar-panel` exporta opcionalmente todos los registros mensuales del
escenario, incluidos cierres en su mes de ocurrencia. Puede ocupar mucho disco.
La lectura, simulación y exportación del panel se realizan por bloques/meses.
Reejecutar sobrescribe las salidas del mismo nombre; no modifica fuentes.

### Resultado de la ejecución mensual verificada

- Se puntuaron las 162 570 cuentas de 2014/36 meses; no hubo exclusiones por nulos
  en los predictores ni datos financieros insuficientes para el escenario.
- Sobre la misma muestra de 50 000 cuentas de 2014: referencia AUC **0,648428**;
  variante con grade AUC **0,666804**. En las 162 570 cuentas, la variante obtuvo
  AUC **0,669640**. Los scores siguen sin calibrar.
- Todas las cuentas de esa cohorte tienen out_prncp observado igual a cero;
  usar ese saldo literalmente daría prioridades monetarias cero. El plan usa
  explícitamente el escenario de amortización, sin sustituir el dato observado
  en la tabla de la cohorte.
- El plan contiene 322 001 registros cuenta/mes, correspondientes a 83 932
  cuentas que entraron en mora sintética durante 53 cortes, de febrero de 2014
  a junio de 2018. Estas cifras dependen de la matriz manual y la semilla.
- El archivo fuente todavía incluyó 33 filas sin estado: el lector las omitió
  automáticamente. No modificó el CSV original.
- Se verificaron fórmula monetaria, ranking mensual, ausencia de duplicados
  cuenta/mes, proporciones de transición y las gráficas exportadas.
