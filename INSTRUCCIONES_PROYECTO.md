# Guía para replicar, entender y mejorar recupera-ai

Última revisión: 7 de noviembre de 2026.

Esta guía está dirigida a la compañera que recibe el proyecto y a cualquier
asistente de IA que le ayude. Describe el estado implementado, las decisiones
metodológicas y las instrucciones de reproducción. Los resultados indicados
son referencias de ejecuciones anteriores, no promesas para datos diferentes.

## 1. Qué estamos construyendo

El interés de negocio es priorizar la gestión de cobranzas y comprender cómo
una cartera migra entre estados de atraso. Los datos disponibles son históricos
de Lending Club, de Estados Unidos, con importes en USD. No son una cartera
actual de un banco peruano.

El proyecto tiene dos componentes principales:

1. **Predicción retrospectiva de castigo:** entrenar con características de
   originación y comprobar si distinguen préstamos que terminaron en
   `Charged Off` de los que terminaron en `Fully Paid`.
2. **Simulación mensual de cobranza:** generar escenarios de mora, saldos,
   roll rates y listas de llamadas para practicar el proceso mensual.

La predicción se evalúa con desenlaces reales. El escenario mensual se construye
con supuestos explícitos. No se evalúa el modelo contra etiquetas simuladas.
La ruta principal actual es `simulacion_cobranzas.ipynb` o
`simulacion_cobranzas.py`.

Existe además un prototipo inicial de recobro. Es útil para entender la
evolución metodológica, pero usa otra definición de target y otra evaluación.
No se deben mezclar sus probabilidades ni sus métricas con las del flujo actual.

## 2. Antes de ejecutar: qué copiar

La carpeta del proyecto puede estar en cualquier ubicación. `D:\recupera-ai`
es la ubicación del equipo de desarrollo, no un requisito para otro equipo.

Copiar todos los scripts, notebooks, `requirements.txt`, documentación y `tests/`.
El script principal importa funciones y variables de otros módulos; copiar solo
`simulacion_cobranzas.py` no es suficiente.

Estructura recomendada:

```text
recupera-ai/
├── AGENTS.md
├── INSTRUCCIONES_PROYECTO.md
├── README.md
├── requirements.txt
├── requirements-excel.txt
├── simulacion_cobranzas.py
├── simulacion_cobranzas.ipynb
├── evaluar_objetivo.py
├── priorizacion_recobro.py
├── priorizacion_recobro.ipynb
├── roll_rates.py
├── auditar_datos.py
├── revisar_archive2.py
├── tests/
└── data/
    ├── archive/
    │   ├── accepted_2007_to_2018q4.csv/
    │   │   └── accepted_2007_to_2018Q4.csv
    │   └── rejected_2007_to_2018q4.csv/
    │       └── rejected_2007_to_2018Q4.csv
    └── archive2/
        ├── loan.csv
        └── LCDataDictionary.xlsx
```

Atención: en esta estructura `accepted_2007_to_2018q4.csv` es una **carpeta** que
contiene el archivo. Al indicar una ruta manual se debe incluir el nombre del
archivo interior, no solo el directorio.

Para el flujo principal basta el CSV `accepted` y el código. `rejected` es
opcional para auditar solicitudes rechazadas. `archive2` y el diccionario se
necesitan para reproducir la revisión de esa segunda fuente.

`data/` y `outputs/` están excluidos de Git. Clonar el repositorio no descarga
los datasets ni las salidas. Hay que transferir los datos por separado y
conservar su procedencia. Los tres CSV actuales suman aproximadamente 4,65 GB;
se necesita espacio adicional para el entorno y los resultados.

## 3. Instalación en Windows y VS Code

El entorno de referencia utilizó Python 3.13.5. El código declara Python 3.10
o posterior; para aproximar la ejecución de referencia conviene usar Python
3.13 y las versiones indicadas más adelante.

Abrir **la carpeta completa** del proyecto en VS Code. En su terminal PowerShell,
ubicada en la raíz del proyecto:

```powershell
# Comprobar el intérprete disponible.
py --version

# Crear un entorno local. Si no hay lanzador py, usar la ruta del intérprete.
py -3.13 -m venv .venv

# No es necesario activar el entorno ni cambiar la política de PowerShell.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Para ejecutar notebooks con ese entorno.
.\.venv\Scripts\python.exe -m pip install ipykernel
```

Solo para usar `--excel` en el prototipo inicial:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-excel.txt
```

`openpyxl` es opcional; no se instala con los requisitos básicos del flujo
temporal. La lectura del diccionario local usa XML y no necesita ese paquete.

Si `py` no existe pero `python` apunta al intérprete deseado, el comando de
creación es `python -m venv .venv`. Si ninguno está en PATH, usar la ruta completa
del Python instalado. La ruta del equipo original, que puede no existir en otro
equipo, fue:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" --version
```

En VS Code, usar las extensiones Python y Jupyter y seleccionar `.venv` como
intérprete y como kernel del notebook. La terminal y el notebook deben usar
el mismo entorno. `ipykernel` es un complemento de uso del notebook; no está
incluido en el `requirements.txt` actual del procesamiento.

### Versiones y reproducibilidad

Las versiones registradas en las ejecuciones de referencia fueron:

| Componente | Versión |
| --- | --- |
| Python | 3.13.5 |
| numpy | 2.3.3 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |

`requirements.txt` contiene rangos, no un entorno bloqueado. Instalarlo en otro
momento puede resolver versiones diferentes. Si se busca acercarse al entorno
original, después de instalar los requisitos se pueden fijar estas bibliotecas
en el entorno de Python 3.13:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy==2.3.3 pandas==2.3.3 scikit-learn==1.7.2
.\.venv\Scripts\python.exe -m pip check
```

Registrar las versiones realmente utilizadas. Un cambio de biblioteca, datos,
orden de filas, tamaño de bloque o muestreo puede alterar resultados; investigar
la diferencia en lugar de modificar una métrica para hacerla coincidir.

## 4. Primera reproducción

### Opción A: notebook

1. Abrir `simulacion_cobranzas.ipynb` y seleccionar el kernel de `.venv`.
2. Revisar la celda de `ConfigTemporal` y confirmar la ubicación de los datos.
3. Ejecutar las celdas en orden. La celda `ejecutar(config)` entrena, evalúa,
   simula y exporta; las siguientes permiten revisar los resultados.
4. Comprobar las curvas, las métricas y los archivos en `outputs/simulacion/`.

El notebook importa el script: modificar una función en el `.py` y volver a
ejecutar una celda no garantiza recargar el módulo ya importado. Después de
cambiar código, reiniciar el kernel y ejecutar las celdas desde el principio.

### Opción B: terminal

```powershell
.\.venv\Scripts\python.exe simulacion_cobranzas.py
```

Para indicar una ruta explícita y evitar sobrescribir una ejecución anterior:

```powershell
.\.venv\Scripts\python.exe simulacion_cobranzas.py --csv "data\archive\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv" --salida "outputs\replica_companera" --jobs 2
```

Se puede comprobar primero la instalación sin leer los millones de préstamos:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La última ejecución de comprobación reportó 22 pruebas correctas. Los tests
utilizan archivos pequeños sintéticos: no requieren copiar todos los CSV.

### Parámetros desde Python o Jupyter

```python
from pathlib import Path
from simulacion_cobranzas import ConfigTemporal, ejecutar

config = ConfigTemporal(
    csv_path=None,  # autodetección; también admite Path a un CSV concreto
    salida=Path("outputs/replica_companera"),
    chunksize=100_000,
    max_train=100_000,
    muestra_evaluacion=50_000,
    arboles=100,
    jobs=2,
    seed=42,
    meses_colas=6,
    fecha_corte_observada=None,
    guardar_panel=False,
)
resultado = ejecutar(config)
```

Con poca memoria se pueden reducir `chunksize` y `max_train`. Menos árboles
reduce tiempo. Esas variantes deben documentarse como nuevas ejecuciones: no
garantizan los mismos AUC. `guardar_panel=True` puede generar una salida grande;
por defecto se conservan el plan de mora y los agregados de transición.
El plan se escribe por mes sin acumular todas sus filas. Se conservan en memoria
la cohorte de 2014, el mes en curso, los IDs únicos gestionados y agregados de
transición; la lectura por bloques no impone un límite estricto de RAM.

## 5. Qué aporta cada fuente

| Fuente | Revisión realizada | Decisión actual |
| --- | --- | --- |
| accepted | 2.260.668 préstamos con estado, 151 columnas, IDs únicos | Fuente del modelo y base del escenario |
| rejected | 27.648.741 solicitudes, 9 columnas, sin resultado de pagos | No se etiqueta un rechazo como pérdida |
| archive2/loan.csv | 2.260.668 filas, 145 columnas, todos los id/member_id vacíos | No se fusiona con accepted sin una clave verificable |
| LCDataDictionary.xlsx | Definiciones de los campos de Lending Club | Apoya la interpretación de variables |

La última ingesta del archivo `accepted` todavía encontró 33 filas sin estado y
las omitió automáticamente. Puede recibirse una copia ya limpia; no se requiere
que ese conteo sea siempre 33. Se debe verificar el contenido recibido.

`archive2` no añade columnas frente a `accepted` y carece de seis columnas FICO.
Sus estados difieren y su último mes de pago observado llega a febrero de 2019,
frente a marzo de 2019 en `accepted`. Esto sugiere otro corte del mismo universo,
pero no acredita por sí solo la identidad de cada préstamo. No concatenar ambas
fuentes como nuevas cuentas ni unirlas por posición, importe o fechas.

Los `.gz` originales se compararon con sus CSV mediante SHA-256 y tamaño del
contenido descomprimido, y coincidieron. Ya no son necesarios para ejecutar el
flujo con los CSV. Si se reciben copias distintas, esa verificación anterior no
demuestra su integridad: repetirla con los archivos efectivamente recibidos.

## 6. Los tres modelos: target y evaluación

| Flujo | Target positivo | Predictores/evaluación | Referencia |
| --- | --- | --- | --- |
| priorizacion_recobro.py | Fully Paid=1 | Tres variables; test aleatorio de cerrados y calibración | AUC 0,5619 |
| evaluar_objetivo.py | Charged Off=1 | Comparación de tres y 16 variables; train 2011–2013 y test 2014, 36 meses | AUC 0,648428 para 16 |
| simulacion_cobranzas.py | Charged Off=1 | Referencia de 16 y variante de 16 con grade, misma separación de cohortes | AUC 0,666804 con grade en la muestra |

**La clase positiva cambia entre el prototipo inicial y el flujo actual.** En
el inicial un valor alto indica mayor score de pago; en el actual, mayor score
de castigo. No intercambiar las probabilidades ni asumir que son complementos
de modelos distintos.

### Evaluación del flujo actual

- Entrenamiento: préstamos de 36 meses otorgados en 2011–2013, con desenlace
  `Fully Paid` o `Charged Off`. Muestra uniforme máxima de 100.000 cuentas.
- Comparación reproducible: muestra uniforme máxima de 50.000 cerrados de 2014.
- Scoring: todas las cuentas de 2014/36 meses; en la fuente revisada son 162.570.
  Se imputan predictores sin eliminar filas por nulos. Si hay estados no cerrados
  en otra copia, se puntúan pero no se les inventa una etiqueta para evaluar.
- Semilla: 42. No se toman simplemente las primeras N filas ni se rebalancean
  las clases artificialmente.
- Algoritmo: Random Forest de 100 árboles, profundidad máxima 12, mínimo 50
  observaciones por hoja y `max_features=0.7`.
- Preparación: mediana e indicadores de ausencia para numéricas; categoría más
  frecuente y One-Hot Encoding para categóricas. Todo se ajusta en entrenamiento.
  Se admiten categorías nuevas al predecir.

La separación por otorgamiento es **retrospectiva**, no una prueba de que un
modelo entrenado así hubiera estado disponible al empezar 2014. Parte de los
desenlaces de 2013 pudo conocerse años después. Para un backtest operativo se
necesitan fechas de disponibilidad de datos y etiquetas.

### Los 16 predictores de referencia

Numéricos: `loan_amnt`, `annual_inc`, `dti`, `fico_range_low`, `delinq_2yrs`,
`revol_util`, `revol_bal`, `inq_last_6mths`, `open_acc`, `total_acc`, `pub_rec`
y `pub_rec_bankruptcies`.

Categóricos: `home_ownership`, `purpose`, `emp_length` y `verification_status`.

La variante `con_grade_16` reemplaza `pub_rec_bankruptcies` por `grade`.
`grade` contiene una decisión de riesgo de Lending Club en originación; no
representa una evaluación independiente de su política. Ninguno de estos
campos es una variable macroeconómica agregada.

Los scores del flujo temporal **no están calibrados**. `probabilidad_perdida`
es el nombre de salida de un proxy del castigo final, no una PD validada a 12
meses ni una probabilidad de respuesta a una llamada.

## 7. Fechas y fuga de información

El diccionario define `issue_d` como el mes de financiación y `last_pymnt_d`
como el último mes en que se recibió un pago. Las fechas originales tienen
precisión mensual; convertirlas al primer día no aporta precisión diaria.

| Campo calculado | Interpretación correcta |
| --- | --- |
| meses_hasta_ultimo_pago | Diferencia de meses entre otorgamiento y último pago |
| meses_activos_antes_del_atraso_aprox | Alias descriptivo del anterior, sin acreditar inicio de mora |
| dias_desde_ultimo_pago_aprox | Diferencia aproximada hasta un corte explícito; no DPD observado |
| dias_mora_sinteticos | Valor representativo del bucket simulado, no medición real |

No inferir días de atraso verdaderos a partir de esas diferencias. `Fully Paid`
no demuestra que el cliente nunca estuvo en mora ni que se regularizó gracias
a una gestión. `Charged Off` no proporciona por sí solo la fecha del default.

Los predictores de originación excluyen `loan_status`, `total_pymnt`,
`last_pymnt_amnt`, fechas de último pago, `out_prncp`, FICO actualizado,
recuperaciones y acuerdos de liquidación: contienen información posterior al
momento de predicción. La exclusión del target de X es obligatoria.

Si se plantea usar una variable de comportamiento en otro modelo, primero
definir el instante de scoring y demostrar que su valor histórico estaba
disponible entonces. No basta con que la columna exista en el CSV final.

## 8. Cómo funciona la simulación mensual

La matriz `MATRIZ_SUPUESTA` en `simulacion_cobranzas.py` define ocho estados:
`CORRIENTE`, `1-30`, `31-60`, `61-90`, `91-120`, `121+`, `PAGADO` y `CASTIGADO`.

La matriz es manual: **no se estimó con transiciones históricas de Lending
Club**. Cada fila debe ser finita, no negativa y sumar uno. `PAGADO` y
`CASTIGADO` son absorbentes: una cuenta que entra no vuelve al proceso ordinario.

El escenario comienza el mes posterior al otorgamiento de cada préstamo de
2014. Con semilla 42 sortea el estado siguiente y calcula capital con estas
hipótesis:

1. Parte del importe original y usa tasa/cuota contractuales.
2. En meses de mora no amortiza capital.
3. Al curar supone regularización del capital vencido hasta la curva contractual.
4. Al alcanzar al menos 36 meses y quedar corriente supone liquidación.
5. Pagado lleva capital cero. Castigado conserva la exposición y sale del plan
   ordinario de llamadas.
6. Se simula hasta seis meses después del vencimiento de la última originación
   de la cohorte, por defecto. Se omiten intereses de mora, comisiones y recobros.

La trayectoria **no consulta el estado final real, el último pago ni los pagos
acumulados**. El score de castigo permanece fijo; cambian estado y exposición.
No se ha estimado una PD mensual condicionada a la mora simulada.

### Saldo observado frente a saldo simulado

En la cohorte revisada, todas las cuentas de 2014 tenían `out_prncp` observado
igual a cero. Multiplicar ese dato por el score produce prioridades cero.

El archivo `cohorte_2014_observada.csv` mantiene el saldo real del CSV. En
`plan_llamadas_priorizado.csv`, en cambio, `out_prncp` es el capital **simulado**,
`out_prncp_observado` conserva el original y se añaden:

```text
tipo_dato = SIMULADO
origen_saldo = AMORTIZACION_SUPUESTA
```

La fórmula de ordenamiento es:

```text
valor_monetario_en_riesgo = probabilidad_perdida × out_prncp_simulado
```

El plan se ordena por mes y valor descendente dentro del mes, con
`orden_llamada_mes` empezando en uno. Incluye solo cuentas con capital positivo
en buckets de mora, excluyendo pagados y castigados de la gestión ordinaria.
Los importes mantienen la unidad USD, sin conversión monetaria.

Este producto no es una pérdida esperada validada: omite LGD y costos. Tampoco
mide el dinero adicional que se recuperaría por llamar. No sumar los valores
mensuales de una misma exposición como si fueran pérdidas independientes.

### Roll rates

Una tasa por cuentas divide las transiciones desde un estado hacia otro entre
las cuentas del estado de origen, para el mes correspondiente. La ponderada
por dinero usa capital **de origen**, no capital de destino.

`roll_rates_simulados.csv` y sus curvas describen el escenario supuesto.
Para datos mensuales reales existe otro módulo, `roll_rates.py`: requiere
`id_cuenta`, `fecha_corte`, `dias_mora`, `saldo` y `estado_cierre` opcional.
Rechaza duplicados cuenta/mes y no interpreta desapariciones como pagos.

El panel opcional del simulador tiene un contrato de columnas diferente y deja
de emitir una cuenta después de cerrar. No pasarlo directamente al módulo de
roll rates reales sin una adaptación explícita de columnas y seguimiento.

## 9. Archivos y responsabilidades

| Archivo | Responsabilidad |
| --- | --- |
| simulacion_cobranzas.py | Fechas, cohortes, dos modelos, ROC y escenario mensual |
| simulacion_cobranzas.ipynb | Uso guiado del módulo anterior |
| evaluar_objetivo.py | Experimento temporal de tres frente a 16 predictores; define las listas de referencia |
| priorizacion_recobro.py | Prototipo inicial y función compartida localizar_csv |
| priorizacion_recobro.ipynb | Notebook del prototipo inicial |
| roll_rates.py | Transiciones con un panel mensual real |
| auditar_datos.py | Integridad y perfil de accepted/rejected por bloques |
| revisar_archive2.py | Revisión del CSV adicional y lectura del diccionario local |
| tests/test_pipeline.py | Prototipo, limpieza, ranking y detección de rutas |
| tests/test_auditoria.py | Integridad de copias y separación de filas sin estado |
| tests/test_simulacion.py | Fechas, cohortes, nulos, amortización, escenario y ranking |

El flujo temporal devuelve un diccionario con modelo en memoria, cohorte de
2014, métricas, figuras y tabla de roll rates. Actualmente no persiste el modelo
en disco; ejecutar el flujo completo vuelve a entrenarlo.

## 10. Salidas y resultados de referencia

En `outputs/simulacion/`, o en la carpeta indicada con `--salida`:

| Salida | Uso |
| --- | --- |
| plan_llamadas_priorizado.csv | Plan mensual sintético ordenado |
| cohorte_2014_observada.csv | Fechas, resultados, exposición observada y scores de 2014 |
| metricas_temporales.json | Conteos, evaluación, hipótesis y limitaciones |
| curva_roc.png | Comparación de modelos en la misma muestra de cerrados |
| roll_rates_simulados.csv | Transiciones sintéticas por cuentas y saldo |
| curvas_roll_rate_simuladas.png | Gráfica de transiciones desde 31-60 |
| matriz_transicion_supuesta.csv | Matriz del escenario |
| cuentas_revision_simulacion.csv | Cuentas sin datos financieros suficientes para simular |
| panel_mensual_simulado.csv | Opcional: trayectorias generadas, no observaciones reales |

`revision_archive2.json` se produce con `revisar_archive2.py`, no con el flujo
principal. Las auditorías adicionales escriben en `outputs/auditoria/`.
Los CSV principales usan UTF-8 con BOM. Reejecutar en la misma carpeta sobrescribe
salidas del mismo nombre; conservar carpetas separadas para comparar experimentos.
La escritura mensual usa archivos temporales para plan y panel: si falla el
bucle de simulación, no publica el plan parcial ni sustituye el anterior.
Esto no garantiza una transacción conjunta de todas las salidas del pipeline.
Sin cuentas válidas se exportan tablas vacías con cabeceras legibles.

Valores de la ejecución de referencia:

| Medida | Resultado |
| --- | ---: |
| Train muestreado | 100.000 cuentas |
| Muestra comparable de test | 50.000 cuentas |
| Cohorte completa 2014/36 meses | 162.570 cuentas |
| AUC referencia_16, muestra | 0,648428 |
| AUC con_grade_16, misma muestra | 0,666804 |
| AUC con_grade_16, todos los cerrados de 2014 | 0,669640 |
| Filas del plan sintético | 322.001 cuenta/mes |
| Cuentas distintas en el plan | 83.932 |
| Meses con plan | 53, de febrero de 2014 a junio de 2018 |

No comparar AUC de poblaciones diferentes como si fueran un experimento
controlado. No exigir coincidencia exacta de cifras si cambian fuente, limpieza,
versiones o configuración. Los JSON generados por cada ejecución son la evidencia
de esa ejecución, no este cuadro de referencia.

## 11. Problemas frecuentes

| Problema | Qué revisar |
| --- | --- |
| python o py no se reconoce | Ruta del intérprete; usar .venv/Scripts/python.exe una vez creado |
| ModuleNotFoundError | Instalar requisitos en el mismo intérprete/kernel que ejecuta el código |
| No se encuentra accepted | Copia del dataset, carpetas anidadas y argumento --csv |
| PermissionError leyendo un .csv | Puede haberse indicado un directorio con extensión .csv, no el archivo interior |
| No aparece outputs | Ejecutar desde la raíz o usar --salida; comprobar si hubo una excepción antes de exportar |
| Cambié el .py pero el notebook sigue igual | Reiniciar kernel y ejecutar las celdas otra vez |
| Priorización observada de 2014 da cero | Es el saldo del corte final; consultar por separado el plan sintético |
| Los AUC difieren | Datos, versiones, predictores, semilla, muestras, tamaño de bloque y cohortes |
| Memoria insuficiente | Reducir bloque/train/árboles; revisar qué tablas se acumulan; dejar guardar_panel=False |
| No se puede unir archive2 | Sus IDs están vacíos; no inventar correspondencias por posición o atributos parecidos |
| Acentos dañados al generar archivos | Guardar Markdown/notebooks como UTF-8; revisar la codificación de la terminal |

## 12. Instrucciones para la IA que continúe el trabajo

Leer primero esta guía y `AGENTS.md`; después revisar los archivos involucrados
en el cambio solicitado y las métricas disponibles. No asumir que la explicación
del chat original estará disponible. Si código, documentación y evidencia
discrepan, comprobar qué se ejecutó y explicar la diferencia.

Al implementar una mejora:

1. Identificar si modifica predicción con datos reales, escenario sintético o
   herramientas para paneles reales. Mantener esa distinción en nombres y salidas.
2. Declarar target, población, instante de scoring, horizonte y disponibilidad
   temporal de variables/etiquetas antes de cambiar predictores.
3. Conservar los CSV fuente. Hacer limpieza en memoria o en un archivo derivado
   con trazabilidad; no sobrescribir fuentes para facilitar un experimento.
4. Evitar cargar completos los CSV de varios GB cuando basten columnas y bloques.
   Revisar también la memoria de modelos y tablas acumuladas.
5. Ajustar imputaciones, codificaciones y calibración solamente con entrenamiento
   o particiones internas correspondientes. No escoger cambios repetidamente
   mirando el test de 2014; para nuevas decisiones reservar validación adicional.
6. Mantener identificadores como texto y validar unicidad antes de unir. Para
   nuevas fuentes exigir procedencia, fechas y cardinalidad de la unión.
7. Registrar supuestos y parámetros; al cambiar la matriz no llamar a sus tasas
   «roll rates históricos». No usar el estado final para generar una historia
   retrospectiva que luego se presente como conocida al llamar.
8. Ejecutar pruebas relevantes para cambios de lógica. Añadir pruebas cuando
   protejan un comportamiento real: fuga, nulos, cohortes, dinero, transiciones,
   duplicación o exportación. No inventar métricas ni cambiar las pruebas solo
   para que acepten un error.
9. Comparar experimentos sobre poblaciones equivalentes. Separar AUC,
   calibración y utilidad económica; una mejora en una no prueba las otras.
10. Actualizar esta guía, README y el notebook si cambia el uso o significado de
    las salidas. Informar qué cambió, qué se verificó y qué sigue sin demostrarse.

No hace falta rehacer auditorías completas o reentrenar por un cambio solo de
documentación. Tampoco crear restricciones o aprobaciones nuevas por rutina:
trabajar dentro del alcance que solicite la responsable del proyecto.

### Mejoras útiles que aún no están implementadas

- Bloquear y registrar el entorno completo, huellas de fuentes y configuración
  de cada ejecución para facilitar reproducciones exactas.
- Persistir modelo y transformadores junto con versión, variables y metadatos.
- Diseñar una validación temporal con disponibilidad real de las etiquetas y
  un horizonte definido; separar validación de selección y test final.
- Evaluar calibración temporal antes de interpretar scores como probabilidades.
- Comparar con una regresión logística y otras alternativas sobre el mismo
  conjunto de validación; no optimizar solo para superar el AUC del test usado.
- Incorporar escenarios alternativos y sensibilidad de matriz/saldos, mostrando
  la incertidumbre de los supuestos y diferencias de capacidad mensual.
- Añadir capacidad del call center, contactabilidad, costo y restricciones de
  gestión cuando existan esos datos. Con los CSV actuales serían supuestos.
- Sustituir la simulación por un panel observado si se consigue, conservando
  ambos ejercicios separados para contrastar hipótesis con evidencia.

Esta lista es una propuesta de siguientes pasos, no una afirmación de que esas
capacidades ya existan ni una orden de implementarlas todas de inmediato.

Ya se implementaron la exportación incremental del plan/panel, la conservación
del plan anterior ante fallos durante la simulación, la revisión de importes no
finitos y la separación de dependencias opcionales de Excel. Esos cambios son
de ejecución y robustez; no añaden capacidad predictiva ni validan recobro real.

## 13. Datos que faltan para validar un sistema de cobranza real

Se necesitan identificador estable de cuenta, cortes mensuales con DPD y capital,
calendario de vencimientos, pagos fechados, fecha de default/castigo/cierre y
gestiones con fecha, canal, resultado y costo. También una definición de cura
con horizonte de seguimiento y una evaluación de la política frente a una base
comparable; el efecto incremental de llamar requiere un diseño con control.

La ausencia del panel no se resuelve usando la fecha actual ni uniendo archivos
sin claves. El proyecto actual sirve para investigar predicción histórica y
probar un flujo de gestión bajo supuestos; no acredita recobro incremental ni
rendimiento en una cartera actual.

## 14. Mensaje sugerido para iniciar otra conversación con una IA

Copiar y adaptar este texto:

> Estoy trabajando en recupera-ai. Lee primero AGENTS.md e
> INSTRUCCIONES_PROYECTO.md y revisa el código del flujo implicado. La entrada
> principal es simulacion_cobranzas.py y el notebook del mismo nombre. El modelo
> evalúa castigo histórico con cohortes reales; la mora mensual y los saldos del
> plan son sintéticos. Conserva esa distinción y revisa fuga de información antes
> de proponer predictores. Mi tarea concreta ahora es: [describir el cambio].
> Implementa lo necesario, verifica el resultado y explica las limitaciones que
> sigan aplicando. No asumas que tienes los datasets: comprueba sus rutas.

Para ampliar detalles, consultar [README.md](README.md), el
[notebook principal](simulacion_cobranzas.ipynb), el
[script principal](simulacion_cobranzas.py) y los JSON de la ejecución disponible.
