# Detección de Anomalías en Series de Tiempo Hidrológicas
## Plan de Acción para el Sistema SAMA — Antioquia, Colombia

**Versión**: 1.0  
**Fecha**: Junio 2026  
**Audiencia**: Ingenieros e investigadores DAGRAN / Universidad de Antioquia

---

## 1. Contexto y Motivación

El Sistema de Alerta y Monitoreo de Antioquia (SAMA), operado por el DAGRAN en convenio con la Universidad de Antioquia, instrumentaliza cuencas de alto riesgo en Antioquia con sensores de nivel, pluviómetros, estaciones meteorológicas y cámaras. Con más de 111 instrumentos en 36 municipios priorizados, el volumen de datos generados supera la capacidad de revisión manual oportuna.

El problema central no es de expertise sino de escala: **la proporción entre volumen de datos y capacidad de revisión humana crece con cada nuevo sensor instalado**. Esto introduce dos riesgos operativos:

- **Falsos negativos**: anomalías reales (eventos de avenida torrencial) no detectadas a tiempo por saturación del analista.
- **Falsos positivos**: alertas espurias por datos de sensor defectuoso que erosionan la confianza del sistema comunitario.

Este informe propone un pipeline automatizado de detección de anomalías en tres capas metodológicas, combinando métodos estadísticos clásicos con machine learning, y describe la infraestructura de MLOps necesaria para su operación sostenible.

---

## 2. Taxonomía de Anomalías en Datos SAMA

Antes de seleccionar métodos es necesario precisar qué constituye una anomalía en este contexto. La literatura hidrológica identifica las siguientes categorías, ordenadas por prioridad operativa:

| Tipo | Descripción | Variable afectada | Causa típica |
|------|-------------|-------------------|--------------|
| **Spike puntual** | Valor aislado fuera de rango, retorno inmediato | Nivel, lluvia | Golpe mecánico, interferencia eléctrica |
| **Valor constante** | Secuencia de valores idénticos > N periodos | Nivel, lluvia | Sensor atascado, falla de comunicación |
| **Level shift** | Salto abrupto sostenido sin retorno | Nivel | Obstrucción, recalibración incorrecta |
| **Drift** | Desviación gradual creciente o decreciente | Nivel | Degradación del sensor, sedimentación |
| **Oscilación artificial** | Ruido periódico no presente en señal física | Lluvia | Interferencia en transmisión |
| **Anomalía contextual** | Valor plausible individualmente pero inconsistente con el contexto multivariado | Nivel | Nivel sube sin lluvia previa en cuenca |

La distinción entre **error de sensor** y **evento hidrológico real** es el desafío central: un spike de nivel puede ser un fallo electrónico o el inicio de una avenida torrencial. El contexto multivariado (lluvia acumulada en la cuenca) es la única forma de discriminarlos automáticamente.

Un hallazgo relevante de la literatura es que la concordancia entre evaluadores humanos al etiquetar anomalías en series de caudal es sorprendentemente baja — en torno al 12% de solapamiento entre pares de analistas (Rebolho et al., 2021). Esto tiene dos implicaciones directas: (1) priorizar métodos no supervisados sobre supervisados, dado que las etiquetas históricas son escasas y ruidosas; (2) diseñar el sistema para que explique su decisión, no solo que emita una bandera binaria.

---

## 3. Métodos Seleccionados

Se seleccionaron tres métodos complementarios que cubren el espacio de detección desde reglas interpretables hasta modelos multivariados.

La Capa 1 aplica detectores distintos según la variable, motivado por sus propiedades estadísticas:

| Variable | Método Capa 1 | Justificación |
|----------|---------------|---------------|
| **Nivel** | STL + IQR | Ciclo diario regular y suave, estacionariedad relativa, varianza estable |
| **Lluvia** | Prophet + intervalo de credibilidad | Distribución asimétrica, estacionalidad bimodal anual, muchos ceros |

### 3.1a STL + IQR — para sensores de nivel

**Fundamento**: Seasonal-Trend decomposition using Loess (STL) descompone la serie en tendencia, componente estacional y residual. El IQR (Interquartile Range) se aplica sobre el residual para identificar puntos cuyo comportamiento no es explicado ni por la tendencia ni por el patrón cíclico esperado.

```
Y(t) = Tendencia(t) + Estacionalidad(t) + Residual(t)

Anomalía si: Residual(t) < Q1 - k·IQR  o  Residual(t) > Q3 + k·IQR
```

STL es apropiado para nivel porque esta variable tiene un ciclo diario relativamente estable (influenciado por temperatura, evapotranspiración y dinámica fluvial) cuya amplitud no varía drásticamente entre temporadas. El IQR sobre el residual es robusto ante la distribución no-gaussiana leve de los residuos de nivel.

**Ventajas en contexto SAMA**:
- Ciclo diario de nivel capturado con `period=24` (datos horarios) o `period=96` (datos 15-min)
- `robust=True` reduce el peso de los propios outliers al estimar los componentes
- Completamente interpretable: se puede mostrar a un operador cuál es el residual y cuál es el umbral
- Liviano computacionalmente — adecuado para reentrenamiento semanal en producción

**Limitaciones**:
- Univariado — no cruza información entre sensores
- Con un único `period`, no captura la modulación anual de la amplitud del ciclo diario — aceptable para nivel, problemático para lluvia
- Batch por naturaleza; en producción se aplica sobre ventana deslizante con reentrenamiento semanal

**Parámetros clave**:

| Parámetro | Descripción | Rango sugerido |
|-----------|-------------|---------------|
| `period` | Longitud del ciclo estacional en muestras | 24 (horario), 96 (15-min) |
| `k` | Multiplicador del IQR para el umbral | 2.5–3.5 según sensibilidad requerida |
| `robust` | Usa regresión M-estimator en Loess | Siempre `True` en datos de campo |

---

### 3.1b Prophet + Intervalo de Credibilidad — para pluviómetros

**Fundamento**: Prophet (Taylor & Letham, 2018) modela la serie como suma de componentes con estructura bayesiana:

```
Y(t) = g(t) + s(t) + h(t) + ε(t)
```

donde `g(t)` es la tendencia (lineal o logística), `s(t)` la suma de componentes estacionales (Fourier) y `h(t)` efectos de eventos especiales. Prophet genera nativamente un **intervalo de predicción** `[ŷ_lower, ŷ_upper]`; un valor real fuera de ese intervalo se marca como anómalo.

```python
from prophet import Prophet

modelo = Prophet(
    interval_width=0.99,          # intervalo de credibilidad al 99%
    yearly_seasonality=True,      # captura bimodalidad anual
    daily_seasonality=True,       # captura ciclo convectivo diario
    seasonality_mode='multiplicative'  # varianza escala con la media — correcto para lluvia
)
modelo.fit(df_train)  # df con columnas 'ds' y 'y'

forecast = modelo.predict(df_future)
anomalias = (df_real['y'] < forecast['yhat_lower']) | \
            (df_real['y'] > forecast['yhat_upper'])
```

**Por qué Prophet y no STL para lluvia**:

- **Estacionalidad bimodal**: el régimen de lluvias en Antioquia tiene dos picos anuales (marzo–mayo, septiembre–noviembre). STL con un único `period` no puede representar esta estructura; Prophet la captura mediante series de Fourier con múltiples armónicos.
- **Distribución asimétrica**: las series de lluvia tienen muchos ceros y cola derecha larga. El modo `multiplicative` de Prophet modela que la varianza crece con la media, lo que es apropiado para esta distribución.
- **Datos faltantes**: Prophet interpola nativamente sobre gaps, a diferencia de STL que requiere imputación previa.
- **Intervalo de credibilidad**: el umbral de detección no es un parámetro arbitrario sino un intervalo estadístico directamente interpretable como "rango de comportamiento esperable al 99%".

**Ventajas en contexto SAMA**:
- Captura la modulación estacional bimodal sin configuración adicional
- Maneja gaps de comunicación sin preprocesamiento
- `interval_width` es el único parámetro de sensibilidad a calibrar por estación

**Limitaciones**:
- Mayor costo computacional que STL — reentrenamiento mensual en lugar de semanal
- Puede sobreajustar si la historia es corta (< 1 año) — no recomendable para estaciones nuevas hasta tener al menos dos ciclos anuales
- Univariado, como STL

**Parámetros clave**:

| Parámetro | Descripción | Valor sugerido |
|-----------|-------------|---------------|
| `interval_width` | Amplitud del intervalo de predicción | 0.99 (conservador para alertas) |
| `seasonality_mode` | Cómo interactúan las estacionalidades | `'multiplicative'` para lluvia |
| `yearly_seasonality` | Armónicos anuales | `True` — captura bimodalidad |
| `daily_seasonality` | Armónicos diarios | `True` — captura convección vespertina |

---

### 3.2 SARIMA Residuals

**Fundamento**: SARIMA (Seasonal AutoRegressive Integrated Moving Average) modela la serie como función de sus valores pasados y errores pasados, con componentes estacionales. Los residuos del modelo fitted representan la parte de la señal no explicada por la dinámica autoregresiva. Valores cuyo residuo supera un umbral de `k·σ` se marcan como anómalos.

La diferencia clave respecto a STL es que SARIMA captura **dependencia temporal de corto plazo**: detecta un valor anómalo no solo porque rompe el patrón estacional sino porque es inconsistente con los últimos N valores observados.

**Notación**: SARIMA(p,d,q)(P,D,Q)_s

- `(p,d,q)`: órdenes autoregresivo, de integración y de media móvil no-estacionales
- `(P,D,Q)_s`: ídem estacionales con período `s`

**Selección automática de orden** con `pmdarima.auto_arima` — se ejecuta una vez durante el setup por estación, no en cada inferencia.

**Ventajas en contexto SAMA**:
- Captura anomalías dinámicas: un nivel que sube cuando llevaba 3 horas bajando
- Ampliamente aceptado en hidrología operacional, fácil de justificar metodológicamente
- Produce intervalos de predicción que dan una medida natural de "cuánto de anómalo" es un valor

**Limitaciones**:
- Univariado
- Asume estacionariedad después de diferenciación — puede requerir transformación logarítmica para series de lluvia
- Sensible a gaps de datos; requiere interpolación previa o modelos con manejo de datos faltantes

**Relación con Capa 1**: los métodos son complementarios, no redundantes. STL/Prophet detectan anomalías respecto al patrón estacional esperado; SARIMA detecta anomalías respecto a la dinámica reciente. Un punto puede pasar la Capa 1 y fallar SARIMA (ej. un valor dentro del rango estacional pero inconsistente con los últimos 6 valores observados).

---

### 3.3 Isolation Forest (multivariado)

**Fundamento**: Isolation Forest (Liu et al., 2008) construye un ensemble de árboles de decisión aleatorios. La intuición es que los puntos anómalos son más fáciles de "aislar" (requieren menos particiones para separarse del resto). El score de anomalía es la profundidad promedio de aislamiento normalizada.

En el contexto de SAMA, la innovación está en el **feature engineering multivariado**: en lugar de operar sobre el valor crudo, se construyen features que capturan el contexto temporal y las relaciones entre sensores.

**Features propuestas**:

```
Para cada variable v en {nivel, lluvia, temperatura}:
  - v_t              : valor actual
  - Δv_t             : primera diferencia (velocidad de cambio)
  - μ(v, w)          : media móvil en ventana w
  - σ(v, w)          : desviación estándar móvil en ventana w

Features cruzadas (nivel × lluvia):
  - lluvia_acum_6h   : lluvia acumulada en las últimas 6 horas (proxy de input de cuenca)
  - ratio_nivel_lluvia: nivel_delta / (lluvia_acum_6h + ε)  — detecta nivel sin lluvia
  - hora_del_dia     : codificación cíclica (sin, cos)
  - mes              : codificación cíclica (sin, cos)
```

**Ventajas en contexto SAMA**:
- No supervisado — no requiere etiquetas
- Captura anomalías contextuales multivariadas indetectables con métodos univariados
- Robusto ante alta dimensionalidad
- Score continuo (no solo binario) permite priorizar alertas

**Limitaciones**:
- La calidad del resultado depende fuertemente del feature engineering
- No es online — corre en batch diario sobre la ventana reciente
- El parámetro `contamination` (proporción esperada de anomalías) requiere calibración por estación y por variable

---

## 4. Arquitectura del Pipeline en Capas

El pipeline implementa un principio de **fail-fast escalonado**: cada capa filtra lo que la anterior no puede, con latencia y costo computacional crecientes. Si una capa detecta una anomalía, se emite la alerta sin invocar las capas siguientes.

```
Dato entrante (sensor nivel / pluviómetro)
        │
        ▼
┌─────────────────────────────────────┐
│  CAPA 0 — Reglas físicas            │  Latencia: < 1 ms
│  Umbrales de dominio hardcoded      │  Costo: mínimo
│  Valor negativo, rango imposible,   │  Interpretabilidad: total
│  sensor atascado (N consecutivos)   │
└─────────────┬───────────────────────┘
              │ pasa
              ▼
┌─────────────────────────────────────┐
│  CAPA 1 — STL + IQR                 │  Latencia: segundos
│  Detección por residual estacional  │  Costo: bajo
│  Univariado, por estación           │  Interpretabilidad: alta
└─────────────┬───────────────────────┘
              │ pasa
              ▼
┌─────────────────────────────────────┐
│  CAPA 2 — SARIMA residuals          │  Latencia: segundos
│  Detección por dinámica autoregres. │  Costo: medio
│  Univariado, por estación           │  Interpretabilidad: alta
└─────────────┬───────────────────────┘
              │ pasa
              ▼
┌─────────────────────────────────────┐
│  CAPA 3 — Isolation Forest          │  Latencia: minutos (batch)
│  Detección multivariada             │  Costo: medio
│  Cruza nivel + lluvia + contexto    │  Interpretabilidad: media
└─────────────┬───────────────────────┘
              │
              ▼
     Alerta con metadatos:
     {capa, tipo, score, variables_involucradas}
```

Cada alerta emitida incluye: timestamp, estación, capa que la detectó, tipo de anomalía, score de confianza y — para Capa 3 — las features con mayor contribución al score (SHAP values).

---

## 5. Estrategia de Reentrenamiento

Esta es una de las decisiones más críticas del sistema y frecuentemente subestimada.

### 5.1 ¿Por qué reentrenar?

Los modelos se degradan por **data drift**: el comportamiento del sensor y del fenómeno cambia en el tiempo por:

- Cambios estacionales (régimen de lluvias bimodal en Antioquia)
- Degradación física del sensor (deriva lenta)
- Cambios en la cuenca (deforestación, obras, eventos geomorfológicos)
- Reemplazo o recalibración del sensor (ruptura estructural en la serie)

Un modelo entrenado en época seca aplicado en temporada de lluvias generará exceso de falsos positivos. Un modelo que no conoce la deriva progresiva de un sensor la marcará indefinidamente como anomalía en lugar de actualizar su baseline.

### 5.2 Frecuencias de reentrenamiento por capa

| Capa | Método | Variable | Frecuencia | Trigger adicional |
|------|--------|----------|------------|-------------------|
| Capa 1 | STL + IQR | Nivel | Semanal (rolling 60 días) | Cambio de temporada |
| Capa 1 | Prophet + IC | Lluvia | Mensual (rolling 365 días) | Inicio de temporada de lluvias |
| Capa 2 | SARIMA | Nivel y lluvia | Mensual (rolling 90 días) | CUSUM sobre MAPE supera 2× baseline |
| Capa 3 | Isolation Forest | Multivariado | Mensual (rolling 90 días) | PSI > 0.2 en features de entrada |

**Razonamiento**:

- **STL+IQR (nivel, semanal)**: liviano y se beneficia de actualización frecuente para seguir cambios graduales en el comportamiento del sensor y la dinámica fluvial.
- **Prophet (lluvia, mensual)**: requiere suficiente historia anual para estimar bien la estacionalidad bimodal; ventana mínima de 365 días. Reentrenamiento mensual con ventana larga estabiliza los armónicos de Fourier.
- **SARIMA (mensual)**: el costo de selección de orden con `auto_arima` hace prohibitivo el reentrenamiento semanal; la ventana de 90 días captura bien la dinámica reciente sin perder contexto estacional.
- **Isolation Forest (mensual)**: requiere suficiente data representativa de condiciones normales; ventanas cortas producen modelos inestables en el espacio de features multivariadas.

### 5.3 Triggers basados en métricas (reentrenamiento reactivo)

Además del reentrenamiento programado, el sistema debe disparar reentrenamiento cuando se detecte degradación del modelo:

**Population Stability Index (PSI)** sobre las features de entrada:
```
PSI = Σ (p_actual - p_ref) · ln(p_actual / p_ref)

PSI < 0.1  : sin cambio significativo
0.1–0.2    : cambio moderado — monitorear
PSI > 0.2  : cambio mayor — reentrenar
```

**MAPE sobre predicciones SARIMA** en ventana reciente:
```
Si MAPE(últimas 168h) > 2 × MAPE(baseline) → reentrenar SARIMA
```

**Tasa de alertas** (proxy de modelo degradado):
```
Si tasa_alertas_7d > 3 × tasa_alertas_baseline → revisar umbral o reentrenar
```

---

## 6. Infraestructura MLOps

### 6.1 Componentes del sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     INFRAESTRUCTURA MLOPS                   │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Ingesta de  │    │  Feature     │    │  Model       │  │
│  │  datos       │───▶│  Store       │───▶│  Registry    │  │
│  │  (SAMA API)  │    │  (features   │    │  (versiones  │  │
│  │              │    │  calculadas) │    │  de modelos) │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                 │           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐  │
│  │  Alertas +   │◀───│  Inference   │◀───│  Scheduler   │  │
│  │  Dashboard   │    │  Service     │    │  (retrain    │  │
│  │              │    │              │    │  triggers)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Monitoring: PSI, MAPE drift, alert rate, data gaps  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Stack tecnológico recomendado

| Componente | Herramienta | Justificación |
|------------|-------------|---------------|
| Orquestación de pipelines | **Apache Airflow** o **Prefect** | DAGs para reentrenamiento programado y reactivo |
| Registro de experimentos | **MLflow** | Tracking de parámetros, métricas y artefactos por modelo/estación |
| Feature store | **Hopsworks** (open source) o tablas en PostgreSQL + TimescaleDB | Features calculadas reutilizables entre capas |
| Servicio de inferencia | **FastAPI** + worker async | Liviano, adecuado para on-premise o cloud bajo |
| Monitoreo de modelos | **Evidently AI** | Detección de data drift y model drift con reportes automáticos |
| Versionado de datos | **DVC** | Reproducibilidad de entrenamientos por estación |
| Almacenamiento de modelos | **MLflow Model Registry** | Stages: Staging → Production → Archived |

### 6.3 DAG de reentrenamiento (Airflow/Prefect)

```python
# Pseudocódigo del DAG de reentrenamiento semanal

@dag(schedule="0 2 * * 1")  # Lunes 2am
def retrain_pipeline_sama():

    @task
    def verificar_drift(estacion_id, modelo):
        """Calcula PSI y MAPE reciente. Retorna True si hay que reentrenar."""
        psi = calcular_psi(features_recientes, features_referencia)
        mape_reciente = calcular_mape_reciente(modelo, estacion_id)
        return psi > 0.2 or mape_reciente > 2 * modelo.mape_baseline

    @task
    def reentrenar(estacion_id, modelo_tipo, ventana_dias=90):
        """Reentrena el modelo con la ventana más reciente de datos limpios."""
        data = cargar_datos_limpios(estacion_id, dias=ventana_dias)
        nuevo_modelo = entrenar(modelo_tipo, data)
        metricas = evaluar(nuevo_modelo, data_validacion)
        return nuevo_modelo, metricas

    @task
    def promover_si_mejora(nuevo_modelo, metricas, modelo_produccion):
        """Solo promueve a producción si el nuevo modelo es mejor."""
        if metricas["f1"] >= modelo_produccion.metricas["f1"] * 0.98:
            mlflow.register_model(nuevo_modelo, stage="Production")
        else:
            mlflow.register_model(nuevo_modelo, stage="Staging")
            alertar_equipo("Modelo nuevo no supera producción — revisión manual")
```

### 6.4 Gestión de datos de entrenamiento

Un aspecto frecuentemente ignorado: **los datos usados para reentrenar deben ser datos limpios** (sin anomalías confirmadas). De lo contrario, el modelo aprende que las anomalías son normales.

Protocolo propuesto:

1. Los datos marcados como anomalía por cualquier capa van a una **cola de revisión**.
2. Un operador confirma o descarta la anomalía (interfaz simple: confirmar / falso positivo).
3. Los datos confirmados como anómalos se **excluyen** del conjunto de entrenamiento.
4. Los datos confirmados como falsos positivos retroalimentan el ajuste de umbrales.

Este loop de feedback humano es el mecanismo de mejora continua del sistema.

---

## 7. Métricas de Evaluación

Dado que las etiquetas son escasas, se propone una evaluación mixta:

### 7.1 Métricas supervisadas (sobre eventos etiquetados disponibles)

| Métrica | Descripción | Prioridad en SAMA |
|---------|-------------|-------------------|
| **Recall** | Fracción de anomalías reales detectadas | **Máxima** — miss de avenida torrencial es inaceptable |
| **Precision** | Fracción de alertas que son reales | Alta — falsas alarmas erosionan confianza comunitaria |
| **F1-score** | Media armónica precision/recall | Referencia general |
| **Tiempo de detección** | Latencia desde inicio de anomalía hasta alerta | Crítico para alertas tempranas |

### 7.2 Métricas no supervisadas (operacionales)

- **Tasa de alertas por estación y temporada**: detecta modelos demasiado sensibles o insensibles
- **Distribución de capas que detectan**: si Capa 3 detecta el 90% de las anomalías, Capas 1-2 están mal calibradas
- **PSI mensual por feature y por estación**: monitoreo de drift en producción

---

## 8. Cronograma de Implementación

| Fase | Duración | Actividades | Entregable |
|------|----------|-------------|------------|
| **F1 — EDA y baseline** | 3 semanas | Análisis exploratorio de series SAMA, identificación de periodicidades, distribución de gaps, construcción del dataset de evaluación con eventos etiquetados históricos | Notebook EDA + dataset de evaluación |
| **F2 — Capas 0 y 1** | 2 semanas | Implementación de reglas físicas + STL+IQR, calibración de umbrales por estación, validación visual con operadores | Módulo `capa0_capa1.py` + resultados de calibración |
| **F3 — Capa 2** | 2 semanas | Selección automática de orden SARIMA por estación (`auto_arima`), evaluación comparativa contra STL+IQR | Módulo `capa2_sarima.py` + reporte comparativo |
| **F4 — Capa 3** | 3 semanas | Feature engineering multivariado, entrenamiento Isolation Forest, análisis SHAP de contribución de features | Módulo `capa3_iforest.py` + análisis de features |
| **F5 — Integración y MLOps** | 3 semanas | Pipeline unificado, DAGs de reentrenamiento en Airflow, Model Registry en MLflow, monitoreo con Evidently | Pipeline en producción + documentación de operación |
| **F6 — Validación operacional** | 2 semanas | Período de operación paralela (sistema automático + revisión manual), ajuste fino de umbrales, protocolo de feedback humano | Informe de validación + parámetros finales |

**Total estimado**: 15 semanas

---

## 9. Consideraciones Específicas para Cuencas Andinas

Los métodos descritos asumen implícitamente ciertas propiedades de las series que deben verificarse para el contexto de Antioquia:

**Estacionalidad bimodal**: el régimen de lluvias en Antioquia tiene dos temporadas de lluvias (marzo–mayo, septiembre–noviembre). Esta es la razón principal para usar Prophet en pluviómetros en lugar de STL: los armónicos de Fourier de Prophet representan nativamente esta estructura sin configuración adicional. Para sensores de nivel, el ciclo diario dominante es suficientemente estable para que STL sea adecuado.

**Respuesta hiperrápida**: cuencas de montaña con alta pendiente generan crecidas súbitas con tiempos de concentración de minutos a pocas horas. La resolución temporal del sensor (típicamente 15 min en SAMA) debe alinearse con el `period` usado en los modelos.

**Distribución no-gaussiana**: las series de lluvia tienen distribución fuertemente asimétrica (muchos ceros, cola derecha larga). El IQR es más robusto que z-score en este caso, pero para SARIMA puede ser necesaria transformación logarítmica o Box-Cox.

**Datos faltantes correlacionados con eventos extremos**: los sensores tienden a fallar precisamente durante eventos de alta lluvia (inundación del equipo, pérdida de comunicación). Imputar estos gaps con la media histórica es incorrecto — se recomienda marcarlos explícitamente como "gap durante evento potencial" y no usarlos para reentrenamiento.

---

## 10. Referencias

- Blázquez-García et al. (2021). A review on outlier/anomaly detection in time series data. *ACM Computing Surveys*, 54(3), 1–33.
- Darban et al. (2024). Deep Learning for Time Series Anomaly Detection: A Survey. *ACM Computing Surveys*.
- DAGRAN (2023). Sistema de Alerta y Monitoreo de Antioquia — SAMA. Gobernación de Antioquia.
- Kulanuwat et al. (2021). Anomaly detection using a sliding window technique and data imputation with machine learning for hydrological time series. *Water*, 13(13).
- Leigh et al. (2019). A framework for automated anomaly detection in high-frequency water-quality data from in situ sensors. *Journal of Hydrology*, 586, 124797.
- Liu et al. (2008). Isolation Forest. *IEEE International Conference on Data Mining*.
- Rebolho et al. (2021). Anomaly detection in streamflow time series — inter-evaluator agreement study, 674 series. *ResearchGate*.
- Taylor, S. J. & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45. (Prophet)
- Ye et al. (2025). A Survey of Deep Anomaly Detection in Multivariate Time Series. *Sensors*, 25(1), 190.

---

*Este documento es un borrador técnico interno. Para comentarios o revisiones, contactar al equipo de modelamiento hidrológico.*
