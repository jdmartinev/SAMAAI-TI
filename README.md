# Detección de Anomalías en Series de Tiempo Hidrológicas
## Plan de Acción para el Sistema SAMA — Antioquia, Colombia

**Versión**: 2.0  
**Fecha**: Junio 2026  

---

## 1. Contexto y Restricciones del Problema

El Sistema de Alerta y Monitoreo de Antioquia (SAMA) instrumentaliza cuencas de alto riesgo con sensores de nivel y pluviómetros, entre otros, con frecuencias de muestreo de 5 a 15 minutos. El punto de partida para este plan es una base de datos histórica de menos de un año, con datos que han pasado por un proceso de calidad básico (verificación de rangos físicos).

Este contexto impone restricciones concretas que definen toda la estrategia metodológica:

**Sin etiquetas de anomalías**: no existe un conjunto de datos anotado que indique cuáles observaciones son anómalas. Esto descarta directamente los métodos supervisados y exige un enfoque completamente no supervisado, donde la validación de los modelos depende de inspección visual y criterio experto hidrológico.

**Historia corta (< 1 año)**: muchos métodos que explotan estacionalidad anual son inestables o no aplicables. En particular, Prophet — que modela ciclos anuales mediante series de Fourier — requiere al menos dos ciclos anuales completos para estimar correctamente la componente estacional bimodal de las cuencas andinas. Prophet se propone como meta de mediano plazo, no como herramienta de arranque.

**Frecuencias de muestreo heterogéneas entre sensores**: distintos instrumentos reportan a 5, 10 o 15 minutos. Esto impide construir matrices multivariadas alineadas temporalmente de forma confiable en esta etapa, por lo que todos los modelos de detección deben ser **estrictamente univariados** — un modelo por serie, por estación.

**Datos con calidad básica aplicada**: el histórico disponible ya pasó por verificación de rangos físicos. Esto significa que los errores más groseros (valores negativos, fuera de rango absoluto) ya fueron removidos o marcados. El pipeline propuesto opera sobre este dataset de entrada y se enfoca en anomalías más sutiles.

---

## 2. Taxonomía de Anomalías Objetivo

Dado que los datos ya tienen calidad de rango aplicada, las anomalías que el pipeline debe detectar son las que ese proceso no captura:

![Figura 1. Taxonomía de anomalías en sistemas de monitoreo hidrológico (SAMA)](figs/fig1.png)
*Figura 1. Taxonomía de anomalías en sistemas de monitoreo hidrológico. Se distinguen fallas instrumentales (A), anomalías contextuales e hidrológicas (B) y su impacto operacional (C).*

| Tipo | Descripción | ¿Detectable con rangos? | Prioridad |
|------|-------------|------------------------|-----------|
| Spike puntual | Valor aislado fuera de lo esperado, retorno inmediato | Parcialmente | Alta |
| Valor constante | Secuencia de valores idénticos por N períodos | No | Alta |
| Level shift | Salto abrupto sostenido sin retorno | No | Alta |
| Drift | Desviación gradual creciente o decreciente | No | Media |
| Oscilación artificial | Ruido periódico no presente en la señal física | No | Media |

La distinción entre **falla de sensor** y **evento hidrológico real** es el problema central. Un aumento brusco de nivel puede ser un spike de sensor o el inicio de una avenida torrencial. Sin datos multivariados alineados, esta distinción en esta etapa debe quedar a criterio del operador; el sistema provee el flag y el contexto, no la decisión final.

Un hallazgo relevante de la literatura es que la concordancia entre evaluadores humanos al etiquetar anomalías en series de caudal es sorprendentemente baja — en torno al 12% de solapamiento entre pares de analistas (Rebolho et al., 2021). Esto refuerza el diseño no supervisado y la necesidad de que el sistema explique su decisión, no solo emita una bandera binaria.

---

## 3. Estrategia Metodológica: Complejidad Incremental

La propuesta central es una **progresión de complejidad creciente**, comenzando con los métodos más simples e interpretables y avanzando a medida que se acumula historia, se adquiere confianza en los modelos y eventualmente se construye un conjunto de datos etiquetado.

Esta progresión tiene una lógica operativa: los métodos simples son fáciles de explicar a operadores e hidrológos, fallan de formas predecibles y establecen un baseline contra el cual evaluar métodos más complejos.

```
E1 — Arranque       E2 — Consolidación    E2.5 — ML            E3 — Madurez
(0–3 meses)         (3–6 meses)           (5–6 meses)          (> 18 meses)

Hampel filter   →   STL+IQR (nivel)   →   Isolation        →   Prophet+IC
Cuantiles           ETS o SARIMA*         Forest               (lluvia)
condicionales       (EDA decide)          (univariado,     →   LSTM
Detector                                  features de          Autoencoder
constante                                 ventana)
```

*La elección entre ETS y SARIMA se determina durante el EDA inicial: si los residuos ETS muestran estructura autoregresiva significativa en el ACF/PACF, se usa SARIMA; si son ruido blanco, ETS es suficiente y más eficiente computacionalmente.

Cada etapa hereda los modelos de la anterior y los mantiene activos. No se reemplazan — se suman capas.

---

## 4. Descripción de Métodos por Etapa

### Etapa 1 — Hampel Filter + Cuantiles Condicionales

Los cuantiles globales son el método más simple posible, pero tienen un problema estructural para datos hidrológicos: calculan umbrales sobre toda la distribución histórica sin considerar contexto temporal. Un nivel de 3m puede ser normal a las 3pm en temporada de lluvias y completamente anómalo a las 3am en estiaje — el cuantil global no distingue eso.

E1 propone en cambio dos detectores complementarios diferenciados por variable, más cuantiles globales como fallback para sensores con historia muy corta (< 7 días).

---

#### 1a — Hampel Filter (detector de spikes puntuales)

**Cuándo aplicar**: desde el primer día para nivel; sobre valores > 0 para lluvia.

**Fundamento**: para cada punto `x_t`, compara contra la mediana y MAD (Median Absolute Deviation) de una ventana centrada local. Al usar mediana en lugar de media, es robusto ante la presencia de otros outliers en la ventana:

```
score(t) = |x_t - mediana(ventana)| / (MAD(ventana) × 1.4826)
Anomalía si score(t) > k  (típicamente k = 3)
```

El factor 1.4826 hace el MAD comparable a σ bajo normalidad. A diferencia de los cuantiles globales, el Hampel compara cada punto contra su **vecindad temporal**, lo que lo hace sensible a spikes locales que los cuantiles globales no capturan.

```python
import numpy as np
import pandas as pd

def hampel_filter(serie: pd.Series, ventana: int = 10,
                  k: float = 3.0) -> pd.DataFrame:
    """
    ventana: número de puntos por lado (ventana total = 2*ventana + 1)
    Para datos a 15-min: ventana=8 → ±2 horas de contexto local.
    Para datos a 5-min:  ventana=24 → ±2 horas de contexto local.
    """
    mediana_roll = serie.rolling(2 * ventana + 1, center=True).median()
    mad_roll = serie.rolling(2 * ventana + 1, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    sigma_rob = 1.4826 * mad_roll

    score = np.abs(serie - mediana_roll) / (sigma_rob + 1e-9)
    anomalia = score > k

    return pd.DataFrame({
        'anomalia': anomalia,
        'score': score,
        'mediana_local': mediana_roll,
        'sigma_rob': sigma_rob
    })
```

**Para nivel**: aplicar directamente sobre la serie completa. La ventana de ±2 horas (8 puntos a 15-min) es suficiente para que la mediana local siga crecidas reales sin enmascarar spikes de sensor.

**Para lluvia — problema estructural del MAD=0**: si en una ventana de 21 puntos hay 15 o más ceros, la mediana es cero y el MAD es cero. Con MAD≈0, cualquier valor positivo genera score infinito — todos los eventos de lluvia serían anomalías. La solución es aplicar el Hampel **solo sobre valores positivos**, tratando la ocurrencia de lluvia como problema separado:

```python
def hampel_lluvia(serie: pd.Series, ventana: int = 10,
                  k: float = 3.0) -> pd.DataFrame:
    """Aplica Hampel solo sobre valores positivos de lluvia."""
    resultado = pd.DataFrame({
        'anomalia': False, 'score': 0.0
    }, index=serie.index)

    mask_positivos = serie > 0
    if mask_positivos.sum() < 2 * ventana + 1:
        return resultado  # insuficientes positivos para la ventana

    serie_pos = serie[mask_positivos]
    hampel_pos = hampel_filter(serie_pos, ventana=ventana, k=k)

    resultado.loc[mask_positivos, 'anomalia'] = hampel_pos['anomalia']
    resultado.loc[mask_positivos, 'score'] = hampel_pos['score']
    return resultado
```

---

#### 1b — Cuantiles Condicionales por Hora (detector de nivel fuera de contexto)

**Cuándo aplicar**: con al menos 30 días de historia (suficiente para tener ~30 observaciones por celda hora×variable).

**Fundamento**: en lugar de un umbral global, se calculan percentiles condicionados a la hora del día. Esto captura la variación diaria sin necesidad de descomposición — un nivel de 3m a las 2am activa la alerta aunque ese mismo nivel a las 4pm sea completamente normal.

```python
def entrenar_cuantiles_condicionales(df: pd.DataFrame,
                                      col: str = 'valor',
                                      alpha: float = 0.01) -> pd.DataFrame:
    """
    Entrena cuantiles por hora del día.
    Con más historia, se puede añadir mes como segunda dimensión.
    """
    tabla = df.groupby(df.index.hour)[col].quantile(
        [alpha / 2, 1 - alpha / 2]
    ).unstack(level=1)
    tabla.columns = ['q_low', 'q_high']
    return tabla

def detectar_cuantiles_condicionales(valor: float, hora: int,
                                      tabla: pd.DataFrame) -> dict:
    q_low = tabla.loc[hora, 'q_low']
    q_high = tabla.loc[hora, 'q_high']
    es_anomalia = (valor < q_low) or (valor > q_high)
    score = max(
        (q_low - valor) / (q_high - q_low + 1e-9),
        (valor - q_high) / (q_high - q_low + 1e-9),
        0.0
    )
    return {"anomalia": es_anomalia, "score": score,
            "q_low": q_low, "q_high": q_high}
```

Cuando se acumule historia de múltiples meses, se puede extender a cuantiles condicionados por **hora × mes**, añadiendo así la dimensión estacional sin necesidad de STL.

---

#### 1c — Detector de Valor Constante (obligatorio para ambas variables)

```python
def detector_constante(serie: pd.Series, n_min: int = 6) -> dict:
    """
    Retorna True si los últimos n_min valores son idénticos.
    Para datos a 15-min: n_min=6 → 90 minutos constantes.
    Para lluvia: n_min=12 → 3 horas en cero durante período lluvioso
                             requiere contexto adicional (flag para operador).
    """
    ultimos = serie.iloc[-n_min:]
    es_constante = ultimos.nunique() == 1
    return {
        "anomalia": es_constante,
        "score": 1.0 if es_constante else 0.0,
        "tipo": "cero_extendido" if ultimos.iloc[-1] == 0 else "valor_constante"
    }
```

**Nota para lluvia**: un bloque de ceros extendido puede ser sensor atascado o simplemente ausencia real de precipitación. Sin datos de estaciones vecinas no es posible discriminar automáticamente en E1. El sistema emite un flag de tipo `cero_extendido` para revisión del operador — no una alerta crítica.

---

#### 1d — Cuantiles Globales (fallback)

Se mantienen únicamente para sensores con historia < 7 días, donde no hay suficientes datos para estratificar por hora ni para que el Hampel tenga ventana completa:

```python
def detector_cuantiles_global(historia: np.ndarray, valor: float,
                               alpha: float = 0.01) -> dict:
    q_low = np.quantile(historia, alpha / 2)
    q_high = np.quantile(historia, 1 - alpha / 2)
    es_anomalia = (valor < q_low) or (valor > q_high)
    score = max(
        (q_low - valor) / (q_high - q_low + 1e-9),
        (valor - q_high) / (q_high - q_low + 1e-9),
        0.0
    )
    return {"anomalia": es_anomalia, "score": score}
```

---

**Resumen E1 por variable**:

| Variable | Detector principal | Detector constante | Fallback (< 7 días) |
|----------|-------------------|-------------------|---------------------|
| Nivel | Hampel (serie completa) + Cuantiles condicionales por hora | Valor constante N consecutivos → alerta | Cuantiles globales |
| Lluvia | Hampel (solo valores > 0) + Cuantiles condicionales por hora | Cero extendido N consecutivos → flag operador | Cuantiles globales |

**Reentrenamiento E1**: los cuantiles condicionales se recalculan mensualmente con toda la historia disponible. El Hampel no requiere reentrenamiento — opera sobre ventana deslizante en tiempo real.

---

### Etapa 2a — STL + IQR (para sensores de nivel)

**Cuándo aplicar**: con al menos 60–90 días de historia limpia a resolución sub-horaria.

**Fundamento**: STL descompone la serie en tendencia, componente estacional y residual. El IQR se aplica sobre el residual. La diferencia clave respecto a los cuantiles es que STL *sustrae el patrón esperado* antes de evaluar: un valor alto en época de lluvias no genera residual alto si el modelo ya sabe que esa época tiene valores altos.

```
Y(t) = Tendencia(t) + Estacionalidad(t) + Residual(t)
Anomalía si: |Residual(t)| > Q3 + k · IQR  o  < Q1 - k · IQR
```

Para datos a 15 minutos con ciclo diario, `period = 96` (96 muestras × 15 min = 24 horas).

```python
from statsmodels.tsa.seasonal import STL

def entrenar_stl(serie: pd.Series, period: int = 96) -> dict:
    stl = STL(serie, period=period, robust=True)
    resultado = stl.fit()
    residual = resultado.resid
    Q1, Q3 = np.percentile(residual, [25, 75])
    IQR = Q3 - Q1
    return {"modelo": resultado, "Q1": Q1, "Q3": Q3, "IQR": IQR,
            "tendencia": resultado.trend, "estacional": resultado.seasonal}

def detectar_stl_iqr(residual_nuevo: float, params: dict,
                     k: float = 3.0) -> dict:
    lim_inf = params["Q1"] - k * params["IQR"]
    lim_sup = params["Q3"] + k * params["IQR"]
    es_anomalia = (residual_nuevo < lim_inf) or (residual_nuevo > lim_sup)
    score = max(
        (lim_inf - residual_nuevo) / (params["IQR"] + 1e-9),
        (residual_nuevo - lim_sup) / (params["IQR"] + 1e-9),
        0.0
    )
    return {"anomalia": es_anomalia, "score": score,
            "lim_inf": lim_inf, "lim_sup": lim_sup}
```

**Por qué STL para nivel y no para lluvia**: el nivel de río tiene un ciclo diario relativamente estable (influenciado por temperatura, evapotranspiración y dinámica fluvial base) cuya estructura STL captura bien con `period=96`. La lluvia, en cambio, es intermitente (muchos ceros), asimétrica y su ciclo anual bimodal no puede representarse con un único `period` — para lluvia, STL no es apropiado.

**Reentrenamiento**: semanal, con ventana deslizante de los últimos 60 días.

---

### Etapa 2b — ETS o SARIMA: el EDA decide (nivel y lluvia)

**Cuándo aplicar**: con al menos 90 días de historia, en paralelo con STL.

**Fundamento compartido**: ambos modelos capturan la **dependencia temporal de corto plazo** — detectan un valor anómalo no solo porque rompe el patrón estacional sino porque es inconsistente con los últimos N valores observados. Esto es complementario a STL: STL detecta anomalías respecto al patrón esperado; estos modelos detectan anomalías respecto a la dinámica reciente.

**Criterio de selección basado en EDA**:

El EDA inicial sobre cada sensor determina cuál modelo usar. El protocolo es:

1. Ajustar ETS sobre la serie de entrenamiento
2. Calcular los residuos del ajuste
3. Examinar ACF y PACF de los residuos

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

def evaluar_residuos_ets(serie: pd.Series, period: int = 96) -> dict:
    """
    Ajusta ETS y evalúa si los residuos tienen estructura residual.
    Retorna recomendación: 'ets' o 'sarima'.
    """
    modelo = ExponentialSmoothing(
        serie,
        trend='add',
        seasonal='add',
        seasonal_periods=period
    ).fit(optimized=True)

    residuos = modelo.resid

    # Test de Ljung-Box: H0 = residuos son ruido blanco
    lb = acorr_ljungbox(residuos, lags=[10, 20, 48], return_df=True)
    p_values = lb['lb_pvalue']

    if (p_values > 0.05).all():
        return {"modelo": modelo, "recomendacion": "ets",
                "razon": "Residuos son ruido blanco — ETS es suficiente"}
    else:
        return {"modelo": modelo, "recomendacion": "sarima",
                "razon": "Residuos tienen estructura AR — SARIMA justificado"}
```

**Opción A — ETS** (preferida si residuos pasan el test de Ljung-Box):

ETS es más rápido, más robusto ante no-estacionariedad y tiene selección automática de estructura mediante AIC. Con 111 sensores, la diferencia de velocidad respecto a SARIMA es significativa en producción.

```python
def entrenar_ets(serie: pd.Series, period: int = 96):
    return ExponentialSmoothing(
        serie,
        trend='add',
        seasonal='add',        # o 'mul' si la amplitud estacional crece con la media
        seasonal_periods=period
    ).fit(optimized=True)

def detectar_ets(modelo, valor_nuevo: float, k_sigma: float = 3.0) -> dict:
    residuos_train = modelo.resid
    sigma = residuos_train.std()
    pred = modelo.forecast(1)[0]
    residuo = abs(valor_nuevo - pred)
    return {"anomalia": residuo > k_sigma * sigma,
            "score": residuo / (sigma + 1e-9),
            "prediccion": pred}
```

**Opción B — SARIMA** (si residuos ETS muestran estructura significativa en ACF):

```python
from pmdarima import auto_arima

def entrenar_sarima(serie: pd.Series, m: int = 96):
    """
    auto_arima con m=96 puede ser lento — ejecutar una vez durante setup,
    no en cada reentrenamiento. Guardar el orden encontrado y re-ajustar
    con ese orden fijo en reentrenamientos subsiguientes.
    """
    return auto_arima(
        serie,
        seasonal=True, m=m,
        stepwise=True,
        suppress_warnings=True,
        max_p=3, max_q=3,
        max_P=1, max_Q=1    # limitar órdenes estacionales por costo computacional
    )
```

**Nota sobre lluvia en ambos modelos**: la distribución fuertemente asimétrica (muchos ceros, cola derecha larga) requiere transformación previa. Se recomienda `log(x + 0.1)`. Alternativamente, modelar la ocurrencia (lluvia/no-lluvia) como proceso binario separado y aplicar ETS/SARIMA solo sobre los valores positivos.

**Reentrenamiento**: mensual, con ventana deslizante de 90 días. El orden SARIMA se redetermina solo cuando el test de Ljung-Box sobre los residuos del modelo vigente rechaza ruido blanco.

---

### Etapa 2.5 — Isolation Forest (univariado con features de ventana)

**Cuándo aplicar**: con al menos 4–5 meses de historia limpia. Requiere suficiente variedad de condiciones (al menos parte de una temporada de lluvias y parte de una seca) para que el modelo aprenda qué es "normal" en ambos regímenes.

**Por qué aquí y no antes**: Isolation Forest opera sobre un espacio de features construido a partir de ventanas temporales. Con poca historia, ese espacio no representa bien la variabilidad normal de la serie — el modelo marcaría como anómalos comportamientos que simplemente no había visto. Con 4–5 meses hay suficiente diversidad.

**Por qué es complementario a STL/ETS/SARIMA**: los modelos estadísticos detectan anomalías punto a punto (un valor vs. su predicción). Isolation Forest detecta **anomalías de ventana** — segmentos cuya morfología completa es anómala aunque ningún punto individual sea extremo. Drift gradual, oscilaciones sostenidas y patrones de forma inusual son ejemplos que los modelos estadísticos no capturan bien pero IF sí.

**Feature engineering univariado sobre ventana deslizante**:

La clave es construir features que capturen la forma y dinámica de la ventana, no solo el valor puntual:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def extraer_features_ventana(serie: pd.Series,
                              ventana: int = 96) -> pd.DataFrame:
    """
    Extrae features estadísticas y de forma sobre ventanas deslizantes.
    ventana=96 → 24h a 15-min.
    """
    features = {}
    roll = serie.rolling(ventana)

    # Estadísticas de primer orden
    features['media'] = roll.mean()
    features['std'] = roll.std()
    features['min'] = roll.min()
    features['max'] = roll.max()
    features['rango'] = features['max'] - features['min']
    features['percentil_10'] = roll.quantile(0.10)
    features['percentil_90'] = roll.quantile(0.90)

    # Dinámica y tendencia
    features['delta_media'] = features['media'].diff()        # cambio en media rolling
    features['pendiente'] = serie.diff(ventana) / ventana     # tendencia lineal aprox.
    features['energia'] = roll.apply(lambda x: np.sum(x**2))  # energía de la señal

    # Forma de la distribución
    features['skewness'] = roll.skew()
    features['kurtosis'] = roll.kurt()

    # Autocorrelación a lag 1 (indicador de suavidad vs. ruido)
    features['autocorr_lag1'] = serie.rolling(ventana).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan
    )

    # Contexto temporal (cíclico) — captura hora del día y época del año
    features['hora_sin'] = np.sin(2 * np.pi * serie.index.hour / 24)
    features['hora_cos'] = np.cos(2 * np.pi * serie.index.hour / 24)
    features['mes_sin'] = np.sin(2 * np.pi * serie.index.month / 12)
    features['mes_cos'] = np.cos(2 * np.pi * serie.index.month / 12)

    return pd.DataFrame(features, index=serie.index).dropna()

def entrenar_isolation_forest(feats: pd.DataFrame,
                               contamination: float = 0.02) -> tuple:
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,   # ~2% esperado de anomalías
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X)
    return clf, scaler

def detectar_isolation_forest(clf, scaler, feats: pd.DataFrame) -> pd.DataFrame:
    X = scaler.transform(feats)
    scores = clf.decision_function(X)   # más negativo = más anómalo
    predicciones = clf.predict(X)       # -1 = anomalía, 1 = normal
    # Normalizar score a [0, 1] donde 1 es más anómalo
    score_norm = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return pd.DataFrame({
        'anomalia': predicciones == -1,
        'score': score_norm
    }, index=feats.index)
```

**Parámetro `contamination`**: en ausencia de etiquetas, 0.02 (2%) es un punto de partida conservador. Se calibra iterativamente usando la tasa de alertas objetivo (< 3% del total de datos). Si el operador confirma que la mayoría de las alertas son válidas, bajar a 0.01; si hay muchos falsos positivos, subir a 0.03–0.05.

**Interpretabilidad con SHAP**: aunque Isolation Forest no es inherentemente interpretable, SHAP permite identificar qué features contribuyeron más al score de anomalía de un punto específico — información útil para el operador:

```python
import shap

def explicar_anomalia(clf, scaler, feats_punto: pd.DataFrame,
                      nombres_features: list) -> dict:
    explainer = shap.TreeExplainer(clf)
    X = scaler.transform(feats_punto)
    shap_values = explainer.shap_values(X)
    return dict(zip(nombres_features, shap_values[0]))
```

**Reentrenamiento**: mensual, con ventana deslizante de 90 días de features calculadas sobre datos limpios.

---

### Etapa 3 — LSTM Autoencoder (nivel y lluvia, largo plazo)

**Cuándo aplicar**: con al menos 18 meses de historia. El autoencoder necesita haber visto suficientes ciclos de comportamiento normal — estacionales, de eventos, de operación diaria — para aprender una representación compacta de "normalidad". Con historia corta, el modelo no puede distinguir anomalías de condiciones simplemente no vistas.

**Fundamento**: el LSTM autoencoder aprende a comprimir y reconstruir secuencias temporales normales. La anomalía se define por el **error de reconstrucción**: si el modelo no puede reconstruir bien una ventana, es porque esa ventana no se parece a nada que haya visto durante el entrenamiento.

```
Ventana temporal → Encoder (LSTM) → Representación latente → Decoder (LSTM) → Reconstrucción
                                                                     ↓
                                              Error de reconstrucción (MAE)
                                              Anomalía si MAE > umbral
```

```python
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 n_layers: int = 2, window_size: int = 96):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, n_layers,
                               batch_first=True, dropout=0.2)
        self.decoder = nn.LSTM(hidden_size, hidden_size, n_layers,
                               batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_size, input_size)
        self.window_size = window_size

    def forward(self, x):
        # x: (batch, window_size, input_size)
        _, (hidden, cell) = self.encoder(x)
        # Repetir el estado latente para generar la secuencia decodificada
        latent = hidden[-1].unsqueeze(1).repeat(1, self.window_size, 1)
        decoded, _ = self.decoder(latent)
        return self.output_layer(decoded)

def calcular_umbral_reconstruccion(modelo, dataloader,
                                    percentil: float = 99.0) -> float:
    """
    El umbral se determina empíricamente: percentil alto del MAE
    sobre datos de entrenamiento (normales).
    """
    errores = []
    modelo.eval()
    with torch.no_grad():
        for batch in dataloader:
            reconstruccion = modelo(batch)
            mae = torch.mean(torch.abs(batch - reconstruccion), dim=[1, 2])
            errores.extend(mae.numpy())
    return np.percentile(errores, percentil)
```

**Ventajas respecto a Isolation Forest**:
- Captura dependencias temporales de largo alcance (LSTM con ventanas de 96–288 puntos)
- El error de reconstrucción es continuo y bien calibrado como score de anomalía
- Funciona bien para detectar anomalías de forma compleja: drift lento, oscilaciones sostenidas, patrones no vistos

**Por qué no antes**: requiere suficientes datos para que el encoder aprenda representaciones latentes estables. Con < 6 meses, el modelo memoriza el ruido en lugar de aprender la estructura.

**Reentrenamiento**: trimestral, con ventana de los últimos 365 días.

---

### Etapa 3 — Prophet + Intervalo de Credibilidad (lluvia, mediano plazo)

**Cuándo aplicar**: cuando se disponga de al menos 2 ciclos anuales completos (~18–24 meses de historia). No aplicar antes — con historia corta, los armónicos de Fourier que representan la estacionalidad anual no convergen bien y el modelo genera intervalos incorrectos.

**Fundamento**: Prophet modela la serie como suma de componentes con estructura bayesiana, incluyendo múltiples estacionalidades simultáneas. Para lluvia en cuencas andinas, la estacionalidad anual bimodal (dos temporadas de lluvias: marzo–mayo y septiembre–noviembre) es precisamente el tipo de estructura que Prophet está diseñado para capturar.

```python
from prophet import Prophet

def entrenar_prophet_lluvia(df_train: pd.DataFrame) -> Prophet:
    """df_train con columnas 'ds' (datetime) y 'y' (lluvia)."""
    modelo = Prophet(
        interval_width=0.99,
        yearly_seasonality=True,       # captura bimodalidad anual
        daily_seasonality=True,        # captura ciclo convectivo vespertino
        seasonality_mode='multiplicative',  # varianza escala con la media
        changepoint_prior_scale=0.05   # tendencia poco flexible (lluvia es estacionaria)
    )
    modelo.fit(df_train)
    return modelo

def detectar_prophet(modelo: Prophet, df_nuevo: pd.DataFrame) -> pd.DataFrame:
    forecast = modelo.predict(df_nuevo[['ds']])
    resultado = df_nuevo.merge(
        forecast[['ds', 'yhat_lower', 'yhat_upper']], on='ds'
    )
    resultado['anomalia'] = (
        (resultado['y'] < resultado['yhat_lower']) |
        (resultado['y'] > resultado['yhat_upper'])
    )
    resultado['score'] = np.maximum(
        (resultado['yhat_lower'] - resultado['y']) /
        (resultado['yhat_upper'] - resultado['yhat_lower'] + 1e-9),
        (resultado['y'] - resultado['yhat_upper']) /
        (resultado['yhat_upper'] - resultado['yhat_lower'] + 1e-9)
    ).clip(lower=0)
    return resultado

```

**Por qué Prophet y no STL para lluvia**:
- STL con un único `period` no puede representar la estacionalidad bimodal anual
- Prophet maneja nativamente gaps de datos (frecuentes en sensores de campo)
- El intervalo de credibilidad es más informativo que un umbral IQR para comunicar incertidumbre a operadores

**Reentrenamiento**: mensual con ventana deslizante de los últimos 365 días.

---

## 5. Arquitectura del Pipeline

![Figura 2. Pipeline escalonado de detección de anomalías para SAMA](fig2.png)
*Figura 2. Pipeline escalonado de detección de anomalías para SAMA. Cada capa actúa como filtro progresivo con complejidad y latencia crecientes.*

El pipeline implementa un principio de **complejidad incremental activa**: en producción corren simultáneamente todos los modelos disponibles en la etapa actual. Las salidas de cada modelo se agregan en un score final ponderado, no en una lógica de fail-fast pura. Esto permite que modelos simples y complejos se complementen y que el operador vea qué modelos flaggearon el punto.

```
Dato entrante (serie univariada por sensor)
        │
        ▼
┌─────────────────────────────────────────┐
│  VALIDACIÓN PREVIA (heredada de QA)     │
│  Rango físico, gaps, ceros inválidos    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  CAPA 1 — Cuantiles históricos          │  Siempre activa (E1+)
│  + detector de valor constante          │  Score: s1
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  CAPA 2 — STL+IQR (nivel)              │  Activa desde E2 (≥ 60 días)
│           ETS o SARIMA (ambas)          │  Score: s2
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  CAPA 3 — Isolation Forest              │  Activa desde E2.5 (≥ 5 meses)
│           (features de ventana)         │  Score: s3
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  CAPA 4 — LSTM Autoencoder              │  Activa desde E3 (≥ 18 meses)
│           Prophet+IC (lluvia)           │  Score: s4
└─────────────┬───────────────────────────┘
              │
              ▼
     Score agregado: S = Σ wᵢ·sᵢ  (solo modelos activos)
     Alerta si S ≥ umbral adaptativo
```

El **score agregado ponderado** permite graduar la alerta (warning vs. crítica) y registrar cuánto contribuyó cada modelo — información valiosa para el operador y para el loop de feedback. Solo participan en el agregado los modelos activos en la etapa actual.

Los pesos se calibran empíricamente durante la validación operacional. Un punto de partida razonable para cada etapa:

| Etapa | s1 (cuantiles) | s2 (STL/ETS) | s3 (IF) | s4 (LSTM/Prophet) |
|-------|---------------|--------------|---------|-------------------|
| E1 | 1.0 | — | — | — |
| E2 | 0.2 | 0.8 | — | — |
| E2.5 | 0.15 | 0.50 | 0.35 | — |
| E3 | 0.10 | 0.35 | 0.25 | 0.30 |

---

## 6. Gestión del Entrenamiento Inicial

### 6.1 Preparación del histórico base

El histórico inicial, aunque ya tiene calidad de rango aplicada, requiere pasos adicionales antes de usarse para entrenar:

1. **Detección de gaps**: identificar períodos sin datos. Los gaps > 2 horas deben marcarse explícitamente, no imputarse con la media — especialmente si coinciden con eventos de lluvia intensa (los sensores fallan precisamente durante eventos extremos).

2. **Limpieza de cuantiles extremos**: aunque el rango físico fue verificado, pueden existir valores en los percentiles 0.1% y 99.9% que son técnicamente válidos pero distorsionan los modelos. Se recomienda aplicar winsorización leve antes del entrenamiento (no del almacenamiento).

3. **Segmentación por sensor**: cada sensor entrena su propio modelo independiente. No hay matriz multivariada en esta etapa.

4. **Inspección visual obligatoria**: antes de entrenar el primer modelo sobre un sensor, un analista debe revisar la serie completa. Dada la ausencia de etiquetas, esta inspección inicial es el único mecanismo de validación de ground truth disponible.

### 6.2 Arranque sin etiquetas: estrategia de validación

La ausencia de etiquetas impide calcular métricas supervisadas (precision, recall, F1). La estrategia de validación en esta etapa es:

**Validación visual**: el analista revisa una muestra de los puntos flaggeados por cada modelo y los clasifica como verdadero positivo o falso positivo. Esta clasificación retroalimenta el ajuste de umbrales y, progresivamente, construye el primer conjunto de datos etiquetado.

**Métricas operacionales proxy**:
- *Tasa de alertas por sensor por semana*: un modelo bien calibrado debería generar entre 0.5% y 3% de alertas sobre el total de datos. Tasas superiores indican umbral demasiado sensible; inferiores, modelo insuficientemente sensible.
- *Distribución temporal de alertas*: las alertas deberían correlacionar con períodos de lluvia intensa o eventos conocidos, no distribuirse aleatoriamente.
- *Concordancia entre modelos*: si Cuantiles y SARIMA coinciden en flaggear un punto, la probabilidad de que sea anomalía real es mayor que si solo uno de ellos lo detecta.

**Protocolo de feedback humano** (desde el primer día):

```
Punto flaggeado
      │
      ▼
Cola de revisión (operador)
      │
      ├──► Confirma anomalía → excluir de entrenamiento futuro
      │                        + agregar a dataset etiquetado
      │
      └──► Falso positivo  → registrar + ajustar umbral del modelo
                             que generó el falso positivo
```

Este loop es el mecanismo principal de mejora continua y la única forma de construir etiquetas en ausencia de un dataset anotado previo.

---

## 7. Pipeline de Reentrenamiento (MLOps)

### 7.1 Frecuencias por modelo

| Modelo | Variable | Frecuencia | Ventana de entrenamiento | Datos mínimos requeridos |
|--------|----------|------------|--------------------------|--------------------------|
| Hampel filter | Nivel y lluvia | No requiere (ventana deslizante) | — | 7 días (ventana operativa) |
| Cuantiles condicionales | Nivel y lluvia | Mensual | Toda la historia disponible | 30 días |
| Cuantiles globales (fallback) | Nivel y lluvia | Mensual | Toda la historia disponible | 7 días |
| STL + IQR | Nivel | Semanal | Últimos 60 días | 60 días |
| ETS / SARIMA | Nivel y lluvia | Mensual | Últimos 90 días | 90 días |
| Isolation Forest | Nivel y lluvia | Mensual | Últimos 90 días (features) | 150 días |
| LSTM Autoencoder | Nivel y lluvia | Trimestral | Últimos 365 días | 540 días |
| Prophet | Lluvia | Mensual | Últimos 365 días | 730 días (2 años) |

**Razonamiento de las ventanas**:
- Los cuantiles se benefician de toda la historia disponible (más datos = mejor estimación de percentiles extremos).
- STL necesita historia suficiente para estimar la estacionalidad diaria, pero no más de 60 días — historia muy larga introduce no-estacionariedad.
- SARIMA con 90 días captura bien la dinámica estacional diaria sin perder sensibilidad a cambios recientes.
- Prophet necesita múltiples ciclos anuales; no se activa hasta tener 2 años.

### 7.2 Triggers de reentrenamiento reactivo

Además del calendario, el sistema debe reentrenar cuando detecte degradación del modelo:

**Deriva en tasa de alertas (proxy de model drift)**:
```
Si tasa_alertas_últimas_2_semanas > 3 × tasa_alertas_baseline:
    → umbral probablemente desactualizado → reentrenar
```

**Cambio estructural en la serie (CUSUM sobre media rolling)**:
```python
def detectar_cambio_estructural(serie: pd.Series,
                                 ventana: int = 96*7,   # 1 semana a 15-min
                                 umbral_delta: float = 0.3) -> bool:
    """Compara la media de la última semana con la semana anterior."""
    if len(serie) < 2 * ventana:
        return False
    media_reciente = serie.iloc[-ventana:].mean()
    media_anterior = serie.iloc[-2*ventana:-ventana].mean()
    rango = serie.quantile(0.95) - serie.quantile(0.05) + 1e-9
    return abs(media_reciente - media_anterior) / rango > umbral_delta
```

**Reemplazo o recalibración de sensor**: este evento debe registrarse en el sistema de gestión de activos y disparar automáticamente el reentrenamiento de todos los modelos de ese sensor, reseteando la ventana de entrenamiento desde la fecha del cambio.

### 7.3 Stack MLOps recomendado

| Componente | Herramienta | Rol |
|------------|-------------|-----|
| Orquestación | **Prefect** (más simple que Airflow) | DAGs de reentrenamiento programado y reactivo |
| Registro de modelos | **MLflow** | Versionar modelos por sensor, tracking de parámetros y métricas |
| Monitoreo de drift | **Evidently AI** | Reportes automáticos de drift en distribución de entrada |
| Almacenamiento de series | **TimescaleDB** (PostgreSQL + extensión) | Eficiente para series de tiempo a alta frecuencia |
| Servicio de inferencia | **FastAPI** + worker async | Liviano, adecuado para on-premise |
| Versionado de datos | **DVC** | Reproducibilidad de entrenamientos por sensor |

### 7.4 DAG de reentrenamiento (Prefect)

```python
from prefect import flow, task
import mlflow

@task
def verificar_datos_suficientes(sensor_id: str, modelo: str,
                                  min_dias: int) -> bool:
    n_dias = contar_dias_historia_limpia(sensor_id)
    return n_dias >= min_dias

@task
def reentrenar_modelo(sensor_id: str, modelo_tipo: str,
                      ventana_dias: int) -> dict:
    datos = cargar_datos_limpios(sensor_id, dias=ventana_dias)
    # excluir puntos confirmados como anómalos por operadores
    datos = excluir_anotados_anomalos(datos, sensor_id)
    nuevo_modelo = entrenar(modelo_tipo, datos)
    metricas = calcular_metricas_operacionales(nuevo_modelo, datos)
    return {"modelo": nuevo_modelo, "metricas": metricas}

@task
def promover_modelo(sensor_id: str, nuevo_modelo: dict,
                    modelo_tipo: str) -> None:
    with mlflow.start_run():
        mlflow.log_params(nuevo_modelo["modelo"].get_params())
        mlflow.log_metrics(nuevo_modelo["metricas"])
        mlflow.sklearn.log_model(nuevo_modelo["modelo"],
                                  f"{sensor_id}_{modelo_tipo}")

@flow(name="reentrenamiento-semanal-sama")
def pipeline_reentrenamiento_semanal():
    sensores = obtener_lista_sensores_activos()
    for sensor_id in sensores:
        # STL+IQR — semanal para sensores de nivel
        if es_sensor_nivel(sensor_id):
            if verificar_datos_suficientes(sensor_id, "stl_iqr", 60):
                modelo = reentrenar_modelo(sensor_id, "stl_iqr", 60)
                promover_modelo(sensor_id, modelo, "stl_iqr")
```

---

## 8. Hoja de Ruta por Etapas

| Etapa | Tiempo | Historia mínima | Modelos que se activan | Hito de validación |
|-------|--------|----------------|------------------------|--------------------|
| **E0 — EDA** | Semanas 1–3 | — | — | ACF/PACF por sensor, selección de `period`, inspección visual, criterio ETS vs SARIMA |
| **E1 — Arranque** | Meses 1–3 | 7 días | Hampel filter + cuantiles condicionales + detector constante | Tasa de alertas < 3%, inspección visual de muestra aleatoria |
| **E2 — Consolidación** | Meses 3–6 | 60–90 días | + STL+IQR (nivel), + ETS/SARIMA | Concordancia entre modelos > 60% en alertas, primeras 50 etiquetas manuales |
| **E2.5 — ML clásico** | Meses 5–6 | 150 días | + Isolation Forest (features de ventana) | Score IF correlaciona con alertas de E2, análisis SHAP de features relevantes |
| **E3 — DL + Prophet** | Mes 18+ | 540–730 días | + LSTM Autoencoder, + Prophet (lluvia) | Evaluación supervisada con dataset etiquetado acumulado, F1 > 0.75 |

### Fase 0 — EDA y setup (previo a E1)

Antes de entrenar cualquier modelo, se requiere:

1. **EDA por sensor**: distribución de valores, histograma, serie temporal completa, identificación de gaps, frecuencia real de muestreo (puede diferir de la nominal).
2. **Caracterización de estacionalidad**: periodograma o autocorrelación para confirmar que existe ciclo diario capturado en los datos disponibles.
3. **Selección de `period`**: para datos a 15-min, `period=96`. Para datos a 5-min, `period=288`. Verificar empíricamente con ACF.
4. **Inspección visual de al menos 4 semanas** por sensor con un analista hidrológico — esta inspección genera las primeras anotaciones informales que orientan la calibración de umbrales.

---

## 9. Consideraciones Específicas para Cuencas Andinas

**Ciclo convectivo vespertino**: en muchas cuencas de Antioquia, la lluvia ocurre preferentemente en las tardes (convección orográfica). Esto genera un ciclo diario fuerte en los pluviómetros que STL y SARIMA capturan, pero que puede confundirse con anomalías si los modelos se inicializan con poca historia.

**Respuesta hiperrápida**: cuencas de montaña con pendientes pronunciadas tienen tiempos de concentración de minutos a pocas horas. A 15-min de resolución, un evento real puede verse como un spike puntual si el tiempo de concentración es menor que el intervalo de muestreo. Estos casos son difíciles de distinguir de fallas de sensor sin contexto adicional.

**Gaps correlacionados con eventos extremos**: los sensores tienden a fallar durante eventos de alta lluvia (inundación del equipo, pérdida de comunicación satelital). Imputar estos gaps con valores promedio es incorrecto y contamina el histórico de entrenamiento. Los gaps > 2 horas durante períodos de lluvia detectada en estaciones vecinas deben marcarse como "gap durante evento potencial" y excluirse del entrenamiento.

**Distribución no-gaussiana de lluvia**: las series de lluvia tienen distribución fuertemente asimétrica con exceso de ceros. Para SARIMA, se recomienda transformación log(x + 0.1) antes del ajuste. Para cuantiles, el percentil 0% es cero la mayor parte del tiempo — usar los percentiles condicionados a lluvia > 0 para el umbral superior, y tratar la ocurrencia (lluvia vs. no-lluvia) como variable binaria separada.

---

## 10. Referencias

- Blázquez-García et al. (2021). A review on outlier/anomaly detection in time series data. *ACM Computing Surveys*, 54(3), 1–33.
- Darban et al. (2024). Deep Learning for Time Series Anomaly Detection: A Survey. *ACM Computing Surveys*.
- DAGRAN (2023). Sistema de Alerta y Monitoreo de Antioquia — SAMA. Gobernación de Antioquia.
- Hampel, F. R. (1974). The influence curve and its role in robust estimation. *Journal of the American Statistical Association*, 69(346), 383–393.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. (ETS/Exponential Smoothing)
- Kulanuwat et al. (2021). Anomaly detection using a sliding window technique and data imputation with machine learning for hydrological time series. *Water*, 13(13).
- Leigh et al. (2019). A framework for automated anomaly detection in high-frequency water-quality data from in situ sensors. *Journal of Hydrology*, 586, 124797.
- Liu et al. (2008). Isolation Forest. *IEEE International Conference on Data Mining*.
- Rebolho et al. (2021). Anomaly detection in streamflow time series — inter-evaluator agreement study, 674 series. *ResearchGate*.
- Taylor, S. J. & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.
- Ye et al. (2025). A Survey of Deep Anomaly Detection in Multivariate Time Series. *Sensors*, 25(1), 190.

---

*Este documento es un borrador técnico interno. Para comentarios o revisiones, contactar al equipo de modelamiento hidrológico.*
