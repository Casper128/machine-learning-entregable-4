# Evento Evaluativo 4 — Equipo 10
 
**Detección de ofertas laborales fraudulentas y análisis de patrones textuales**
 
Asignatura: Análisis de Datos · Docente: Daniel Alexis Nieto Mora · Ingeniería en Ciencia de Datos, periodo 2026-2.
 
## Integrantes
 
- @Casper128 (Darwin Alexander Osorio Ospina)
- @dariomontoya0505 (Dario Giovanny Montoya Londoño)
- @Franyelica (Franyelica Valentina Garcia Fernandez)
- @santiagopabon15 (Santiago Pabón Acevedo)
## Problema y datos
 
Las plataformas de empleo pueden usarse para publicar ofertas falsas que buscan estafar a los solicitantes u obtener sus datos personales. Este trabajo aborda dos preguntas complementarias:
 
1. **Supervisada:** ¿se puede clasificar una oferta como legítima o fraudulenta a partir de su texto y de sus variables estructuradas?
2. **No supervisada:** ¿qué temas y grupos existen en las publicaciones cuando se ignoran las etiquetas, y se relacionan con el fraude?
**Datos:** [Real or Fake Job Posting Prediction (Kaggle)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction). El archivo original tiene 17 880 ofertas, 866 de ellas fraudulentas. Tras eliminar duplicados exactos quedan **17 599 ofertas, de las cuales 856 son fraudulentas (4,86 %)**. Este desbalance condiciona todo el diseño: un clasificador que respondiera siempre «legítima» tendría más de 95 % de exactitud sin detectar ningún fraude, por lo que se evalúa con precision, recall, F1 y PR-AUC.
 
## Estructura del proyecto
 
| Ruta | Contenido |
| --- | --- |
| `evento4_fake_jobs.ipynb` | Desarrollo completo ejecutado (entrega principal). |
| `evento4_fake_jobs.py` | Mismas celdas en formato `# %%`, útil para revisar cambios. |
| `descargar_datos.py`, `data/README.md` | Descarga reproducible y huella (SHA-256) del CSV; los datos no se versionan. |
| `results/` | Métricas, perfiles de grupos, selección de k, tópicos NMF y `resumen.json`. |
| `figures/` | 18 figuras generadas por el notebook. |
| `requirements.txt`, `requirements-lock.txt` | Dependencias y versiones fijadas. |
 
## Ejecución
 
Desde esta carpeta, con Python 3.12:
 
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
python descargar_datos.py
jupyter notebook evento4_fake_jobs.ipynb
```
 
En Jupyter, elegir el kernel del entorno y usar *Restart Kernel and Run All Cells*. La búsqueda de hiperparámetros y las proyecciones (t-SNE, UMAP) pueden tardar varios minutos. Conviene revisar las advertencias de convergencia.
 
## Metodología
 
1. **Partición sin fuga.** El conjunto de prueba se reserva por grupos de textos repetidos **antes** del EDA, de modo que una misma oferta duplicada no aparezca en entrenamiento y en prueba. El EDA usa solo el conjunto de entrenamiento.
2. **Preprocesamiento en pipelines.** Limpieza de texto, TF-IDF, codificación y escalado se ajustan únicamente con datos de entrenamiento.
3. **Componente supervisado.** Regresión logística, SVM lineal (LinearSVC) y modelos de referencia sobre variables estructuradas (Random Forest, HistGradientBoosting). Tratamiento del desbalance con ponderación de clases y submuestreo, y ajuste del umbral con puntajes *out-of-fold* (OOF). La selección del modelo se hace por precisión promedio (AP) en validación, nunca con la prueba.
4. **Componente no supervisado.** TF-IDF reducido con SVD (LSA). K-Means y agrupamiento jerárquico de Ward, comparados en la misma muestra aleatoria y sin usar etiquetas. Los perfiles se calculan sobre el corpus completo; NMF (tópicos) e Isolation Forest (anomalías) complementan el análisis.
5. **Visualización.** SVD, t-SNE y UMAP, además de la comparación entre enfoques y sus limitaciones.
## Resultados
 
### Componente supervisado
 
La incorporación del texto es lo que más mejora la detección: con solo variables estructuradas el mejor modelo alcanza F1 = 0,696, mientras que los modelos lineales sobre texto y variables estructuradas llegan a 0,860–0,883.
 
**Validación cruzada (5 pliegues, entrenamiento).** Los modelos con texto superan claramente a los de variables estructuradas (AP ≈ 0,88–0,91 frente a ≈ 0,46–0,80). Submuestrear degrada el AP (0,64). La ponderación de clases casi no cambia el AP de LinearSVC (0,909 frente a 0,906 sin tratamiento), pero mejora el recall y el F1 con el umbral por defecto (F1 0,832 frente a 0,786 en validación). Ver `results/validacion_cruzada.csv` y `figures/09_comparacion_cv.png`.
 
**Selección.** LinearSVC fue el modelo elegido (AP de 0,875 en la búsqueda de hiperparámetros de 3 pliegues; no es comparable con el AP de 5 pliegues de la comparación inicial, p. ej. 0,909).
 
**Evaluación en prueba** (3 520 ofertas, 171 fraudulentas):
 
| Modelo | Umbral | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest (estructurado) | Por defecto | 0,660 | 0,737 | 0,696 | 0,824 | 0,981 |
| LogReg (texto + estructurado) | Por defecto | 0,860 | 0,860 | 0,860 | 0,922 | 0,991 |
| LogReg (texto + estructurado) | Ajustado (OOF) | 0,898 | 0,825 | 0,860 | 0,922 | 0,991 |
| **LinearSVC (texto + estructurado)** | **Por defecto** | **0,907** | 0,860 | **0,883** | **0,930** | **0,992** |
| LinearSVC (texto + estructurado) | Ajustado (OOF) | 0,821 | 0,883 | 0,851 | 0,930 | 0,992 |
 
*PR-AUC se calcula como precisión promedio (`average_precision_score`), no por integración trapezoidal. El umbral ajustado maximiza el F1 en OOF; no maximiza el recall de forma explícita.*
 
**Matrices de confusión de LinearSVC en prueba:**
 
| Umbral | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Por defecto (mejor F1) | 147 | 24 | 15 | 3 334 |
| Ajustado OOF (`resumen.json`) | 151 | 20 | 33 | 3 316 |
 
**Lectura de resultados.** Con el umbral por defecto, LinearSVC detecta 147 de 171 fraudes con solo 15 falsas alarmas (F1 = 0,883). El umbral ajustado recupera 4 fraudes adicionales (151) a costa de 18 falsas alarmas más (33) y de un F1 menor (0,851). Cuál conviene depende de la capacidad de moderación humana y del costo relativo de cada error: un falso positivo perjudica a una empresa legítima y un falso negativo expone a personas reales.
 
### Componente no supervisado
 
Se usó k = 11 sobre TF-IDF + SVD (100 componentes, que conservan 27,9 % de la varianza). K-Means y Ward se compararon en la misma muestra aleatoria de hasta 6 000 documentos.
 
| Método | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI vs. etiqueta | NMI vs. etiqueta |
| --- | --- | --- | --- | --- | --- |
| K-Means | 0,121 | 2,946 | 219,0 | −0,004 | 0,024 |
| Ward | 0,107 | 2,980 | 189,6 | −0,010 | 0,019 |
 
**Sobre la elección de k.** La mejor silhouette del rango completo (k = 2 a 15) se obtiene con k = 2 (0,141), una partición demasiado general para describir temas laborales. Se fijó de antemano un rango de granularidad temática (k = 4 a 12) y, dentro de él, la silhouette selecciona k = 11 (0,110). Entre k = 8 y k = 12 las diferencias son menores que el ruido de la muestra, por lo que k = 11 se mantiene porque produce temas interpretables, no porque las métricas lo señalen como óptimo.
 
**Hallazgos.**
 
- **Separación débil.** Las silhouettes bajas (≈ 0,11–0,12) indican grupos poco separados, algo esperable en texto de alta dimensión. La estabilidad de K-Means entre semillas es moderada (ARI medio = 0,578).
- **Los grupos no reproducen la etiqueta, pero sí concentran el riesgo.** El ARI y el NMI cercanos a cero muestran que los grupos no coinciden con la partición legítima/fraude, lo esperable porque se formaron solo con el texto. Aun así, los grupos **1** (oficina y administración: *office, duties, accounting*) y **6** (servicios, atención al cliente y trabajo en casa: *service, home, care, customer*) reúnen el **30,1 %** de las ofertas y el **78,2 %** de los fraudes (669 de 856; tasas de fraude de 16,8 % y 10,1 %, frente a 4,86 % global).
- **Temas sin fraude etiquetado.** Los grupos de enseñanza en el extranjero (724 ofertas), aprendizaje/*apprenticeship* (402) y el grupo 2 (540) no contienen ningún fraude etiquetado. Ausencia de fraude etiquetado no garantiza que las ofertas nuevas de esos temas sean legítimas.
- **Tópico de alto riesgo (NMF).** El tópico de remodelación y mejoras del hogar (*home, siding, remodeling, homeowners*) tiene 20,8 % de fraude (654 ofertas, 4,3 veces la tasa global).
- **Ausencia de logo no implica fraude.** El grupo 9 (listados con redirección a otros sitios de empleo) tiene 99,8 % de ofertas sin logo y solo 1 fraude entre 626.
- **Isolation Forest.** AP = 0,097 (azar: 0,049) y ROC-AUC = 0,70: el fraude es solo moderadamente atípico en el espacio textual.
### Comparación de enfoques
 
| Aspecto | Supervisado (LinearSVC) | No supervisado (K-Means / Ward) |
| --- | --- | --- |
| Pregunta que responde | ¿Esta oferta es fraudulenta? | ¿Qué tipos de ofertas existen? |
| Uso de etiquetas | Entrenamiento y evaluación | Solo para interpretar los grupos |
| Desempeño frente al fraude | F1 = 0,883; PR-AUC = 0,930 | 78,2 % del fraude en 2 de 11 grupos |
| Principal limitación | Requiere etiquetas confiables | Separación débil (silhouette ≈ 0,12) |
| Uso recomendado | Clasificador principal | Contexto para priorizar y explicar |
 
**Recomendación.** Usar LinearSVC como clasificador principal con revisión humana de las alertas, y emplear los grupos y tópicos como herramienta de análisis y priorización, no como detector.
 
### Limitaciones
 
- Las etiquetas reflejan solo el fraude ya detectado; puede haber fraude no etiquetado en el conjunto.
- El lenguaje de las estafas cambia con el tiempo, por lo que el modelo requeriría reentrenamiento periódico.
- SVD conserva solo 27,9 % de la varianza; t-SNE y UMAP sirven para visualizar, pero deforman distancias y no deben leerse como separación real.
- Ward se calcula sobre una muestra por su costo de memoria O(n²), y la estabilidad medida (ARI = 0,578) no cubre toda la cadena TF-IDF/SVD.
- Con 171 fraudes en prueba, una diferencia de 4 detecciones es pequeña y debe tomarse con cautela.
## Fuentes
 
- Dataset: [Real or Fake Job Posting Prediction, Kaggle (shivamb)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction). SHA-256 del CSV en `data/README.md`.