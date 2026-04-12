# Sistema de Gestión de una Microred – Reto03 – Grupo05

##  Descripción del proyecto
Este proyecto forma parte del **Reto 03** del Máster en Inteligencia Artificial Aplicada de Mondragon Unibertsitatea.  
El objetivo es desarrollar un sistema inteligente capaz de **optimizar el coste operacional de una microred eléctrica** y **predecir la eficiencia de módulos fotovoltaicos (PV)** mediante técnicas de:

- Aprendizaje por Refuerzo (RL)
- Aprendizaje Automático (ML)
- Federated Learning
- Gestión de proyectos de IA

> “Desarrollar modelos con los que optimizar el coste operacional de una microred eléctrica. Así mismo se analizará y predecirá la eficiencia de los modelos PV mediante variables ambientales y operativas.”  


---

##  Objetivos del proyecto

###  Objetivo 1 — Optimización del coste de operación de la microred
- Simulación de microred con **pymgrid**
- Definición de estados, acciones y recompensas
- Implementación de algoritmos RL (Q-Learning, PPO)
- Modificación del entorno (batería, coste horario, etc.)

###  Objetivo 2 — Predicción de la potencia máxima (Pmp) de módulos PV
- Uso de dataset con **33 módulos PV en 3 localizaciones**
- Entrenamiento centralizado vs Federated Learning
- Evaluación con métricas de regresión

---

##  Estructura del repositorio 
```txt
Reto03_MicroRedes
┣ OBJETIVO1/
┃ ┣ 2_Q-learning/                # Enfoque tabular: entorno discreto y agentes Q-Learning
┃ ┃ ┣ images/
┃ ┃ ┣ Q-learning/                # Contiene mejor q-table del hp search y base de datos de búsqueda de mejores hiperparámetros
┃ ┃ ┣ Resultados_Entrenamiento_Q/# q-tables de las 30 ejecuciones independientes de entrenamiento
┃ ┃ ┣ 2_q-learning.ipynb         # Entrenamiento y evaluación del agente tabular
┃ ┃ ┣ custom_env_tabular2.py     # Wrapper del entorno pymgrid discretizado
┃ ┃ ┗ Decidir_bins.ipynb         # Análisis para la discretización de variables
┃ ┣ data/                        # Datos de series temporales (load, PV, precios e-sios)
┃ ┣ conversion_pv_a_kw.py        # Conversion de irradancia a kW (.csv de PV)                 
┃ ┗ PPO/                         # Enfoque Deep-RL: entorno continuo y agentes PPO
┃   ┣ Analisis_Resultados_v2.ipynb  # Evaluación visual y métricas del modelo final
┃   ┣ custom_env_continuous_v2.py # Wrapper del entorno pymgrid continuo
┃   ┣ Estimacion_C_continuous.py # Estimar constante normalizacion
┃   ┗ training_PPO_v1.py     # Script de entrenamiento principal PPO
┃ 
┣ OBJETIVO2/
┃ ┣ Evaluacion/                    # Evaluación de incertidumbre del modelo de aprendizaje federado
┃ ┣ federated-docker-avg/                    # Federated Learning Average 
┃ ┣ federated-docker-prox/                    # Federated Learning Proximal
┃ ┣ federated-docker/                    # Federated Learning Proximal con hyperparameter tuning (definitivo para despliegue)
┃ ┗ Baseline.ipynb/                 # Análisis de la distribución de los datos y comparación de modelos de ML 
┗ README.md
```


---

## Metodología

###  Gestión de proyectos de IA
- Definición de requisitos y especificaciones  
- Diseño de arquitectura  
- Testeo con oráculos de regresión  
- Dockerización y despliegue en AWS  


###  Aprendizaje por Refuerzo
- Creación del environment con pymgrid  
- Discretización o acciones continuas (Gymnasium + SB3)  
- Entrenamiento y comparación de modelos  


###  Aprendizaje Automático II
- Modelos de predicción para Pmp  
- Federated Learning con FedProx + hyperparameter tuning 
- Análisis de heterogeneidad (non-IID)  


---

##  Datos utilizados

###  Objetivo 1 — Microred simulada
Tres CSVs, informado de tres diferentes variables para cada hora:
- Precio de la luz (€/kWh)
- PV: producción fotovoltaica
- Load: demanda horaria

###  Objetivo 2 — Dataset PV
Cada CSV contiene:
- Irradiancia POA  
- Temperatura del módulo  
- Pmp (W)  
- Humedad, presión, precipitación  
- DNI, GHI, DHI
- Feature Engineering  

---

##  Cómo ejecutar el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/palvcam/Reto3_MicroRedes.git
cd <repo>

```
### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt


```

### 4. Ejecutar notebooks/Scripts

OBJETIVO 1
Se distingue en dos etapas:
1. Entorno Tabular (se desarrolla por completo en la carpeta 2_Q-learning):
- Decidir_bins.ipynb decide los bins de discretización para definir los estados discretos (no necesario ejecutarlo ya que los resultados ya están implementados en 2_q-learning.ipynb).
- custom_env_tabular2.py contiene el entorno creado (tampoco hay que ejecutarlo)
- 2_q-learning.ipynb recoge todo el proceso de instanciación del simulador de la red, el hp search, el entrenamiento definitivo y las visualizaciones
- Ejecutar Analisis_Resultados.ipynb para un análisis estadístico

2. Deep RL (se desarrolla por completo en la carpeta PPO):
- El entorno continuo creado se encuentra en el archivo custom_env_continuous_v2.py
- Establecer las características físicas de simulación en pymgrid.
- Ejecutar Estimacion_C_continuous.py respectando esas caracteristicas para estimar C.
- Para proceder al entenamiento y optimización de HP se ejecutará el script training_PPO_v1.py, adaptando los parametros y variables según el interés.
- Se emplea el notebook Analisis_resultados_PPO_v1.ipynb para el analisis de los resultados obtenidos.
  

OBJETIVO 2
- Ejecutar el notebook Baseline.ipynb
- Ejecutar el notebook de evaluacion_incertidumbre.ipynb para ajustar el umbral del guardrail
- Despliegue en AWS + local
  1. Crear instancia de EC2 y una carpeta en esa instancia para guardar los resultados del aprendizaje federado
  2. Crear los contenedores de docker para el client y el server
  3. Guardar la imagen del server y subir la imagen a EC2
  4. Cargar la imagen del server en EC2 y lanzar el server 
  5. Lanzar los clientes en local, cada uno desde una terminal diferente
  6. Desde la carpeta de federated-docker/server en local recuperar las imágenes de los resultados de EC2

## Tests?




