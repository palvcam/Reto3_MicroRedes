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
- Implementación de algoritmos RL (Q-Learning, A2C, PPO…)
- Modificación del entorno (batería, coste horario, etc.)

###  Objetivo 2 — Predicción de la potencia máxima (Pmp) de módulos PV
- Uso de dataset con **33 módulos PV en 3 localizaciones**
- Entrenamiento centralizado vs Federated Learning
- Evaluación con métricas de regresión

---

##  Estructura del repositorio 
*(rellenamos cuando tengamos, la siguiente estructura es solo una propuesta)*
```txt
Reto03_MicroRedes
┣ OBJETIVO1/
┃ ┣ /          # Datos para simulación (Objetivo 1)
┃ ┗ /            # CSVs de módulos PV (Objetivo 2)
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
- PV, batería, grid, load  
- Tarifas horarias  
- Outages  
- Costes de generación  


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




