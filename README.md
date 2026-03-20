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
- Entrenamiento centralizado vs Federated Learning (FedAvg)
- Evaluación con métricas de regresión

---

##  Estructura del repositorio (rellenamos cuando tengamos, la siguiente estructura es solo una idea)
SistemaMicrored-Reto03-Grupo05
┣ data/
┃ ┣ pymgrid_data/          # Datos para simulación (Objetivo 1)
┃ ┗ pv_modules/            # CSVs de módulos PV (Objetivo 2)
┣ src/
┃ ┣ rl/                    # Algoritmos de Aprendizaje por Refuerzo
┃ ┣ fl/                    # Federated Learning
┃ ┣ ml/                    # Modelos de predicción Pmp
┃ ┗ utils/                 # Funciones auxiliares
┣ notebooks/
┃ ┣ RL_experiments.ipynb
┃ ┣ PV_prediction.ipynb
┃ ┗ FederatedLearning.ipynb
┣ tests/
┃ ┗ pytest_regression/       # Oráculos de regresión
┣ Dockerfile
┣ requirements.txt
┣ README.md
┗ LICENSE


---

## 🧪 Metodología

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
- Modelos de regresión para Pmp  
- Federated Learning con FedAvg  
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


---

##  Cómo ejecutar el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/palvcam/Reto3_MicroRedes.git
cd <repo>


