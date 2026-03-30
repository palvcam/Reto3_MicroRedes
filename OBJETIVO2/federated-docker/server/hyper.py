# FedEx oficial del repo mkhodak/FedEx
import numpy as np
from scipy.special import logsumexp
from copy import deepcopy
from itertools import product
from numpy.linalg import norm


def discounted_mean(trace, factor=1.0):
    """
    Calcula la media ponderada de un historial de valores.
    Los valores más recientes tienen mayor peso.
    
    factor=0.0 → solo usa el valor más reciente
    factor=1.0 → media aritmética simple de todos los valores
    factor=0.9 → media exponencialmente ponderada (valores recientes pesan más)
    """
    weight = factor ** np.flip(np.arange(len(trace)), axis=0) # Crea pesos: factor^(n-1), factor^(n-2), ..., factor^0
    return np.inner(trace, weight) / weight.sum() # Media ponderada normalizada


class FedEx:
    """
    Optimización de hiperparámetros para aprendizaje federado.
    
    Usa gradiente exponenciado (Exponentiated Gradient) para aprender
    qué configuración de hiperparámetros (epochs, mu) funciona mejor,
    sin necesidad de centralizar los datos de los clientes.
    
    La idea central: cada config tiene una probabilidad. Tras cada ronda,
    las configs que mejoraron más el modelo suben de probabilidad y las
    que mejoraron menos bajan. Con suficientes rondas, FedEx converge
    a la mejor config.
    """
    def entropy(self):
        """
        Calcula la entropía de la distribución de probabilidad sobre las configs.
        
        Entropía alta → FedEx está explorando (incertidumbre sobre qué config es mejor)
        Entropía baja → FedEx ha convergido (una config tiene mucha más probabilidad)
        """
        entropy = 0.0
        for probs in product(*(theta[theta>0.0] for theta in self._theta)): # itera sobre todas las combinaciones posibles de probabilidades
            prob = np.prod(probs) # Probabilidad de la combinación
            entropy -= prob * np.log(prob) # Contribución a la entropía: -p*log(p)
        return entropy

    def mle(self):
        """
        Devuelve la probabilidad máxima de la mejor configuración.
        
        MLE bajo  → FedEx explorando (las probs están repartidas)
        MLE alto  → FedEx convergió (una config domina)
        """
        return np.prod([theta.max() for theta in self._theta])

    def __init__(
                 self, 
                 server, 
                 configs, # Lista de configs
                 eta0='auto', # Tamaño base del paso del gradiente exponenciado
                 sched='auto', # Schedule del learning rate
                 cutoff=0.0, # Entropía mínima para parar automáticamente
                 baseline=0.0, # Factor de descuento del historial para el baseline
                 diff=False, # Si True usa diferencia before/after; si False usa valor absoluto
                 ):
        '''
        Parámetros:
            server: Objeto que implementa dos métodos, 'communication_round' y 'full_evaluation',
                    que reciben como único argumento 'get_config', una función sin parámetros
                    que devuelve una configuración de la lista 'configs'.
                    - 'communication_round' selecciona un grupo de clientes, asigna una config
                    a cada uno usando 'get_config', y ejecuta el entrenamiento local con esa config.
                    Luego agrega los modelos locales para dar un paso de entrenamiento y devuelve
                    tres listas o arrays: el error de validación de cada cliente ANTES del
                    entrenamiento local, el error de validación DESPUÉS del entrenamiento local,
                    y el peso de cada cliente (por ejemplo, el tamaño de su conjunto de validación).
                    - 'full_evaluation' asigna una config a cada cliente usando 'get_config' y ejecuta
                    el entrenamiento local con esa config. Devuelve tres listas o arrays: el error
                    de test de cada cliente ANTES del entrenamiento local, el error de test DESPUÉS
                    del entrenamiento local, y el peso de cada cliente (por ejemplo, el tamaño de
                    su conjunto de test).
            configs: lista de configuraciones usadas para entrenamiento y evaluación por 'server',
                    O diccionario de pares (string, lista) que define un grid de configuraciones.
            eta0: tamaño base del paso del gradiente exponenciado. Si es 'auto' usa sqrt(2*log(k))
                donde k es el número de configuraciones.
            sched: schedule del learning rate para el gradiente exponenciado:
                    - 'adaptive': usa eta0 / sqrt(suma de normas al cuadrado del gradiente)
                    - 'aggressive': usa eta0 / norma infinito del gradiente
                    - 'auto': usa eta0 / sqrt(t) donde t es el número de rondas
                    - 'constant': usa eta0 fijo
                    - 'scale': usa eta0 * sqrt(2 * log(k))
            cutoff: nivel de entropía por debajo del cual se deja de actualizar las probabilidades
                    y se usa directamente la mejor configuración (MLE)
            baseline: factor de descuento del historial para calcular el baseline.
                    0.0 = solo usa la ronda más reciente, 1.0 = media de todas las rondas
            diff: si True usa la diferencia de rendimiento antes/después del entrenamiento local;
                si False usa el rendimiento absoluto después del entrenamiento
        '''

        self._server = server
        self._configs = configs
        self._grid = [] if type(configs) == list else sorted(configs.keys())
        # Tamaño del grid para cada hiperparámetro
        sizes = [len(configs[param]) for param in self._grid] if self._grid else [len(configs)]
        self._eta0 = [np.sqrt(2.0 * np.log(size)) if eta0 == 'auto' else eta0 for size in sizes]
        self._sched = sched
        self._cutoff = cutoff
        self._baseline = baseline
        self._diff = diff
        # Inicializar distribución uniforme sobre todas las configs
        # z son los log-probabilidades (log-weights) — inicialmente todos iguales: -log(k)
        # Esto da probabilidad uniforme: exp(-log(k)) = 1/k para cada config
        self._z = [np.full(size, -np.log(size)) for size in sizes]
        self._theta = [np.exp(z) for z in self._z]
        self._store = [0.0 for _ in sizes]
        self._stopped = False
        # Historial de métricas para análisis
        # 'global' = loss antes del entrenamiento local
        # 'refine' = loss después del entrenamiento local
        # 'entropy' y 'mle' inicializados con valor de la distribución uniforme inicial
        self._trace = {'global': [], 'refine': [], 'entropy': [self.entropy()], 'mle': [self.mle()]}

    def stop(self):
        """Fuerza la parada de FedEx — a partir de aquí siempre usa la mejor config"""
        self._stopped = True

    def sample(self, mle=False, _index=[]):
        """
        Muestrea una configuración según la distribución de probabilidad actual.
        
        mle=False → muestrea aleatoriamente según las probabilidades (exploración)
        mle=True  → devuelve la config con mayor probabilidad (explotación)
        """
        # Si ya convergió o se fuerza MLE, devolver la mejor config
        if mle or self._stopped:
            if self._grid:
                # Para cada hiperparámetro, devolver el valor con mayor probabilidad
                return {param: self._configs[param][theta.argmax()] 
                        for theta, param in zip(self._theta, self._grid)}
            return self._configs[self._theta[0].argmax()]
        # Muestrear un índice para cada hiperparámetro según su distribución
        # np.random.choice muestrea con probabilidades theta
        _index.append([np.random.choice(len(theta), p=theta) for theta in self._theta])

        # Devolver la config correspondiente a los índices muestreados
        if self._grid:
            return {param: self._configs[param][i] for i, param in zip(_index[-1], self._grid)}
        return self._configs[_index[-1][0]]

    def settings(self):
        """Devuelve la configuración actual de FedEx"""
        output = {'configs': deepcopy(self._configs)}
        output['eta0'], output['sched'] = self._eta0, self._sched
        output['cutoff'], output['baseline'] = self._cutoff, self._baseline 
        if self._trace['refine']:
            output['theta'] = self.theta() # Incluir probabilidades actuales si ya entrenó
        return output

    def step(self):
        """
        Ejecuta una ronda federada y actualiza las probabilidades de cada config.
        
        Es el núcleo de FedEx:
        1. Asigna configs a los clientes y entrena (communication_round)
        2. Calcula el gradiente exponenciado basado en la mejora de cada config
        3. Actualiza las probabilidades: configs que mejoraron más suben de probabilidad
        """

        index = [] # Acumulará los índices de configs asignados a cada cliente
        # Ejecutar una ronda federada completa
        before, after, weight = self._server.communication_round(lambda: self.sample(_index=index))        
        before, after = np.array(before), np.array(after) # Loss de validación antes/después del entrenamiento local
        weight = np.array(weight, dtype=np.float64) / sum(weight) # Normalizar pesos

        # Calcular baseline: referencia para medir si una config es mejor de lo normal
        if self._trace['refine']:
            trace = self.trace('refine') # Historial de losses después del entrenamiento
            if self._diff:
                trace -= self.trace('global') # Si diff=True, usar mejora relativa
            # Media ponderada del historial como baseline
            baseline = discounted_mean(trace, self._baseline)
        else:
            baseline = 0.0 # Primera ronda: no hay historial, baseline = 0
        # Guardar métricas de esta ronda en el historial
        self._trace['global'].append(np.inner(before, weight))
        self._trace['refine'].append(np.inner(after, weight))
        # Si ningún cliente recibió config, marcar como convergido
        if not index:
            self._trace['entropy'].append(0.0)
            self._trace['mle'].append(1.0)
            return

        # ACTUALIZACIÓN DE PROBABILIDADES (Gradiente Exponenciado)
        for i, (z, theta) in enumerate(zip(self._z, self._theta)):
            grad = np.zeros(len(z)) # Gradiente para este hiperparámetro

            for idx, s, w in zip(index, after-before if self._diff else after, weight):
                grad[idx[i]] += w * (s - baseline) / theta[idx[i]]
            # Calcular el learning rate según el schedule elegido
            if self._sched == 'adaptive':
                # Disminuye según la raíz de la suma acumulada de normas al cuadrado
                self._store[i] += norm(grad, float('inf')) ** 2
                denom = np.sqrt(self._store[i])
            elif self._sched == 'aggressive':
                # Normaliza por la norma actual — pasos más agresivos
                denom = 1.0 if np.all(grad == 0.0) else norm(grad, float('inf'))
            elif self._sched == 'auto':
                # Disminuye como 1/sqrt(t)
                self._store[i] += 1.0
                denom = np.sqrt(self._store[i])
            elif self._sched == 'constant':
                # Learning rate fijo
                denom = 1.0
            elif self._sched == 'scale':
                # Escalado por el tamaño del grid
                denom = 1.0 / np.sqrt(2.0 * np.log(len(grad))) if len(grad) > 1 else float('inf')
            else:
                raise NotImplementedError
            eta = self._eta0[i] / denom # Learning rate final para esta ronda
            z -= eta * grad # Actualizar log-probabilidades: z -= eta * grad
            # Proyectar al simplex usando logsumexp para estabilidad numérica
            # Equivale a normalizar las probabilidades para que sumen 1
            z -= logsumexp(z)
            self._theta[i] = np.exp(z) # Actualizar probabilidades desde log-probabilidades

        self._trace['entropy'].append(self.entropy())
        self._trace['mle'].append(self.mle())
        # Parar automáticamente si la entropía cae por debajo del umbral (convergencia)
        if self._trace['entropy'][-1] < self._cutoff:
            self.stop()

    def test(self, mle=False):
        """
        Evaluación final del modelo con la config encontrada.
        
        Devuelve:
            'global' → loss del modelo global sin fine-tune local
            'refine' → loss después del fine-tune local con la mejor config
        """

        before, after, weight = self._server.full_evaluation(lambda: self.sample(mle=mle))
        return {'global': np.inner(before, weight) / weight.sum(), # Loss global ponderado
                'refine': np.inner(after, weight) / weight.sum()} # Loss refinado ponderado

    def theta(self):
        """Devuelve una copia de las probabilidades actuales de cada config"""
        return deepcopy(self._theta)

    def trace(self, key):
        """
        Devuelve el historial de una métrica a lo largo de las rondas.
        
        key='entropy' → evolución de la entropía (debe bajar si FedEx converge)
        key='global'  → evolución del loss antes del entrenamiento local
        key='refine'  → evolución del loss después del entrenamiento local
        key='mle'     → evolución de la probabilidad máxima (debe subir si converge)
        """
        return np.array(self._trace[key])


def frac(p, q):
    """Utilidad para formatear fracciones como string"""
    return str(p) + '/' + str(q)
