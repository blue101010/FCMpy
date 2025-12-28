# Tutoriel FCMpy (fr)

Ce tutoriel presente FCMpy aux personnes qui decouvrent les cartes
cognitives floues. L objectif est d installer la bibliotheque, de creer
un premier modele et de lancer une simulation simple avec des
interventions.

## 1. Pourquoi utiliser FCMpy ?

Une carte cognitive floue (FCM) represente un systeme complexe sous
forme de concepts (noeuds) relies par des causalites ponderees. FCMpy
fournit des outils pour :

- convertir des jugements qualitatifs en matrices de poids;
- executer des simulations dynamiques afin d observer l evolution des
    concepts;
- tester des interventions (scenarios "what-if") et explorer des
    algorithmes d apprentissage.

## 2. Pre-requis et installation

FCMpy fonctionne avec Python 3.9 ou plus recent (teste jusqu a 3.14).
Deux approches sont proposees :

### 2.1 Via PyPI (attention aux compatibilités de packages)

```bash
pip install fcmpy
```

Ajoutez des dépendances optionnelles si nécessaire :

- `pip install "fcmpy[ml]"` -> algorithmes scikit-learn;
- `pip install "fcmpy[viz]"` -> graphiques matplotlib/seaborn;
- `pip install "fcmpy[ml-tf]"` -> classifier ELTCN basé sur TensorFlow.

### 2.2 Via le lanceur fourni (recommandé python 3.14)

A la racine du depot, executez :

```bash
python launcher.py --extras ml viz
```

Le script cree `.venv`, installe les dependances listees dans
`requirements.txt`, ajoute le paquet en mode editable puis lance
`fcpm_py.py`.

## 3. Premiers pas : generation d une matrice de poids

```python
from fcmpy import ExpertFcm
import numpy as np

fcm = ExpertFcm()

fcm.universe = np.arange(-1, 1.001, 0.001)
fcm.linguistic_terms = {
    '-VH': [-1, -1, -0.75],
    '-H': [-1, -0.75, -0.50],
    '-M': [-0.75, -0.5, -0.25],
    '-L': [-0.5, -0.25, 0],
    'No': [-0.01, 0, 0.01],
    '+L': [0, 0.25, 0.50],
    '+M': [0.25, 0.5, 0.75],
    '+H': [0.5, 0.75, 1],
    '+VH': [0.75, 1, 1],
}

fcm.fuzzy_membership = fcm.automf(method='trimf')
```

Preparez ensuite les evaluations d experts. La fonction utilitaire ci
dessous convertit un dictionnaire simple en structure attendue par
`ExpertFcm.build`.

```python
from collections import OrderedDict
import pandas as pd

def ratings_to_ordereddict(terms, ratings):
    cols = [term.lower() for term in terms]
    output = OrderedDict()
    n_experts = max(len(v) for v in ratings.values())
    for idx in range(n_experts):
        rows = []
        for (src, dst), values in ratings.items():
            if idx >= len(values):
                continue
            row = {col: 0 for col in cols}
            label = values[idx].lower()
            row[label] = 1
            row['From'] = src
            row['To'] = dst
            rows.append(row)
        output[f'Expert{idx}'] = pd.DataFrame(rows)
    return output

raw = {
    ('Stress', 'Sommeil'): ['-H', '-M'],
    ('Stress', 'Cafe'): ['+M', '+L'],
    ('Sommeil', 'Productivite'): ['+H', '+VH'],
    ('Cafe', 'Sommeil'): ['-M', '-L'],
    ('Cafe', 'Productivite'): ['+L', '+M'],
}

expert_data = ratings_to_ordereddict(fcm.linguistic_terms, raw)
weights = fcm.build(data=expert_data)
print(weights)
```

## 4. Simuler un scenario

```python
from fcmpy import FcmSimulator
import pandas as pd

sim = FcmSimulator()
weight_matrix = pd.DataFrame(
    [
        [0.0, 0.0, 0.6],
        [0.1, 0.0, 0.0],
        [0.0, 0.7, 0.0],
    ],
    columns=['C1', 'C2', 'C3'],
    index=['C1', 'C2', 'C3'],
)

init_state = {'C1': 1, 'C2': 0, 'C3': 0}
traj = sim.simulate(
    initial_state=init_state,
    weight_matrix=weight_matrix,
    transfer='sigmoid',
    inference='mKosko',
    thresh=0.001,
    iterations=50,
    l=1,
)
print(traj.tail(1))
```

## 5. Tester une intervention

```python
from fcmpy import FcmIntervention

inter = FcmIntervention(FcmSimulator)
inter.initialize(
    initial_state=init_state,
    weight_matrix=weight_matrix,
    transfer='sigmoid',
    inference='mKosko',
    thresh=0.001,
    iterations=50,
    l=1,
)

inter.add_intervention('campagne', impact={'C1': -0.2}, effectiveness=1)
inter.test_intervention('campagne')
print(inter.equilibriums)
```

## 6. Ressources complementaires

- Documentation anglaise : <https://maxiuw.github.io/fcmpyhtml>
- Article de reference : <https://arxiv.org/abs/2111.12749>
- Tutoriels unit tests : `unittests/`

