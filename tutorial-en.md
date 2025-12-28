# FCMpy Tutorial (EN)

This tutorial introduces FCMpy to newcomers to fuzzy cognitive maps. The goal is to install the library, build a first model, and run a simple simulation with interventions.

## 1. Why use FCMpy?

A fuzzy cognitive map (FCM) represents a complex system as concepts (nodes) connected by weighted causalities. FCMpy provides tools to:

- convert qualitative judgments into weight matrices;
- run dynamic simulations to observe how concepts evolve;
- test interventions ("what-if" scenarios) and explore learning algorithms.

## 2. Prerequisites and installation

FCMpy works with Python 3.9 or newer (tested up to 3.14). Two approaches are proposed:

### 2.1 Via PyPI (mind package compatibilities)

```bash
pip install fcmpy
```

Add optional dependencies if needed:

- `pip install "fcmpy[ml]"` -> scikit-learn algorithms;
- `pip install "fcmpy[viz]"` -> matplotlib/seaborn charts;
- `pip install "fcmpy[ml-tf]"` -> TensorFlow-based ELTCN classifier.

### 2.2 Via the provided launcher (Python 3.14 recommended)

At the repository root, run:

```bash
python launcher.py --extras ml viz
```

The script creates `.venv`, installs the dependencies listed in `requirements.txt`, adds the package in editable mode, then launches `fcpm_py.py`.

## 3. First steps: generating a weight matrix

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

Prepare the expert evaluations. The utility function below converts a simple dictionary into the structure required by `ExpertFcm.build`.

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
	('Stress', 'Sleep'): ['-H', '-M'],
	('Stress', 'Coffee'): ['+M', '+L'],
	('Sleep', 'Productivity'): ['+H', '+VH'],
	('Coffee', 'Sleep'): ['-M', '-L'],
	('Coffee', 'Productivity'): ['+L', '+M'],
}

expert_data = ratings_to_ordereddict(fcm.linguistic_terms, raw)
weights = fcm.build(data=expert_data)
print(weights)
```

## 4. Simulate a scenario

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

## 5. Test an intervention

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

inter.add_intervention('campaign', impact={'C1': -0.2}, effectiveness=1)
inter.test_intervention('campaign')
print(inter.equilibriums)
```

## 6. Additional resources

- English documentation: <https://maxiuw.github.io/fcmpyhtml>
- Reference paper: <https://arxiv.org/abs/2111.12749>
- Tutorial unit tests: `unittests/`


