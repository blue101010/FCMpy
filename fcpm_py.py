# Installation: pip install fcmpy
from collections import OrderedDict

from fcmpy import ExpertFcm
import numpy as np
import pandas as pd

# Creation de l'objet FCM
fcm = ExpertFcm()

# Definition de l'univers du discours [-1, 1]
fcm.universe = np.arange(-1, 1.001, 0.001)

# Definition des termes linguistiques
# Format: [min, centre, max] pour fonctions triangulaires
fcm.linguistic_terms = {
    '-VH': [-1, -1, -0.75],      # Tres haut negatif
    '-H': [-1, -0.75, -0.50],    # Haut negatif
    '-M': [-0.75, -0.5, -0.25],  # Moyen negatif
    '-L': [-0.5, -0.25, 0],      # Faible negatif
    'No': [-0.01, 0, 0.01],      # Pas de causalite
    '+L': [0, 0.25, 0.50],       # Faible positif
    '+M': [0.25, 0.5, 0.75],     # Moyen positif
    '+H': [0.5, 0.75, 1],        # Haut positif
    '+VH': [0.75, 1, 1]          # Tres haut positif
}

# Generation des fonctions d'appartenance
fcm.fuzzy_membership = fcm.automf(method='trimf')

# Donnees des experts (2 experts)
raw_expert_ratings = {
    ('Stress', 'Sommeil'): ['-H', '-M'],
    ('Stress', 'Cafe'): ['+M', '+L'],
    ('Sommeil', 'Productivite'): ['+H', '+VH'],
    ('Cafe', 'Sommeil'): ['-M', '-L'],
    ('Cafe', 'Productivite'): ['+L', '+M']
}


def expand_simple_inputs(ratings: dict[tuple[str, str], list[str]]) -> OrderedDict:
    """Convert a compact mapping of ratings into ExpertFcm's expected structure."""

    term_columns = [term.lower() for term in fcm.linguistic_terms]

    def make_row(source: str, target: str, label: str) -> dict[str, object]:
        row = {col: 0 for col in term_columns}
        normalized = label.strip().lower()
        if normalized not in row:
            raise KeyError(f"The linguistic term '{label}' is not defined in fcm.linguistic_terms")
        row[normalized] = 1
        row['From'] = source
        row['To'] = target
        return row

    n_experts = max(len(values) for values in ratings.values())
    ordered = OrderedDict()
    for expert_idx in range(n_experts):
        rows = []
        for (source, target), values in ratings.items():
            if expert_idx >= len(values):
                continue
            rows.append(make_row(source, target, values[expert_idx]))
        ordered[f'Expert{expert_idx}'] = pd.DataFrame(rows)
    return ordered


structured_data = expand_simple_inputs(raw_expert_ratings)

# Construction de la matrice de poids
weight_matrix = fcm.build(
    data=structured_data,
    implication_method='Mamdani',
    aggregation_method='fMax',
    defuzz_method='centroid'
)
print("Matrice de poids generee:")
print(weight_matrix)