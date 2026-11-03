"""Label-free additional SMARTS counts and fixed aggregates for the v8 study."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from rdkit import Chem
from alerts import ALERTS as OLD_ALERTS, build_alerts as build_old
import numpy as np

# The v7 panel already covers methylenedioxyphenyl, furan, thiophene,
# alkynes, cyclopropylamine, hydrazine and dealkylatable 2°/3° amines.
# These additions represent adjacent reactive-metabolite/MI-complex motifs.
EXTRA = [
    ('aryl_methoxy', '[c]-[O]-[CH3]'),
    ('benzylic_ch', '[C;X4;!H0]-[c]'),
    ('pyrrole', '[nH]1[c][c][c][c]1'),
    ('nitro_aromatic', '[c]-[N+](=[O])[O-]'),
    ('quinone', '[O]=[C]1[C]=[C][C](=[O])[C]=[C]1'),
    ('hydroquinone', '[O;H1,-1]-[c]1[c][c][c]([O;H1,-1])[c][c]1'),
    ('epoxide', '[O]1[C][C]1'),
    ('thiourea', '[N]-[C](=[S])-[N]'),
    ('thioamide', '[N]-[C](=[S])-[#6]'),
    ('hydroxylamine', '[N]-[O;H1]'),
    ('n_nitroso', '[N]-[N]=[O]'),
    ('imidazole', '[n]1[c,n][n][c,n][c,n]1'),
]
PATTERNS = [Chem.MolFromSmarts(s) for _, s in EXTRA]
assert all(p is not None for p in PATTERNS)
NAMES = [name for name, _ in OLD_ALERTS + EXTRA]
AGGREGATES = [
    'total_alert_count', 'distinct_alert_count', 'amine_plus_methylenedioxy',
    'amine_plus_alkyne', 'amine_plus_furan_thiophene',
    'amine_plus_aryl_methoxy',
]


def build(smiles):
    smiles = list(smiles)
    old = build_old(smiles)
    extra = np.empty((len(smiles), len(EXTRA)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not mol.GetNumAtoms():
            raise ValueError(f'invalid SMILES row {i}')
        for j, pattern in enumerate(PATTERNS):
            extra[i, j] = len(mol.GetSubstructMatches(pattern, uniquify=True))
    raw = np.column_stack([old, extra])
    ix = {name: i for i, name in enumerate(NAMES)}
    amine = (raw[:, ix['secondary_n_dealkylatable']] +
             raw[:, ix['tertiary_n_dealkylatable']]) > 0
    pairs = [
        raw[:, ix['methylenedioxyphenyl']] > 0,
        (raw[:, ix['terminal_alkyne']] + raw[:, ix['internal_alkyne']]) > 0,
        (raw[:, ix['furan']] + raw[:, ix['thiophene']]) > 0,
        raw[:, ix['aryl_methoxy']] > 0,
    ]
    agg = np.column_stack([raw.sum(axis=1), (raw > 0).sum(axis=1),
                           *[(amine & x).astype(np.float32) for x in pairs]])
    return np.column_stack([raw, agg]).astype(np.float32)


