"""Fixed, label-free structural alert counts for the tdi_v7 study.

These are broad bioactivation hypotheses, not experimentally established TDI
mechanisms or a causal assay. Overlapping alerts are deliberately retained.
Run ``python -B tdi_v7/alerts.py`` for positive/negative checks and an audit;
add ``--cache`` for arrays in original classification/test CSV row order.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent

# Each tuple is (column name, SMARTS). Recursive single-atom queries count
# reactive centers once, independent of the number of matching substituents.
# N-dealkylation requires at least one attached sp3 carbon carrying hydrogen;
# amide, amidine, sulfonamide, and phosphorylamide nitrogens are excluded.
_N_EXCLUDE = "!$(N[C,S,P](=[O,S,N]));!$(NS(=O)=O)"
ALERTS = [
    ("terminal_alkyne", "[C;H1]#[C]"),
    ("internal_alkyne", "[C;H0]#[C;H0]"),
    ("furan", "[o]1[c][c][c][c]1"),
    ("thiophene", "[s]1[c][c][c][c]1"),
    ("methylenedioxyphenyl", "[CH2]1[O][c]2[c][c][c][c][c]2[O]1"),
    ("secondary_n_dealkylatable",
     f"[N;{_N_EXCLUDE};$([N]-[C;X4;!H0]);$([N;X3;H1;+0]),$([N;X4;H2;+1])]"),
    ("tertiary_n_dealkylatable",
     f"[N;{_N_EXCLUDE};$([N]-[C;X4;!H0]);$([N;X3;H0;+0]),$([N;X4;H1;+1])]"),
    ("hydrazine", f"[N;X3,X4;{_N_EXCLUDE}]-[N;X3,X4;{_N_EXCLUDE}]"),
    ("cyclopropylamine", f"[N;X3,X4;{_N_EXCLUDE}]-[C]1-[C]-[C]-1"),
    ("allylic_carbon_with_h", "[C;X4;!H0;$([C]-[C;X3]=[C;X3])]"),
    ("thioether", "[S;X2;+0;!$(S[C](=[O,S,N]))]([#6])[#6]"),
    ("arylamine_aniline", f"[N;X3,X4;{_N_EXCLUDE};$([N]-[c])]"),
    ("phenol_or_phenolate", "[O;H1,-1;$([O]-[c])]"),
    ("catechol", "[O;H1,-1]-[c]1[c]([O;H1,-1])[c][c][c][c]1"),
    ("alpha_beta_unsaturated_carbonyl", "[C;X3]=[C;X3]-[C;X3]=[O]"),
    ("propargylamine", f"[N;X3,X4;{_N_EXCLUDE};$([N]-[C;X4]-[C]#[C])]"),
]
PATTERNS = tuple(Chem.MolFromSmarts(smarts) for _, smarts in ALERTS)
assert all(pattern is not None for pattern in PATTERNS), "Invalid fixed SMARTS"


def build_alerts(smiles) -> np.ndarray:
    """Return shape (n_molecules, 16) float32 counts in input row order.

    ``smiles`` is an iterable of SMILES strings (a single string gives one row).
    Invalid or empty molecules raise ValueError instead of silently imputing.
    No fitting, label access, standardization, or scaffold information is used.
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    smiles = list(smiles)
    result = np.empty((len(smiles), len(ALERTS)), dtype=np.float32)
    for row, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid or empty SMILES at row {row}: {smi!r}")
        for col, pattern in enumerate(PATTERNS):
            result[row, col] = len(mol.GetSubstructMatches(pattern, uniquify=True))
    return result


