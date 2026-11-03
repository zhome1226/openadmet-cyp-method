"""Deterministic molecular features; no label-derived preprocessing."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdReducedGraphs
from rdkit.Chem.Scaffolds import MurckoScaffold

from expanded_alerts import build as build_alerts

REG_DESC = (
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.RingCount,
    Descriptors.FractionCSP3,
)
TDI_DESC = list(Descriptors._descList)
REG_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
TDI_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def identity(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
    )


def _bits(fp, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, out)
    return out


def _descriptors(mol, defs) -> np.ndarray:
    values = []
    for descriptor in defs:
        fn = descriptor[1] if isinstance(descriptor, tuple) else descriptor
        try:
            val = float(fn(mol))
        except Exception:
            val = 0.0
        values.append(val if np.isfinite(val) else 0.0)
    return np.clip(np.asarray(values, dtype=np.float32), -1e6, 1e6)


def build(smiles, kind: str = "regression") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return features, canonical identities, and Murcko scaffolds in input order."""
    smiles = [str(s) for s in smiles]
    feats, canon, scaffold = [], [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES: {s!r}")
        c, g = identity(s)
        canon.append(c)
        scaffold.append(g)
        if kind == "regression":
            blocks = [_bits(REG_MORGAN.GetFingerprint(mol), 256), _descriptors(mol, REG_DESC)]
        elif kind == "tdi":
            blocks = [
                _bits(TDI_MORGAN.GetFingerprint(mol), 1024),
                _bits(pyAvalonTools.GetAvalonFP(mol, nBits=512), 512),
                _bits(Chem.RDKFingerprint(mol, fpSize=512), 512),
                _descriptors(mol, TDI_DESC),
                np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32),
            ]
        else:
            raise ValueError(kind)
        feats.append(np.concatenate(blocks))
    if kind == "tdi" and smiles:
        matrix = np.column_stack((np.stack(feats), build_alerts(smiles)))
    elif feats:
        matrix = np.stack(feats)
    else:
        matrix = np.empty((0, 264 if kind == "regression" else 2614), np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite feature")
    return matrix.astype(np.float32), np.asarray(canon), np.asarray(scaffold)
