"""Le split groupé doit empêcher qu'un même véhicule soit des deux côtés."""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def _frame():
    rows = []
    for vid in range(20):
        for shot, view in enumerate(["front", "rear", "left", "right"]):
            rows.append({"image": f"V{vid}_{shot}.jpg", "view": view, "vehicle_id": f"V{vid}"})
    return pd.DataFrame(rows)


def test_grouped_split_has_no_vehicle_overlap():
    df = _frame()
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    for tr, va in splitter.split(df, df["view"], df["vehicle_id"]):
        overlap = set(df.iloc[tr]["vehicle_id"]) & set(df.iloc[va]["vehicle_id"])
        assert not overlap, f"fuite: {overlap}"


def test_ungrouped_split_leaks_vehicles():
    # Documente le comportement de l'ancien code: la fuite est systématique,
    # ce qui gonflait les métriques rapportées.
    df = _frame()
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    leaks = [
        set(df.iloc[tr]["vehicle_id"]) & set(df.iloc[va]["vehicle_id"])
        for tr, va in splitter.split(df, df["view"])
    ]
    assert all(leaks), "le split par image devrait produire des chevauchements"
