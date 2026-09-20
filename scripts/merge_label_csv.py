#!/usr/bin/env python3
"""Concatène plusieurs CSV de labels en gardant les colonnes du premier.

Sert à mélanger les domaines: images entières et crops de pièces alimentent le
même classifieur, et `photo_id` reste la clé de regroupement des splits — une
photo et les crops qui en sortent ne doivent jamais se retrouver de part et
d'autre d'un fold.

    python scripts/merge_label_csv.py dataset/damage_derived.csv \
           dataset/damage_crops.csv --out dataset/damage_mixed.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="CSV à concaténer (le 1er fixe les colonnes)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    columns: list[str] | None = None
    for path in args.inputs:
        chunk = list(csv.DictReader(Path(path).open(encoding="utf-8")))
        if not chunk:
            print(f"  {path}: vide, ignoré")
            continue
        if columns is None:
            columns = list(chunk[0])
        missing = set(columns) - set(chunk[0])
        if missing:
            raise SystemExit(f"{path}: colonnes manquantes {sorted(missing)}")
        rows += [{c: r[c] for c in columns} for r in chunk]
        print(f"  {path}: {len(chunk)} lignes")

    if not rows or columns is None:
        raise SystemExit("Aucune ligne à écrire.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{out} — {len(rows)} lignes, {len({r['photo_id'] for r in rows})} photos")
    for col in ("damage", "severity", "source"):
        if col in columns:
            counts = collections.Counter(r[col] for r in rows)
            print(f"  {col:9}", dict(counts.most_common()))


if __name__ == "__main__":
    main()
