#!/usr/bin/env python3
"""Remappe les labels YOLO du dataset source vers la taxonomie V1.

Remplace scripts/clean_yolo_parts.py, qui réécrivait le dossier de labels
EN PLACE et supprimait les fichiers devenus vides — sans dry-run, sans
sauvegarde et avec des chemins codés en dur. Une exécution malheureuse
détruisait les annotations.

Ici: lecture seule sur la source, écriture dans un dossier distinct,
`--dry_run` par défaut désactivé mais disponible, et rapport détaillé.

    python scripts/remap_yolo_parts.py --src dataset/parts_yolo_raw/labels \
        --dst dataset/parts_yolo/labels --dry_run
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from claimsight.domain.taxonomy import EXCLUDED_PARTS_V1, PART_TO_ID


def load_source_mapping(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))["mapping"]
    return {int(k): v for k, v in raw.items()}


def remap_file(src: Path, source_map: dict[int, str]) -> tuple[list[str], Counter]:
    """Retourne (lignes remappées, compteur d'événements)."""
    stats: Counter = Counter()
    out: list[str] = []

    for lineno, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            old_id = int(parts[0])
        except (ValueError, IndexError):
            stats[f"malformed(l.{lineno})"] += 1
            continue

        name = source_map.get(old_id)
        if name is None:
            stats["unmapped_id"] += 1
            continue
        if name in EXCLUDED_PARTS_V1:
            stats[f"excluded:{name}"] += 1
            continue
        new_id = PART_TO_ID.get(name)
        if new_id is None:
            stats[f"unknown_part:{name}"] += 1
            continue

        parts[0] = str(new_id)
        out.append(" ".join(parts))
        stats["kept"] += 1

    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Dossier de labels source (lecture seule)")
    ap.add_argument("--dst", required=True, help="Dossier de labels de sortie")
    ap.add_argument("--mapping", default="configs/source_part_ids.json")
    ap.add_argument("--dry_run", action="store_true", help="N'écrit rien, affiche le rapport")
    ap.add_argument("--keep_empty", action="store_true",
                    help="Écrit aussi les fichiers sans aucune boîte retenue")
    args = ap.parse_args()

    src_dir, dst_dir = Path(args.src), Path(args.dst)
    if not src_dir.is_dir():
        raise SystemExit(f"Dossier source introuvable: {src_dir}")
    if src_dir.resolve() == dst_dir.resolve():
        raise SystemExit("--src et --dst doivent différer (pas de réécriture en place).")

    source_map = load_source_mapping(Path(args.mapping))
    if not args.dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    totals: Counter = Counter()
    written = dropped = 0

    for label_file in sorted(src_dir.glob("*.txt")):
        lines, stats = remap_file(label_file, source_map)
        totals.update(stats)
        if not lines and not args.keep_empty:
            dropped += 1
            continue
        if not args.dry_run:
            (dst_dir / label_file.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Fichiers écrits: {written} | "
          f"ignorés (aucune boîte V1): {dropped}")
    print("Détail des boîtes:")
    for key, count in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {key:32s} {count}")
    if args.dry_run:
        print("\nAucun fichier modifié. Relancez sans --dry_run pour écrire.")


if __name__ == "__main__":
    main()
