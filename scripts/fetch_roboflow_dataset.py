#!/usr/bin/env python3
"""Télécharge un dataset Roboflow Universe — après avoir vérifié sa licence.

L'ordre compte. Roboflow Universe héberge aussi bien du CC BY 4.0 que du
non-commercial et du «tous droits réservés», et la licence ne se lit pas sur la
page de téléchargement. On interroge donc les métadonnées d'abord, et on
refuse de télécharger ce qu'on ne pourra pas redistribuer.

La clé d'API se lit **uniquement** dans l'environnement: une clé passée en
argument finit dans l'historique du shell, et une clé écrite dans un fichier
finit dans un commit.

    export ROBOFLOW_API_KEY=...
    python scripts/fetch_roboflow_dataset.py \
        --workspace car-damaged-detection-e66m0 \
        --project car-damaged-severity-detection \
        --version 29 --dest ../datasets/car-damage-severity

    # Lire la licence sans rien télécharger:
    python scripts/fetch_roboflow_dataset.py --workspace ... --project ... --check_only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.roboflow.com"

#: Licences dont les termes permettent de publier le code ET les poids dérivés,
#: attribution faite. Tout le reste exige une décision humaine.
ALLOWED = {"CC BY 4.0", "CC BY 3.0", "Public Domain", "MIT", "Apache 2.0"}


def api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit(
            "ROBOFLOW_API_KEY absent de l'environnement.\n"
            "  export ROBOFLOW_API_KEY=...   (jamais en argument, jamais dans un fichier)"
        )
    return key


def get_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        hint = " — clé invalide ou projet privé ?" if exc.code in (401, 403) else ""
        sys.exit(f"HTTP {exc.code} sur {url.split('?')[0]}{hint}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--format", default="yolov11", help="Format d'export (défaut: yolov11)")
    ap.add_argument("--dest", default=None, help="Dossier de destination")
    ap.add_argument("--check_only", action="store_true",
                    help="Afficher la licence et s'arrêter")
    ap.add_argument("--accept_any_licence", action="store_true",
                    help="Passer outre le garde-fou — à n'utiliser qu'en connaissance de cause")
    args = ap.parse_args()

    key = api_key()
    meta = get_json(f"{API}/{args.workspace}/{args.project}?api_key={key}")
    project = meta.get("project", meta)
    licence = project.get("license", "inconnue")

    print(f"Projet   : {project.get('name', args.project)}")
    print(f"Images   : {project.get('images', '?')}")
    print(f"Licence  : {licence}")

    if licence not in ALLOWED:
        msg = (f"\nLicence «{licence}» hors de la liste autorisée {sorted(ALLOWED)}.\n"
               "Les poids entraînés dessus en héritent: ne les publiez pas sans vérifier.")
        if not args.accept_any_licence:
            sys.exit(msg + "\nRelancez avec --accept_any_licence si c'est un choix délibéré.")
        print(msg)

    if args.check_only:
        return
    if not args.dest:
        sys.exit("--dest est requis pour télécharger (ou utilisez --check_only).")

    version = args.version or project.get("versions")
    if not isinstance(version, int):
        sys.exit("--version est requis: l'API ne donne pas de version par défaut fiable.")

    export = get_json(
        f"{API}/{args.workspace}/{args.project}/{version}/{args.format}?api_key={key}"
    )
    link = export.get("export", {}).get("link")
    if not link:
        sys.exit(f"Pas de lien d'export pour la version {version} au format {args.format}.")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "dataset.zip"
    print(f"\nTéléchargement de la v{version} ({args.format}) -> {dest}")
    with urllib.request.urlopen(link, timeout=600) as r, archive.open("wb") as fh:
        shutil.copyfileobj(r, fh)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    archive.unlink()

    splits = [d.name for d in sorted(dest.iterdir()) if (d / "images").is_dir()]
    print(f"Extrait  : {', '.join(splits) or 'aucun split standard détecté'}")
    print(f"\nÀ reproduire partout où le modèle est distribué:\n"
          f"  > {project.get('name', args.project)}, Roboflow Universe, licensed under {licence}.")


if __name__ == "__main__":
    main()
