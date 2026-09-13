"""Stockage des dossiers de sinistre et suivi des analyses.

L'analyse d'un dossier dure trop longtemps pour tenir dans une requête HTTP
synchrone sans risquer un timeout navigateur. Le contrat est donc asynchrone:
le dépôt répond 202 avec un identifiant, et le client interroge l'état jusqu'à
`done`.

Le suivi vit en mémoire, le résultat sur disque: un redémarrage perd les
analyses en cours mais jamais celles qui ont abouti.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)

THUMB_SIZE = 480

#: États d'un dossier, alignés sur le type `ClaimState` du front.
QUEUED, PROCESSING, DONE, ERROR = "queued", "processing", "done", "error"


@dataclass
class Claim:
    claim_id: str
    status: str = QUEUED
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    n_images: int = 0


class ClaimStore:
    """Dossiers sur disque + état des analyses en mémoire."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._claims: dict[str, Claim] = {}
        self._lock = threading.Lock()

    # -- chemins -------------------------------------------------------------

    def directory(self, claim_id: str) -> Path:
        """Dossier d'un sinistre.

        `claim_id` vient d'un uuid4 hexadécimal, mais on revalide: un
        identifiant venu de l'URL ne doit jamais pouvoir remonter l'arborescence.
        """
        if not claim_id.isalnum() or len(claim_id) > 64:
            raise ValueError(f"Identifiant de dossier invalide: {claim_id!r}")
        return self.root / claim_id

    def images_dir(self, claim_id: str) -> Path:
        return self.directory(claim_id) / "images"

    def thumbs_dir(self, claim_id: str) -> Path:
        return self.directory(claim_id) / "thumbs"

    def result_path(self, claim_id: str) -> Path:
        return self.directory(claim_id) / "result.json"

    # -- cycle de vie --------------------------------------------------------

    def create(self) -> Claim:
        claim = Claim(claim_id=uuid.uuid4().hex)
        self.images_dir(claim.claim_id).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._claims[claim.claim_id] = claim
        return claim

    def get(self, claim_id: str) -> Claim | None:
        with self._lock:
            claim = self._claims.get(claim_id)
        if claim is not None:
            return claim
        # Un résultat sur disque survit au redémarrage du service.
        if self.result_path(claim_id).exists():
            return Claim(claim_id=claim_id, status=DONE)
        return None

    def list(self) -> list[Claim]:
        with self._lock:
            known = dict(self._claims)
        for path in self.root.iterdir():
            if path.is_dir() and path.name not in known and (path / "result.json").exists():
                known[path.name] = Claim(claim_id=path.name, status=DONE)
        return sorted(known.values(), key=lambda c: c.created_at, reverse=True)

    def set_status(self, claim_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            claim = self._claims.setdefault(claim_id, Claim(claim_id=claim_id))
            claim.status = status
            claim.error = error

    def save_result(self, claim_id: str, result: dict) -> None:
        self.result_path(claim_id).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_result(self, claim_id: str) -> dict | None:
        path = self.result_path(claim_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, claim_id: str) -> None:
        shutil.rmtree(self.directory(claim_id), ignore_errors=True)
        with self._lock:
            self._claims.pop(claim_id, None)

    # -- images --------------------------------------------------------------

    def image_path(self, claim_id: str, filename: str, thumb: bool = False) -> Path | None:
        """Chemin d'une image du dossier, ou None si elle n'existe pas.

        Le nom est réduit à son basename: un `../` dans l'URL ne peut pas
        atteindre un fichier hors du dossier.
        """
        safe = Path(filename).name
        if not safe:
            return None
        base = self.thumbs_dir(claim_id) if thumb else self.images_dir(claim_id)
        candidate = base / safe
        if thumb and not candidate.exists():
            candidate = self.images_dir(claim_id) / safe   # repli sur l'original
        return candidate if candidate.exists() else None

    def make_thumbnail(self, claim_id: str, filename: str) -> None:
        """Miniature pour la grille du front. Jamais bloquante."""
        source = self.images_dir(claim_id) / Path(filename).name
        try:
            with Image.open(source) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                dest = self.thumbs_dir(claim_id) / Path(filename).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                img.save(dest, "JPEG", quality=80, optimize=True)
        except Exception as exc:
            LOGGER.debug("Miniature impossible pour %s: %s", filename, exc)
