"""API HTTP de ClaimSight.

    claimsight-serve                # ou: uvicorn claimsight.api.server:app --reload

Deux partis pris.

**Les modèles sont chargés une fois** au démarrage, via le `lifespan`, jamais
par requête: construire trois ResNet et un YOLO coûte plusieurs centaines de
millisecondes. `ModelBundle` a été conçu pour ça.

**Le dépôt est asynchrone.** Analyser un dossier dépasse ce qu'une requête
HTTP synchrone peut tenir sans risquer un timeout navigateur: `POST /api/claims`
répond 202 avec un identifiant, et le client interroge l'état jusqu'à `done`.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..pipeline.config import INFERENCE_BATCH_SIZE
from ..pipeline.io_utils import IMAGE_EXTS
from ..pipeline.models import ModelBundle
from ..pipeline.run_pipeline import PIPELINE_VERSION, run_pipeline
from .schemas import ClaimCreated, ClaimStatus, ClaimSummary, Health, ModelStatus
from .store import DONE, ERROR, PROCESSING, ClaimStore

LOGGER = logging.getLogger("claimsight.api")

#: Garde-fous d'upload: un dossier de sinistre reste petit, et accepter
#: l'illimité offrirait un déni de service trivial.
MAX_FILES = int(os.getenv("CLAIMSIGHT_MAX_FILES", "30"))
MAX_FILE_BYTES = int(os.getenv("CLAIMSIGHT_MAX_FILE_MB", "25")) * 1024 * 1024

STATE: dict = {"bundle": None, "error": None, "store": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["store"] = ClaimStore(os.getenv("CLAIMSIGHT_DATA", "outputs/claims"))
    try:
        STATE["bundle"] = ModelBundle.load(
            coverage_checkpoint=os.getenv("CLAIMSIGHT_COVERAGE", "models/coverage.pt"),
            view_checkpoint=os.getenv("CLAIMSIGHT_VIEW", "models/view.pt"),
            damage_checkpoint=os.getenv("CLAIMSIGHT_DAMAGE") or None,
            severity_checkpoint=os.getenv("CLAIMSIGHT_SEVERITY") or None,
            parts_checkpoint=os.getenv("CLAIMSIGHT_PARTS", "models/parts.pt"),
        )
        LOGGER.info("Modèles chargés.")
    except Exception as exc:  # le service démarre quand même; /api/health l'expose
        STATE["error"] = str(exc)
        LOGGER.error("Chargement des modèles impossible: %s", exc)
    yield
    STATE["bundle"] = None


app = FastAPI(
    title="ClaimSight",
    version=PIPELINE_VERSION,
    summary="Analyse automatisée de photos de sinistres automobiles.",
    lifespan=lifespan,
)

# Le front tourne sur une autre origine en développement (Vite: 5173).
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CLAIMSIGHT_CORS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _store() -> ClaimStore:
    store = STATE["store"]
    if store is None:
        raise HTTPException(503, "Service non initialisé.")
    return store


def _bundle():
    bundle = STATE["bundle"]
    if bundle is None:
        raise HTTPException(503, f"Modèles indisponibles: {STATE['error']}")
    return bundle


# --------------------------------------------------------------------- santé


@app.get("/api/health", response_model=Health, tags=["service"])
def health() -> Health:
    """État des modèles. `ready` dès qu'une orientation est disponible."""
    bundle = STATE["bundle"]
    if bundle is None:
        return Health(status="unavailable", pipeline_version=PIPELINE_VERSION,
                      models={"_": ModelStatus(loaded=False, detail=STATE["error"])})
    models = {
        "coverage": ModelStatus(loaded=bundle.coverage.available),
        "view": ModelStatus(loaded=bundle.view.available),
        "damage": ModelStatus(loaded=bundle.damage.available,
                              detail=None if bundle.damage.available else "non entraîné"),
        "severity": ModelStatus(loaded=bundle.severity.available,
                                detail=None if bundle.severity.available else "non entraîné"),
        "parts": ModelStatus(loaded=bundle.parts.available),
    }
    ready = bundle.coverage.available or bundle.view.available
    return Health(status="ready" if ready else "degraded",
                  pipeline_version=PIPELINE_VERSION, models=models)


# -------------------------------------------------------------------- dépôt


def _stage_uploads(files: list[UploadFile], directory: Path) -> list[str]:
    """Écrit les fichiers reçus, en refusant ce qui n'est pas exploitable."""
    allowed = {e.lower() for e in IMAGE_EXTS}
    names: list[str] = []
    for upload in files:
        name = Path(upload.filename or "").name       # neutralise tout ../
        if not name or Path(name).suffix.lower() not in allowed:
            raise HTTPException(
                415, f"{name or '(sans nom)'}: extension non supportée. Attendu: {sorted(allowed)}"
            )
        size = 0
        with (directory / name).open("wb") as fh:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise HTTPException(413, f"{name}: dépasse {MAX_FILE_BYTES // 1048576} Mo")
                fh.write(chunk)
        names.append(name)
    return names


def _analyse(claim_id: str) -> None:
    """Analyse en tâche de fond. Ne lève jamais: l'échec devient l'état du dossier."""
    store, bundle = STATE["store"], STATE["bundle"]
    store.set_status(claim_id, PROCESSING)
    try:
        result = run_pipeline(
            images_dir=str(store.images_dir(claim_id)), out_json=None,
            bundle=bundle, batch_size=INFERENCE_BATCH_SIZE,
        )
        result["vehicle_id"] = claim_id
        store.save_result(claim_id, result)
        store.set_status(claim_id, ERROR if result["status"] == "error" else DONE,
                         result["summary"] if result["status"] == "error" else None)
    except Exception as exc:  # pragma: no cover - filet de sécurité
        LOGGER.exception("Analyse du dossier %s en échec", claim_id)
        store.set_status(claim_id, ERROR, str(exc))


@app.post("/api/claims", response_model=ClaimCreated, status_code=202, tags=["claims"])
def create_claim(
    files: Annotated[list[UploadFile], File(description="Photos du sinistre")],
    background: BackgroundTasks,
) -> ClaimCreated:
    """Dépose un dossier. Répond 202; interrogez `GET /api/claims/{id}` ensuite."""
    _bundle()
    store = _store()
    if not files:
        raise HTTPException(400, "Aucun fichier reçu.")
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"{len(files)} fichiers reçus, maximum {MAX_FILES}.")

    claim = store.create()
    try:
        names = _stage_uploads(files, store.images_dir(claim.claim_id))
    except HTTPException:
        store.delete(claim.claim_id)
        raise

    claim.n_images = len(names)
    for name in names:
        store.make_thumbnail(claim.claim_id, name)
    background.add_task(_analyse, claim.claim_id)
    return ClaimCreated(claim_id=claim.claim_id, status=claim.status, n_images=claim.n_images)


@app.get("/api/claims", response_model=list[ClaimSummary], tags=["claims"])
def list_claims() -> list[ClaimSummary]:
    return [ClaimSummary(**vars(c)) for c in _store().list()]


@app.get("/api/claims/{claim_id}", response_model=ClaimStatus, tags=["claims"])
def get_claim(claim_id: str) -> ClaimStatus:
    """État d'un dossier, avec le constat dès que l'analyse est terminée."""
    store = _store()
    try:
        claim = store.get(claim_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if claim is None:
        raise HTTPException(404, f"Dossier introuvable: {claim_id}")
    return ClaimStatus(
        claim_id=claim.claim_id, status=claim.status, error=claim.error,
        created_at=claim.created_at, n_images=claim.n_images,
        result=store.load_result(claim_id) if claim.status == DONE else None,
    )


@app.get("/api/claims/{claim_id}/images/{filename}", tags=["claims"])
def get_claim_image(claim_id: str, filename: str, thumb: bool = False) -> FileResponse:
    """Sert une photo du dossier. `?thumb=1` renvoie la miniature."""
    store = _store()
    try:
        path = store.image_path(claim_id, filename, thumb=thumb)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if path is None:
        raise HTTPException(404, f"Image introuvable: {filename}")
    return FileResponse(path)


def serve() -> None:
    """Point d'entrée `claimsight-serve`."""
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Sert l'API ClaimSight.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    uvicorn.run("claimsight.api.server:app", host=args.host, port=args.port, reload=args.reload)


# --------------------------------------------------------------------- front
# Monté EN DERNIER: les routes /api/* sont déclarées avant, donc elles priment.
# En développement, Vite sert le front sur :5173 et proxifie /api vers ici.
_WEB_DIST = Path(os.getenv("CLAIMSIGHT_WEB", "web/dist"))
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
    LOGGER.info("Front servi depuis %s", _WEB_DIST)
else:
    LOGGER.info("Pas de bundle front dans %s — lancez `npm run build` dans web/.", _WEB_DIST)
