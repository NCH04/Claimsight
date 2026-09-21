"""Tests de l'API — sans modèles entraînés, donc exécutables en CI."""

import pytest

fastapi = pytest.importorskip("fastapi", reason="extra [api] non installé")
from fastapi.testclient import TestClient  # noqa: E402

from claimsight.api import server as api_module  # noqa: E402
from claimsight.api.schemas import ClaimReport, DamagedPart  # noqa: E402
from claimsight.api.store import ClaimStore  # noqa: E402


class _FakeClassifier:
    def __init__(self, available=True):
        self.available = available


class _FakeBundle:
    """Bundle minimal: /health et les garde-fous d'upload n'ont pas besoin de poids."""

    def __init__(self):
        self.coverage = _FakeClassifier(True)
        self.view = _FakeClassifier(False)
        self.damage = _FakeClassifier(False)
        self.severity = _FakeClassifier(False)
        self.parts = _FakeClassifier(True)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setitem(api_module.STATE, "bundle", _FakeBundle())
    monkeypatch.setitem(api_module.STATE, "error", None)
    monkeypatch.setitem(api_module.STATE, "store", ClaimStore(tmp_path))
    return TestClient(api_module.app)


# ------------------------------------------------------------------ /health
def test_health_reports_each_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ready"
    assert set(body["models"]) == {"coverage", "damage", "severity", "parts"}
    assert body["models"]["damage"]["detail"] == "not trained"


def test_health_omits_the_retired_view_model(client):
    """`view` a été remplacé par `coverage`: l'absence n'est pas une panne.

    Le lister comme `loaded: false` faisait passer une décision produit pour
    une dégradation — et le front affichait un avertissement permanent.
    """
    assert "view" not in client.get("/api/health").json()["models"]


def test_health_is_unavailable_without_models(monkeypatch):
    monkeypatch.setitem(api_module.STATE, "bundle", None)
    monkeypatch.setitem(api_module.STATE, "error", "checkpoint absent")
    body = TestClient(api_module.app).get("/api/health").json()
    assert body["status"] == "unavailable"


# ------------------------------------------------------- garde-fous d'upload
def test_non_image_is_refused(client):
    r = client.post("/api/claims", files=[("files", ("notes.txt", b"x", "text/plain"))])
    assert r.status_code == 415


def test_too_many_files_is_refused(client):
    files = [("files", (f"p{i}.jpg", b"x", "image/jpeg"))
             for i in range(api_module.MAX_FILES + 1)]
    assert client.post("/api/claims", files=files).status_code == 413


def test_oversized_file_is_refused(client, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_FILE_BYTES", 10)
    r = client.post("/api/claims", files=[("files", ("big.jpg", b"x" * 64, "image/jpeg"))])
    assert r.status_code == 413


def test_upload_filename_cannot_escape_the_claim_directory(client):
    """Un `../` dans le nom de fichier ne doit jamais écrire hors du dossier."""
    r = client.post("/api/claims",
                    files=[("files", ("../../etc/passwd.jpg", b"\xff\xd8\xff", "image/jpeg"))])
    assert r.status_code == 202
    store = api_module.STATE["store"]
    written = list(store.images_dir(r.json()["claim_id"]).iterdir())
    assert [p.name for p in written] == ["passwd.jpg"]


def test_claim_lifecycle_returns_an_id_then_a_status(client):
    r = client.post("/api/claims", files=[("files", ("a.jpg", b"\xff\xd8\xff", "image/jpeg"))])
    assert r.status_code == 202
    body = r.json()
    assert body["n_images"] == 1 and body["status"] == "queued"
    status = client.get(f"/api/claims/{body['claim_id']}").json()
    assert status["claim_id"] == body["claim_id"]


def test_unknown_claim_is_404(client):
    assert client.get("/api/claims/" + "0" * 32).status_code == 404


def test_claim_id_with_traversal_is_rejected(client):
    assert client.get("/api/claims/..%2F..%2Fetc").status_code in (400, 404)


def test_image_of_unknown_claim_is_404(client):
    assert client.get(f"/api/claims/{'0' * 32}/images/a.jpg").status_code == 404


def test_service_unavailable_without_bundle(monkeypatch):
    monkeypatch.setitem(api_module.STATE, "bundle", None)
    monkeypatch.setitem(api_module.STATE, "error", "boom")
    r = TestClient(api_module.app).post(
        "/api/claims", files=[("files", ("a.jpg", b"x", "image/jpeg"))])
    assert r.status_code == 503


# ------------------------------------------------------------------ contrat
def test_openapi_exposes_the_report_schema(client):
    schema = client.get("/openapi.json").json()
    assert "ClaimReport" in schema["components"]["schemas"]
    assert "/api/claims" in schema["paths"]


def test_confidence_is_constrained_to_unit_interval():
    with pytest.raises(ValueError):
        DamagedPart(name="hood", damage="dent", severity="minor",
                    confidence=1.5, views_evidence=[], detected_on=[])


def test_report_rejects_a_missing_required_field():
    with pytest.raises(ValueError):
        ClaimReport(pipeline_version="v1", status="ok", summary="x",
                    confidence=0.5, processing_time_ms=1)  # image_level_damage manquant
