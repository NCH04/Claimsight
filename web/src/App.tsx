import { useEffect, useMemo, useRef, useState } from "react";
import { createClaim, imageUrl, pollClaim } from "./api";
import { ConfidenceMeter, Dropzone, MissingList, PartsTable, StatusPill } from "./components/bits";
import RoseDesVues, { type RoseData } from "./components/RoseDesVues";
import { damageLabel, pct, QUALITY_LABELS, severityLabel, viewLabel } from "./i18n";
import type { ClaimResult, ClaimStatus, InputImage } from "./types";

type Phase = "idle" | "ready" | "uploading" | "processing" | "done" | "failed";

const SEV_ORDER: Record<string, number> = { minor: 1, moderate: 2, severe: 3 };

function buildRoseData(result: ClaimResult): RoseData {
  const countByView: Record<string, number> = {};
  const severityByView: Record<string, string> = {};
  for (const img of result.input_images) {
    if (img.deduplicated) continue;
    // Le modèle de couverture est multi-label: une photo en diagonale
    // documente deux faces. On retombe sur la vue quand il est absent.
    const faces =
      img.covered_faces?.length > 0
        ? img.covered_faces
        : [img.detected_view].filter(
            (v) => v && v !== "closeup" && v !== "out_of_scope" && v !== "unknown",
          );
    const damage = img.damage_prediction.label;
    const sev = img.severity_prediction.label;
    for (const view of faces) {
      countByView[view] = (countByView[view] ?? 0) + 1;
      if (damage !== "none" && damage !== "unknown" && SEV_ORDER[sev]) {
        if ((SEV_ORDER[severityByView[view]] ?? 0) < SEV_ORDER[sev]) {
          severityByView[view] = sev;
        }
      }
    }
  }
  return { countByView, severityByView, missing: result.missing_photos };
}

function verdictText(result: ClaimResult): string {
  const { damage, severity } = result.image_level_damage;
  if (damage === "none") return "Aucun dommage détecté";
  if (damage === "unknown") return "Damage to be confirmed";
  if (severity === "unknown" || severity === "none") return damageLabel(damage);
  return `${damageLabel(damage)} ${severityLabel(severity).toLowerCase()}`;
}

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [phase, setPhase] = useState<Phase>("idle");
  const [claim, setClaim] = useState<ClaimStatus | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [healthWarning, setHealthWarning] = useState<string | null>(null);
  const [dropPulse, setDropPulse] = useState(false);
  const [hint, setHint] = useState<string | null>(null);

  const dropRef = useRef<HTMLDivElement>(null);
  const cancelPoll = useRef<(() => void) | null>(null);
  const previews = useRef(new Map<string, string>());

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => {
        if (!h.models_loaded) {
          setHealthWarning(
            "Modèle de vue non chargé côté serveur. Placez le checkpoint dans models/view.pt puis redémarrez l'API.",
          );
        }
      })
      .catch(() =>
        setHealthWarning("Le serveur ne répond pas. Lancez l'API : uvicorn src.api.main:app --port 8000"),
      );
    const map = previews.current;
    return () => {
      cancelPoll.current?.();
      map.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  const previewUrl = (file: File): string => {
    let url = previews.current.get(file.name);
    if (!url) {
      url = URL.createObjectURL(file);
      previews.current.set(file.name, url);
    }
    return url;
  };

  const addFiles = (incoming: File[]) => {
    const images = incoming.filter((f) => f.type.startsWith("image/"));
    if (images.length === 0) return;
    setFiles((prev) => {
      const known = new Set(prev.map((f) => f.name));
      return [...prev, ...images.filter((f) => !known.has(f.name))];
    });
    setHint(null);
    setPhase((p) => (p === "idle" || p === "failed" ? "ready" : p));
  };

  const removeFile = (name: string) => {
    setFiles((prev) => prev.filter((f) => f.name !== name));
    const url = previews.current.get(name);
    if (url) {
      URL.revokeObjectURL(url);
      previews.current.delete(name);
    }
  };

  const analyze = async () => {
    if (files.length === 0) return;
    setErrorMsg(null);
    setPhase("uploading");
    try {
      const { claim_id } = await createClaim(files);
      setPhase("processing");
      cancelPoll.current?.();
      cancelPoll.current = pollClaim(
        claim_id,
        (status: ClaimStatus) => {
          setClaim(status);
          if (status.status === "done") setPhase("done");
          if (status.status === "error") {
            setPhase("failed");
            setErrorMsg(status.error ?? "Analysis failed on the server.");
          }
        },
        (message) => {
          setPhase("failed");
          setErrorMsg(message);
        },
      );
    } catch (e) {
      setPhase("failed");
      setErrorMsg(e instanceof Error ? e.message : String(e));
    }
  };

  const reset = () => {
    cancelPoll.current?.();
    previews.current.forEach((url) => URL.revokeObjectURL(url));
    previews.current.clear();
    setFiles([]);
    setClaim(null);
    setErrorMsg(null);
    setHint(null);
    setPhase("idle");
  };

  const askForView = (category: string) => {
    setHint(`Vue ${viewLabel(category).toLowerCase()} missing — ajoutez une photo ci-dessous.`);
    setDropPulse(true);
    // Différé: laisse passer le scroll natif de focus du secteur cliqué
    window.setTimeout(() => {
      dropRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 80);
    window.setTimeout(() => setDropPulse(false), 2600);
  };

  const result = claim?.status === "done" ? claim.result ?? null : null;
  const roseData = useMemo(() => (result ? buildRoseData(result) : null), [result]);

  const analyzedNames = useMemo(
    () => new Set(result?.input_images.map((i) => i.filename) ?? []),
    [result],
  );
  const pendingFiles = result ? files.filter((f) => !analyzedNames.has(f.name)) : files;

  const working = phase === "uploading" || phase === "processing";
  const pillState =
    phase === "done" ? "done" : phase === "failed" ? "error" : working ? "working" : "idle";

  const count = files.length > 0 ? ` (${files.length} photo${files.length > 1 ? "s" : ""})` : "";
  const analyzeLabel = working
    ? phase === "uploading"
      ? "Uploading photos…"
      : "Analysing…"
    : result
      ? `Relancer l'analyse${count}`
      : `Analyse le dossier${count}`;

  return (
    <>
      <header className="masthead">
        <h1>
          Fiche d'expertise automobile
          <small>Analyse photographique du sinistre — pipeline v1</small>
        </h1>
        <div className="ref">
          {claim ? (
            <>
              dossier <strong>{claim.claim_id}</strong>
              <br />
            </>
          ) : null}
          <StatusPill state={pillState} />
        </div>
      </header>

      {healthWarning && (
        <div className="api-error" role="alert">
          {healthWarning.includes("uvicorn") ? (
            <>
              Le serveur ne répond pas. Lancez l'API : <code>uvicorn src.api.main:app --port 8000</code>
            </>
          ) : (
            healthWarning
          )}
        </div>
      )}

      <main className="workbench">
        {/* ----- Volet dossier ----- */}
        <section className="panel" aria-label="Claim photos">
          <div className="panel-head">
            <span className="eyebrow">Claim photos</span>
            {result && (
              <button type="button" className="btn ghost" onClick={reset}>
                Nouveau dossier
              </button>
            )}
          </div>
          <div className="panel-body">
            {hint && <p className="empty-note" style={{ marginBottom: 10 }}>{hint}</p>}
            <Dropzone onFiles={addFiles} pulse={dropPulse} innerRef={dropRef} />

            <div className="photo-list">
              {pendingFiles.map((f) => (
                <div key={f.name} className="photo-card">
                  <img src={previewUrl(f)} alt="" />
                  <div className="meta">
                    <div className="name">{f.name}</div>
                    {result && (
                      <div className="chips">
                        <span className="chip mono">à analyser</span>
                      </div>
                    )}
                  </div>
                  <button
                    type="button"
                    className="remove"
                    aria-label={`Retirer ${f.name}`}
                    onClick={() => removeFile(f.name)}
                    disabled={working}
                  >
                    ×
                  </button>
                </div>
              ))}

              {result &&
                claim &&
                result.input_images.map((img: InputImage) => (
                  <div key={img.filename} className={`photo-card${img.deduplicated ? " is-dup" : ""}`}>
                    <img src={imageUrl(claim.claim_id, img.filename, true)} alt="" />
                    <div className="meta">
                      <div className="name">{img.filename}</div>
                      <div className="chips">
                        <span className="chip view">
                          {img.covered_faces?.length
                            ? img.covered_faces.map(viewLabel).join(" + ")
                            : viewLabel(img.detected_view)}
                        </span>
                        {img.quality_flag !== "ok" && (
                          <span className="chip quality">
                            {QUALITY_LABELS[img.quality_flag] ?? img.quality_flag}
                          </span>
                        )}
                        {img.deduplicated && <span className="chip dup">Duplicate</span>}
                        {img.damage_prediction.label !== "unknown" && (
                          <span className="chip mono">
                            {damageLabel(img.damage_prediction.label)} · {pct(img.damage_prediction.confidence)}
                          </span>
                        )}
                      </div>
                    </div>
                    <span />
                  </div>
                ))}
            </div>

            {result && result.errors.length > 0 && (
              <p className="empty-note" style={{ marginTop: 10 }}>
                {result.errors.length} photo(s) illisible(s) — remplacez-les :{" "}
                {result.errors.map((e) => e.split(":")[0]).join(", ")}
              </p>
            )}

            <button
              type="button"
              className="btn primary"
              onClick={analyze}
              disabled={files.length === 0 || working}
            >
              {analyzeLabel}
            </button>
          </div>
        </section>

        {/* ----- Volet fiche ----- */}
        <section className="panel" aria-label="Fiche d'expertise">
          {phase === "failed" && errorMsg && (
            <div className="api-error" role="alert">
              L'analyse a échoué : {errorMsg}
            </div>
          )}

          {!result ? (
            <div className="report-empty">
              <RoseDesVues data={null} />
              <p>
                {working
                  ? "Analysing — la fiche se remplira automatiquement."
                  : "Aucun dossier analysé. Drop les photos du sinistre pour ouvrir la fiche : vues couvertes, dommages, pièces touchées et photos à compléter."}
              </p>
            </div>
          ) : (
            <>
              <div className="verdict">
                <div>
                  <h2>{verdictText(result)}</h2>
                  <div className="sub">
                    {result.input_images.filter((i) => !i.deduplicated).length} photo(s) analysée(s) en{" "}
                    {(result.processing_time_ms / 1000).toFixed(1)} s
                  </div>
                </div>
                <ConfidenceMeter value={result.confidence} />
              </div>

              {result.suspected_total_loss && (
                <div className="banner-total-loss" role="alert">
                  Total loss suspectée
                  <span>— plusieurs pièces sévèrement touchées. Expertise physique recommandée.</span>
                </div>
              )}

              <div className="fiche-section">
                <span className="eyebrow">Coverage des vues</span>
                <div className="rose-wrap">
                  <RoseDesVues data={roseData} onMissingClick={askForView} />
                  <div className="rose-legend">
                    <div className="item">
                      <span className="swatch" style={{ background: "var(--bleu-voile)", border: "1px solid var(--bleu-constat)" }} />
                      Vue couverte, sans dommage relevé
                    </div>
                    <div className="item">
                      <span className="swatch" style={{ background: "var(--sev-minor)" }} />
                      Dommage léger
                    </div>
                    <div className="item">
                      <span className="swatch" style={{ background: "var(--sev-moderate)" }} />
                      Dommage modéré
                    </div>
                    <div className="item">
                      <span className="swatch" style={{ background: "var(--sev-severe)" }} />
                      Dommage sévère
                    </div>
                    <div className="item">
                      <span className="swatch" style={{ border: "1.5px dashed var(--sev-severe)", background: "var(--fiche)" }} />
                      Vue missing — cliquez pour compléter
                    </div>
                  </div>
                </div>
              </div>

              <div className="fiche-section">
                <span className="eyebrow">Damaged parts</span>
                <PartsTable parts={result.damaged_parts} />
              </div>

              <div className="fiche-section">
                <span className="eyebrow">Photos à compléter</span>
                <MissingList missing={result.missing_photos} onComplete={askForView} />
              </div>

              <div className="fiche-section">
                <span className="meta-line">
                  {result.summary} — pipeline {result.pipeline_version} · statut {result.status}
                </span>
              </div>
            </>
          )}
        </section>
      </main>
    </>
  );
}
