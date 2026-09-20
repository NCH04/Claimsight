import { useRef, useState } from "react";
import { damageLabel, pct, severityLabel, viewLabel } from "../i18n";
import type { DamagedPart } from "../types";

/* Petits composants de la fiche: dépôt de fichiers, jauge, statut, tableaux. */

export function Dropzone({
  onFiles,
  pulse,
  innerRef,
}: {
  onFiles: (files: File[]) => void;
  pulse: boolean;
  innerRef?: React.Ref<HTMLDivElement>;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragover, setDragover] = useState(false);

  const openPicker = () => inputRef.current?.click();

  return (
    <div
      ref={innerRef}
      className={`dropzone${dragover ? " dragover" : ""}${pulse ? " pulse" : ""}`}
      role="button"
      tabIndex={0}
      aria-label="Drop the claim photos"
      onClick={openPicker}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openPicker();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setDragover(true);
      }}
      onDragLeave={() => setDragover(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragover(false);
        onFiles(Array.from(e.dataTransfer.files));
      }}
    >
      <strong>Drop the claim photos</strong>
      <span>ou parcourir vos fichiers — jpg, png, webp</span>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => {
          onFiles(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
    </div>
  );
}

export function StatusPill({ state }: { state: "idle" | "working" | "done" | "error" }) {
  const label =
    state === "working" ? "analyse en cours" : state === "done" ? "analysis complete" : state === "error" ? "échec" : "en attente";
  const cls = state === "working" ? "working" : state === "done" ? "done" : state === "error" ? "error" : "";
  return (
    <span className={`status-pill ${cls}`}>
      <span className="dot" aria-hidden="true" />
      {label}
    </span>
  );
}

export function ConfidenceMeter({ value }: { value: number }) {
  return (
    <div className="confidence">
      <div className="label">
        <span>Confidence globale</span>
        <span className="value">{pct(value)}</span>
      </div>
      <div
        className="track"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 100)}
        aria-label="Confidence globale de l'analyse"
      >
        <div className="fill" style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}

export function SeverityChip({ severity }: { severity: string }) {
  const cls = ["minor", "moderate", "severe", "none"].includes(severity) ? `sev-${severity}` : "sev-none";
  return <span className={`chip ${cls}`}>{severityLabel(severity)}</span>;
}

export function PartsTable({ parts }: { parts: DamagedPart[] }) {
  if (parts.length === 0) {
    return (
      <p className="empty-note">
        Aucune pièce localisée. Ajoutez le détecteur de pièces (models/parts.pt) pour obtenir ce
        niveau de détail.
      </p>
    );
  }
  return (
    <table className="parts">
      <thead>
        <tr>
          <th scope="col">Pièce</th>
          <th scope="col">Dommage</th>
          <th scope="col">Sévérité</th>
          <th scope="col">Confidence</th>
          <th scope="col">Vues</th>
        </tr>
      </thead>
      <tbody>
        {parts.map((p) => (
          <tr key={p.name}>
            <td>{p.name}</td>
            <td>{damageLabel(p.damage)}</td>
            <td>
              <SeverityChip severity={p.severity} />
            </td>
            <td className="conf">{pct(p.confidence)}</td>
            <td>{p.views_evidence.map(viewLabel).join(", ")}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function MissingList({
  missing,
  onComplete,
}: {
  missing: string[];
  onComplete: (category: string) => void;
}) {
  if (missing.length === 0) {
    return <p className="empty-note">Les quatre faces du véhicule sont couvertes.</p>;
  }
  return (
    <div className="missing-list">
      {missing.map((cat) => (
        <div key={cat} className="missing-item">
          <span>
            Vue <strong>{viewLabel(cat).toLowerCase()}</strong> missing — ajoutez une photo.
          </span>
          <button type="button" onClick={() => onComplete(cat)}>
            Compléter le dossier
          </button>
        </div>
      ))}
    </div>
  );
}
