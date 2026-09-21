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
      <span>or browse your files — jpg, png, webp</span>
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
    state === "working" ? "analysing" : state === "done" ? "analysis complete" : state === "error" ? "failed" : "idle";
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
        <span>Overall confidence</span>
        <span className="value">{pct(value)}</span>
      </div>
      <div
        className="track"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 100)}
        aria-label="Overall confidence of the analysis"
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
        No part localised. Add the part detector (models/parts.pt) to get this level of
        detail.
      </p>
    );
  }
  return (
    <table className="parts">
      <thead>
        <tr>
          <th scope="col">Part</th>
          <th scope="col">Damage</th>
          <th scope="col">Severity</th>
          <th scope="col">Confidence</th>
          <th scope="col">Faces</th>
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
    return <p className="empty-note">All four faces of the vehicle are covered.</p>;
  }
  return (
    <div className="missing-list">
      {missing.map((cat) => (
        <div key={cat} className="missing-item">
          <span>
            <strong>{viewLabel(cat)}</strong> face missing — add a photo.
          </span>
          <button type="button" onClick={() => onComplete(cat)}>
            Complete the claim
          </button>
        </div>
      ))}
    </div>
  );
}
