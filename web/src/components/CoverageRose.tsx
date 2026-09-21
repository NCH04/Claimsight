import { useEffect, useState } from "react";
import { viewLabel } from "../i18n";

/* View rosette: silhouette du véhicule vue de dessus, entourée des 8 secteurs
   de prise de vue. Secteur plein = face couverte (couleur = pire gravité),
   secteur pointillé = face manquante (cliquable). */

export interface RoseData {
  countByView: Record<string, number>;
  severityByView: Record<string, string>;
  missing: string[];
}

interface Props {
  data: RoseData | null;
  onMissingClick?: (category: string) => void;
}

const SECTORS: { view: string; angle: number; short: string }[] = [
  { view: "front", angle: -90, short: "F" },
  { view: "front-right", angle: -45, short: "FR" },
  { view: "right", angle: 0, short: "R" },
  { view: "rear-right", angle: 45, short: "RR" },
  { view: "rear", angle: 90, short: "B" },
  { view: "rear-left", angle: 135, short: "RL" },
  { view: "left", angle: 180, short: "L" },
  { view: "front-left", angle: -135, short: "FL" },
];

const CARDINALS = new Set(["front", "rear", "left", "right"]);

const SEV_COLORS: Record<string, string> = {
  minor: "var(--sev-minor)",
  moderate: "var(--sev-moderate)",
  severe: "var(--sev-severe)",
};

const CX = 170;
const CY = 170;
const R1 = 74;
const R2 = 136;
const HALF = 18.5; // demi-ouverture du secteur (45° - écart)

function polar(r: number, deg: number): [number, number] {
  const a = (deg * Math.PI) / 180;
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
}

function wedgePath(angle: number): string {
  const a1 = angle - HALF;
  const a2 = angle + HALF;
  const [x1, y1] = polar(R2, a1);
  const [x2, y2] = polar(R2, a2);
  const [x3, y3] = polar(R1, a2);
  const [x4, y4] = polar(R1, a1);
  return `M ${x1} ${y1} A ${R2} ${R2} 0 0 1 ${x2} ${y2} L ${x3} ${y3} A ${R1} ${R1} 0 0 0 ${x4} ${y4} Z`;
}

export default function CoverageRose({ data, onMissingClick }: Props) {
  // Révélation différée: les secteurs partent neutres puis se remplissent
  // en cascade (transition CSS + délais), sauf reduced-motion (géré en CSS).
  // setTimeout plutôt que requestAnimationFrame: rAF ne se déclenche pas
  // dans un onglet en arrière-plan, ce qui figerait la rose en état neutre.
  const [revealed, setRevealed] = useState(false);
  useEffect(() => {
    if (!data) {
      setRevealed(false);
      return;
    }
    const timer = window.setTimeout(() => setRevealed(true), 30);
    return () => window.clearTimeout(timer);
  }, [data]);

  const covered = data && revealed ? data.countByView : {};
  const missing = new Set(data && revealed ? data.missing : []);

  const coveredCount = Object.values(covered).filter((n) => n > 0).length;
  const ariaLabel = data
    ? `Photo coverage: ${coveredCount} face(s) covered` +
      (data.missing.length ? `, missings: ${data.missing.map(viewLabel).join(", ")}` : "")
    : "No photos analysed yet";

  return (
    <svg viewBox="0 0 340 340" width="300" height="300" role="img" aria-label={ariaLabel}>
      {SECTORS.map((s, i) => {
        const count = covered[s.view] ?? 0;
        const isCovered = count > 0;
        const isMissing = !isCovered && CARDINALS.has(s.view) && missing.has(s.view);
        const sevColor = SEV_COLORS[data?.severityByView[s.view] ?? ""];

        const cls = isCovered ? "rose-sector covered" : isMissing ? "rose-sector missing" : "rose-sector";
        const style: React.CSSProperties = { transitionDelay: `${i * 60}ms` };
        if (isCovered) {
          style.fill = sevColor ?? "var(--bleu-voile)";
          style.fillOpacity = sevColor ? 0.3 : 1;
          if (sevColor) style.stroke = sevColor;
        }

        const [lx, ly] = polar(R2 + 18, s.angle);
        const [mx, my] = polar((R1 + R2) / 2, s.angle);
        const labelCls = isCovered ? "rose-label covered" : isMissing ? "rose-label missing" : "rose-label";

        const interactive = isMissing && onMissingClick;
        return (
          <g key={s.view}>
            <path
              d={wedgePath(s.angle)}
              className={cls}
              style={style}
              role={interactive ? "button" : undefined}
              tabIndex={interactive ? 0 : undefined}
              aria-label={
                interactive
                  ? `${viewLabel(s.view)} face missing — add a photo`
                  : undefined
              }
              onClick={interactive ? () => onMissingClick(s.view) : undefined}
              onKeyDown={
                interactive
                  ? (e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onMissingClick(s.view);
                      }
                    }
                  : undefined
              }
            >
              <title>
                {viewLabel(s.view)}
                {isCovered ? ` — ${count} photo${count > 1 ? "s" : ""}` : isMissing ? " — missing" : ""}
              </title>
            </path>
            <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle" className={labelCls}>
              {s.short}
            </text>
            {isCovered && (
              <text x={mx} y={my} textAnchor="middle" dominantBaseline="middle" className="rose-count">
                {count}
              </text>
            )}
          </g>
        );
      })}

      {/* Silhouette du véhicule (vue de dessus, avant vers le haut) */}
      <g>
        <path
          className="rose-car-body"
          d="M 170 116
             C 184 116 193 122 194 136
             L 196 204
             C 196 217 185 224 170 224
             C 155 224 144 217 144 204
             L 146 136
             C 147 122 156 116 170 116 Z"
        />
        <path className="rose-car-line" d="M 150 153 Q 170 144 190 153" />
        <path className="rose-car-line" d="M 151 196 Q 170 204 189 196" />
        <path className="rose-car-line" d="M 146 150 l -7 -5" />
        <path className="rose-car-line" d="M 194 150 l 7 -5" />
      </g>
    </svg>
  );
}
