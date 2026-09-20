// UI labels. The API speaks the V1 technical taxonomy; this maps it to prose.

export const VIEW_LABELS: Record<string, string> = {
  front: "Front",
  rear: "Rear",
  left: "Left",
  right: "Right",
  "front-left": "Front left",
  "front-right": "Front right",
  "rear-left": "Rear left",
  "rear-right": "Rear right",
  closeup: "Close-up",
  out_of_scope: "Out of scope",
  unknown: "Undetermined",
};

export const DAMAGE_LABELS: Record<string, string> = {
  none: "No damage",
  scratch: "Scratch",
  dent: "Dent",
  crack: "Crack",
  broken_glass: "Broken glass",
  deformation_impact: "Deformation / impact",
  missing_part: "Missing part",
  unknown: "Undetermined",
};

export const SEVERITY_LABELS: Record<string, string> = {
  none: "None",
  minor: "Minor",
  moderate: "Moderate",
  severe: "Severe",
  unknown: "Undetermined",
};

export const QUALITY_LABELS: Record<string, string> = {
  blurry: "Blurry",
  dark: "Under-exposed",
  low_confidence: "Low confidence",
};

export const viewLabel = (v: string) => VIEW_LABELS[v] ?? v;
export const damageLabel = (d: string) => DAMAGE_LABELS[d] ?? d;
export const severityLabel = (s: string) => SEVERITY_LABELS[s] ?? s;

export const pct = (x: number) => `${Math.round(x * 100)}%`;
