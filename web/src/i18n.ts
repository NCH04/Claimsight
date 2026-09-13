// Libellés français côté interface (l'API parle la taxonomie technique V1)

export const VIEW_LABELS: Record<string, string> = {
  front: "Avant",
  rear: "Arrière",
  left: "Gauche",
  right: "Droite",
  "front-left": "Avant gauche",
  "front-right": "Avant droite",
  "rear-left": "Arrière gauche",
  "rear-right": "Arrière droite",
  closeup: "Gros plan",
  out_of_scope: "Hors périmètre",
};

export const DAMAGE_LABELS: Record<string, string> = {
  none: "Aucun dommage",
  scratch: "Rayure",
  dent: "Bosse",
  crack: "Fissure",
  broken_glass: "Bris de glace",
  deformation_impact: "Déformation / impact",
  missing_part: "Pièce manquante",
  unknown: "Indéterminé",
};

export const SEVERITY_LABELS: Record<string, string> = {
  none: "Aucune",
  minor: "Légère",
  moderate: "Modérée",
  severe: "Sévère",
  unknown: "Indéterminée",
};

export const QUALITY_LABELS: Record<string, string> = {
  blurry: "Floue",
  dark: "Sombre",
  low_confidence: "Confiance faible",
};

export const viewLabel = (v: string) => VIEW_LABELS[v] ?? v;
export const damageLabel = (d: string) => DAMAGE_LABELS[d] ?? d;
export const severityLabel = (s: string) => SEVERITY_LABELS[s] ?? s;

export const pct = (x: number) => `${Math.round(x * 100)} %`;
