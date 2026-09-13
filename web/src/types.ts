// Types alignés sur le contrat Pydantic de l'API
// (claimsight/api/schemas.py — source de vérité, exposée via /openapi.json)

export interface Prediction {
  label: string;
  confidence: number;
}

export interface InputImage {
  filename: string;
  detected_view: string;
  view_prediction: Prediction;
  damage_prediction: Prediction;
  severity_prediction: Prediction;
  /** Faces documentées par cette photo. Une diagonale en couvre deux. */
  covered_faces: string[];
  quality_flag: "ok" | "blurry" | "dark" | "low_confidence" | string;
  /** Mesures brutes: variance du Laplacien et luminance moyenne. */
  quality: { blur: number; brightness: number };
  deduplicated: boolean;
  /** Fichier dont cette photo est un doublon perceptuel. */
  duplicate_of: string | null;
}

export interface DamagedPart {
  name: string;
  damage: string;
  severity: string;
  confidence: number;
  views_evidence: string[];
  detected_on: string[];
}

export interface ClaimResult {
  pipeline_version: string;
  status: string;
  vehicle_id: string | null;
  summary: string;
  damaged_parts: DamagedPart[];
  missing_photos: string[];
  /** Union des faces couvertes par le dossier. */
  covered_faces: string[];
  input_images: InputImage[];
  suspected_total_loss: boolean;
  confidence: number;
  raw_detections: unknown[];
  errors: string[];
  warnings: string[];
  processing_time_ms: number;
  inference_time_ms: number | null;
  duplicates_removed: number;
  image_level_damage: {
    damage: string;
    severity: string;
    confidence: number;
    evidence_image: string | null;
  };
}

export type ClaimState = "queued" | "processing" | "done" | "error" | "interrupted";

export interface ClaimStatus {
  claim_id: string;
  status: ClaimState;
  error?: string | null;
  created_at?: string | null;
  n_images?: number;
  result?: ClaimResult | null;
}
