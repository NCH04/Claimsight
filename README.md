# Assurance_agent

# Run commande:
python src/train_view_kfold.py \
  --csv_path dataset/labels.csv \
  --images_dir dataset/images_mapped \
  --k 5 \
  --epochs 35 \
  --batch_size 16 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --lr_step_size 8 \
  --lr_gamma 0.1 \
  --label_smoothing 0.05 \
  --freeze_backbone_epochs 1 \
  --img_size 256 \
  --mirror_prob 0.5 \
  --use_weighted_sampler \
  --backbone resnet34 \
  --seed 42

# Options clés
- Normalisation ImageNet + cadrage aléatoire (RandomResizedCrop), légère translation/rotation et jitter côté train (pas de flip).
- Flip horizontal optionnel avec swap automatique des labels gauche/droite: `--mirror_prob`.
- Optimizer AdamW + StepLR (par défaut: step_size=8, gamma=0.1).
- CrossEntropy avec label smoothing (0.05) et poids de classes inverses pour compenser le déséquilibre.
- Sur-échantillonnage optionnel des classes rares via WeightedRandomSampler: `--use_weighted_sampler`.
- Warmup optionnel du backbone: `--freeze_backbone_epochs N` (0 = désactivé).
- Les vues `closeup` sont exclues automatiquement du training.
- Option pour exclure `out_of_scope`: `--drop_out_of_scope`.

## How to run V1 pipeline (view + image-level damage/severity)
1) Place/convert votre checkpoint de vue dans `models/view.pt` avec les clés: `model`, `classes`, `arch`, `img_size`, `normalize` (format torch.save). Si absent, le pipeline lèvera une erreur et vous rappellera de le générer via `src/train_view_kfold.py`.
2) (Optionnel) Placez des checkpoints damage/severity au même format (sinon stubs “unknown” seront utilisés).
3) Exécutez:
```
python -m src.pipeline.run_pipeline \
  --images_dir dataset/images_mapped \
  --out_json outputs/result.json \
  --view_checkpoint models/view.pt
```
4) Exemple de sortie JSON (format aligné schema v1):
```json
{
  "pipeline_version": "v1",
  "status": "ok",
  "vehicle_id": null,
  "summary": "Front-left view shows moderate dent damage.",
  "damaged_parts": [],
  "missing_photos": ["right"],
  "input_images": [
    {
      "filename": "img1.jpg",
      "detected_view": "front-left",
      "view_prediction": {"label": "front-left", "confidence": 0.92},
      "damage_prediction": {"label": "dent", "confidence": 0.88},
      "severity_prediction": {"label": "moderate", "confidence": 0.85},
      "quality_flag": "ok",
      "deduplicated": false
    }
  ],
  "suspected_total_loss": false,
  "confidence": 0.90,
  "raw_detections": [],
  "errors": [],
  "processing_time_ms": 742,
  "image_level_damage": {"damage": "dent", "severity": "moderate", "confidence": 0.88},
  "views_detected": []  // debug/legacy
}
```
