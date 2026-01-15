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
4) Exemple de sortie JSON:
```json
{
  "summary": "Front-left view shows moderate dent damage.",
  "views_detected": [
    {"image": "img1.jpg", "view": "front-left", "confidence": 0.92},
    {"image": "img2.jpg", "view": "rear", "confidence": 0.88}
  ],
  "missing_photos": ["right"],
  "image_level_damage": {"damage": "dent", "severity": "moderate", "confidence": 0.88},
  "damaged_parts": [],
  "confidence": 0.90
}
```
