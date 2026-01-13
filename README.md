# Assurance_agent

# Run commande:
python src/train_view_kfold.py \
  --csv_path dataset/labels.csv \
  --images_dir dataset/images_mapped \
  --k 5 \
  --epochs 25 \
  --batch_size 16 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --lr_step_size 8 \
  --lr_gamma 0.1 \
  --label_smoothing 0.05 \
  --freeze_backbone_epochs 1 \
  --img_size 256 \
  --mirror_prob 0.5 \
  --use_weighted_sampler \
  --drop_closeup

# Options clés
- Normalisation ImageNet + cadrage aléatoire (RandomResizedCrop), légère translation/rotation et jitter côté train (pas de flip).
- Flip horizontal optionnel avec swap automatique des labels gauche/droite: `--mirror_prob`.
- Optimizer AdamW + StepLR (par défaut: step_size=8, gamma=0.1).
- CrossEntropy avec label smoothing (0.05) et poids de classes inverses pour compenser le déséquilibre.
- Sur-échantillonnage optionnel des classes rares via WeightedRandomSampler: `--use_weighted_sampler`.
- Warmup optionnel du backbone: `--freeze_backbone_epochs N` (0 = désactivé).
