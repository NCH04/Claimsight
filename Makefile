.PHONY: install install-cpu test lint web-install web-dev web-build calibrate serve pipeline sanity check-skew calibrate-oof derive-coverage fetch-damage derive-damage annotate train-coverage train-view train-damage train-severity train-parts yolo-config clean

# Utilise le venv du projet s'il existe: `make` ne doit pas dépendre du fait
# que l'utilisateur ait activé l'environnement.
PY ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
IMAGES ?= dataset/images_mapped
CSV    ?= dataset/labels.csv
PARTS_DATASET ?= ../datasets/carparts-seg
DAMAGE_DATASET ?= ../datasets/car-damage-severity
DAMAGE_CSV ?= dataset/damage_mixed.csv
CKPT   ?= models/coverage.pt

install:                       ## Installe le projet + tous les extras
	$(PY) -m pip install -e ".[all]"

install-cpu:                   ## Idem mais torch CPU (plus léger)
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[all]"

test:                          ## Tests unitaires
	$(PY) -m pytest tests/ -q

lint:                          ## Lint + tri des imports
	$(PY) -m ruff check claimsight/ scripts/ tests/ sanity_check.py

derive-coverage:               ## Déduit des labels de couverture depuis un dataset de pièces
	$(PY) scripts/derive_coverage_labels.py --dataset $(PARTS_DATASET) --out dataset/coverage_derived.csv

fetch-damage:                  ## Télécharge le dataset de dégâts (exige ROBOFLOW_API_KEY)
	$(PY) scripts/fetch_roboflow_dataset.py \
	      --workspace car-damaged-detection-e66m0 \
	      --project car-damaged-severity-detection \
	      --version 29 --dest $(DAMAGE_DATASET)

derive-damage:                 ## Déduit les labels de dégât/gravité: images entières + crops de pièces
	$(PY) scripts/derive_damage_labels.py --dataset $(DAMAGE_DATASET) \
	      --none_from $(PARTS_DATASET)/images/train --out dataset/damage_derived.csv
	$(PY) scripts/derive_damage_crops.py --dataset $(DAMAGE_DATASET) \
	      --intact_from $(PARTS_DATASET) --out dataset/damage_crops.csv
	$(PY) scripts/merge_label_csv.py dataset/damage_derived.csv dataset/damage_crops.csv \
	      --out $(DAMAGE_CSV)

annotate:                      ## Ouvre l'outil d'annotation clavier
	$(PY) scripts/annotate_coverage.py --images_dir $(IMAGES)

train-coverage:                ## Entraîne le modèle de couverture photo (multi-label)
	$(PY) -m claimsight.training.train_classifier --task coverage --csv_path dataset/coverage_derived.csv --images_dir $(IMAGES) --final_fit

train-view:                    ## Entraîne le classifieur de vue (optionnel)
	$(PY) -m claimsight.training.train_classifier --task view --csv_path $(CSV) --images_dir $(IMAGES) --final_fit

train-damage:                  ## Entraîne le classifieur de dégât (images entières + crops)
	$(PY) -m claimsight.training.train_classifier --task damage --csv_path $(DAMAGE_CSV) \
	      --images_dir / --group_column photo_id --class_weights --amp --final_fit

train-severity:                ## Entraîne le classifieur de gravité (images entières + crops)
	$(PY) -m claimsight.training.train_classifier --task severity --csv_path $(DAMAGE_CSV) \
	      --images_dir / --group_column photo_id --class_weights --amp --final_fit

yolo-config:                   ## (Re)génère configs/parts_yolo.yaml depuis la taxonomie
	$(PY) scripts/build_yolo_config.py

train-parts: yolo-config       ## Entraîne le détecteur de pièces
	$(PY) -m claimsight.training.train_parts_yolo

web-install:                   ## Installe les dépendances du front
	cd web && npm install

web-dev:                       ## Front en développement (Vite :5173, proxy /api -> :8000)
	cd web && npm run dev

web-build:                     ## Construit le bundle servi par l'API
	cd web && npm run build

check-skew:                    ## Vérifie que le classifieur voit les crops comme les images
	$(PY) scripts/check_domain_skew.py --images_dir $(PARTS_DATASET)/images/val

calibrate:                     ## Calibre un checkpoint (température + seuils par classe)
	$(PY) -m claimsight.training.calibrate --checkpoint $(CKPT) --csv_path $(CSV) --images_dir $(IMAGES) --write

calibrate-oof:                 ## Calibre damage + severity hors-fold (après --final_fit)
	$(PY) -m claimsight.training.calibrate --checkpoint models/damage.pt \
	      --csv_path $(DAMAGE_CSV) --images_dir / --group_column photo_id \
	      --oof_from outputs/train/damage_mixed --write
	$(PY) -m claimsight.training.calibrate --checkpoint models/severity.pt \
	      --csv_path $(DAMAGE_CSV) --images_dir / --group_column photo_id \
	      --oof_from outputs/train/severity_mixed --write

serve:                         ## Lance l'API sur http://127.0.0.1:8000 (docs sur /docs)
	$(PY) -m uvicorn claimsight.api.server:app --reload

pipeline:                      ## Lance le pipeline sur $(IMAGES)
	$(PY) -m claimsight.pipeline.run_pipeline --images_dir $(IMAGES) --view_checkpoint models/view.pt

sanity:                        ## Contrôle rapide du pipeline
	$(PY) sanity_check.py --images_dir $(IMAGES)

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info
