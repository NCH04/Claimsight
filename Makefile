.PHONY: install install-cpu test lint pipeline sanity train-view train-damage train-severity train-parts yolo-config clean

PY ?= python
IMAGES ?= dataset/images_mapped
CSV    ?= dataset/labels.csv

install:                       ## Installe le projet + tous les extras
	$(PY) -m pip install -e ".[all]"

install-cpu:                   ## Idem mais torch CPU (plus léger)
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[all]"

test:                          ## Tests unitaires
	$(PY) -m pytest tests/ -q

lint:                          ## Lint + tri des imports
	$(PY) -m ruff check claimsight/ scripts/ tests/ sanity_check.py

train-view:                    ## Entraîne le classifieur de vue
	$(PY) -m claimsight.training.train_classifier --task view --csv_path $(CSV) --images_dir $(IMAGES) --final_fit

train-damage:                  ## Entraîne le classifieur de dégât
	$(PY) -m claimsight.training.train_classifier --task damage --csv_path $(CSV) --images_dir $(IMAGES) --final_fit

train-severity:                ## Entraîne le classifieur de gravité
	$(PY) -m claimsight.training.train_classifier --task severity --csv_path $(CSV) --images_dir $(IMAGES) --final_fit

yolo-config:                   ## (Re)génère configs/parts_yolo.yaml depuis la taxonomie
	$(PY) scripts/build_yolo_config.py

train-parts: yolo-config       ## Entraîne le détecteur de pièces
	$(PY) -m claimsight.training.train_parts_yolo

pipeline:                      ## Lance le pipeline sur $(IMAGES)
	$(PY) -m claimsight.pipeline.run_pipeline --images_dir $(IMAGES) --view_checkpoint models/view.pt

sanity:                        ## Contrôle rapide du pipeline
	$(PY) sanity_check.py --images_dir $(IMAGES) --view_checkpoint models/view.pt

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info
