test:
	python -m unittest discover -s tests -v

prepare:
	python scripts/preprocess_all.py
