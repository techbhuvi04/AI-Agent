.PHONY: install generate test eval demo

install:
	pip install -r requirements.txt

generate:
	python3 -m generator.cli --seed 42 --difficulty medium --output-dir data/

test:
	python3 -m pytest tests/ -v

eval:
	python3 -m eval.harness --data-dir data/ --ablation --max-tier 4

curve:
	python3 -m eval.harness --data-dir data/ --curve --max-tier 4

demo:
	streamlit run demo/app.py --server.headless true
