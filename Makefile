synthesize:
	uv run --env-file=.env python src/jev_eval/synthesize.py

evaluate:
	uv run --env-file=.env python src/jev_eval/evaluate.py