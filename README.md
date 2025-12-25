# ImageEngine (Streamlit)

- Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

- Run with Docker Compose:

```bash
docker compose up --build
```

- App opens at: http://localhost:8501

- Project layout:
	- `src/`: core code (`classification.py`, `image_utils.py`, `logging_utils.py`, `metrics.py`)
	- `data/`: `celebrities.json`
	- `testsets/`: configs and indices
	- `image_outputs/`: saved annotated images
	- `test_outputs/`: CSV results

