# ML-Labs

This repository is a course-spanning machine learning workspace. It collects foundational lab notebooks, class tasks, fast experiments, supporting slides and reports, and a few small application-style projects built with Flask and Streamlit.

The work moves from early data handling and classical machine learning into clustering, neural networks, transformers, retrieval-augmented generation, and YOLO-based computer vision. It is best understood as a learning archive plus a set of prototype projects rather than a single packaged application.

## Repository at a glance

- Classical ML notebooks covering data loading, preprocessing, feature engineering, regression, classification, and evaluation
- Unsupervised learning work on clustering and related exploratory analysis
- Neural network exercises using TensorFlow and custom NumPy implementations
- NLP experiments with Hugging Face pipelines and GPT-2
- Small web apps built with Flask and Streamlit
- RAG and local-document question answering experiments
- A multi-task YOLO11 computer vision demo
- Course slides, PDFs, reports, exported notebooks, and zipped lab submissions

## Main contents

### 1. Core lab notebooks

Top-level notebooks contain the main machine learning lab progression:

- `1167-Lab1-BSCS_A_7.ipynb`
  Data loading practice with CSV, Excel, JSON, dictionaries, and URL-based sources using pandas.
- `1167-Lab2-BSCS-A_7.ipynb`
  Early regression concepts including logistic and polynomial regression, plus train/test split workflow.
- `1167-Lab3-BSCS_A_7.ipynb`
  Data preprocessing and feature selection.
- `1167-Lab3-BSCS_A_7_ClassTask.ipynb`
  Logistic regression from scratch, logistic regression with scikit-learn, and polynomial regression practice.
- `1167-Lab4-BSCS-A_7_Activity.ipynb`
  KNN, SVM, voting classifiers, and credit card fraud detection with SMOTE on imbalanced data.
- `1167-Lab4-BSCS_A.ipynb`
  Short notebook tied to the imbalance topic from Lab 4.
- `1167-Lab5-BSCS_A_7_Activity.ipynb`
  Unsupervised learning and clustering using the Mall Customers dataset.
- `1167-Lab5-BSCS_A_7_ClassTask.ipynb`
  Unsupervised learning workflow on the mushrooms dataset with exploration and visualization.
- `1167-Lab6-BSCS_A_7_ClassTask.ipynb`
  Unsupervised learning work on the wine dataset with scaling and analysis setup.
- `1167-Lab8-9-BSCS_A_ClassTask.ipynb`
  ANN work on Iris using TensorFlow/Keras, including hidden-layer activation comparisons.

### 2. Fast labs and experiments

- `fast_lab_1.ipynb`
  Synthetic moons dataset work with augmentation, visualization, and model-focused experimentation.
- `fast_lab_2.ipynb`
  A more hands-on deep learning exercise with a custom `DeepNeuralNetwork` implementation built in NumPy, including forward pass, backpropagation, and training helpers.
- `test.ipynb`
  Small scratch notebook.

### 3. NLP and transformers

The `Lab10/` folder contains transformer-focused experiments:

- `Lab10/1167-Lab10-BSCS_A_ClassTask.ipynb`
  Text classification, summarization, translation, and image-to-text oriented tasks.
- `Lab10/simple_emotion_pipeline.ipynb`
  Emotion/sentiment classification with Hugging Face pipelines.
- `Lab10/Transformers Hugging face.ipynb`
  Summarization and translation pipeline experiments.
- `Lab10/Text_Generation_with__GPT2.ipynb`
  GPT-2 text generation workflow.

### 4. Reports and supporting material

- `Ensemble Report/`
  Ensemble learning notebooks, an exported HTML notebook, and PDF reports.
- `ML_Lab1.pptx`, `ML_Lab2_3.pptx`, `ML_Lab4.pptx`, `ML_Lab_05.pptx`, `ML_Lab7.pptx`
  Course slide decks used alongside the labs.
- `Lab 08-09 - Instructions.pdf`
  Instructions for the later neural-network-related labs.
- `Lab11/*.pdf`
  PDF assets related to the final lab and offline PDF QA work.

### 5. App-style projects

These are the most application-oriented parts of the repository.

#### `Flask/`

A basic Flask + SQLAlchemy to-do app with Jinja templates and SQLite persistence.

Features visible in the code:

- Create, edit, list, and delete tasks
- SQLite-backed storage via SQLAlchemy
- Template inheritance with Jinja
- Simple Tailwind-oriented template structure

Key files:

- `Flask/app.py`
- `Flask/templates/`
- `instance/database.db`

#### `Lab11/part1_flask/`

A Flask interface for training and evaluating simple ML models against either built-in datasets or uploaded CSV files.

Features visible in the code:

- Built-in datasets: Iris, Digits, Wine
- Models: Logistic Regression, KNN, SVM
- Standard train/test workflow with scaling
- Metrics: accuracy, precision, recall, F1-score
- Auto-generated bar chart of evaluation results

Key files:

- `Lab11/part1_flask/app.py`
- `Lab11/part1_flask/templates/index.html`
- `Lab11/part1_flask/static/metrics.png`

#### `Lab11/part2_RAG_StreamLit/`

A Streamlit RAG prototype for asking questions over an uploaded PDF document.

Features visible in the code:

- PDF ingestion with `PyPDF2`
- Text chunking with LangChain text splitters
- Embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Vector search using FAISS
- Local answer generation through GPT4All

Important note:

- The code currently points to a local GGUF model path outside this repository, and `.gitignore` also excludes `q4_0-orca-mini-3b.gguf`. That means the app is not fully self-contained out of the box without supplying a local GPT4All model.

Key file:

- `Lab11/part2_RAG_StreamLit/RAG.py`

#### `Lab11/Lab11_HomeTask/yolo/`

A more ambitious Flask-based computer vision demo wrapping multiple YOLO11 tasks behind separate routes.

Tasks implemented:

- Object detection
- Instance segmentation
- Image classification
- Pose estimation
- Oriented bounding box detection

Features visible in the code:

- Individual Flask endpoints per task
- Separate service functions for each inference path
- A single HTML front end for uploading images and viewing structured results
- Pretrained YOLO11 weights stored in the repo under `models/`

Key files:

- `Lab11/Lab11_HomeTask/yolo/app.py`
- `Lab11/Lab11_HomeTask/yolo/routes/yolo_routes.py`
- `Lab11/Lab11_HomeTask/yolo/services/yolo_service.py`
- `Lab11/Lab11_HomeTask/yolo/models/yolo_model.py`
- `Lab11/Lab11_HomeTask/yolo/templates/main.html`

### 6. Miscellaneous notes

- `Playwright/PlaywrightNotes.ipynb`
  Browser automation notes separate from the ML material.
- `Lab11/part1_flask.zip`, `Lab11/part2_RAG_StreamLit.zip`, `Lab11/Lab11_HomeTask/1167-lab11-hometask.zip`
  Archived project bundles/submissions.

## Tech stack

Based on `pyproject.toml`, the repository uses or references:

- Python 3.12
- Jupyter Notebook / IPython kernel
- NumPy, pandas, matplotlib, seaborn
- scikit-learn, imbalanced-learn, Optuna, Keras Tuner
- TensorFlow
- PyTorch, torchvision, torchaudio
- Hugging Face `transformers`, `sentencepiece`, `sacremoses`
- Diffusers
- Flask, Flask-SQLAlchemy, Streamlit
- PyPDF2, LangChain, sentence-transformers, FAISS, GPT4All
- Ultralytics YOLO
- `uv` for environment management
- `ruff` for linting

## Getting started

### Environment setup

This repo is configured as a Python project with `uv`.

```bash
uv sync
```

If you prefer working directly in notebooks:

```bash
uv run jupyter notebook
```

### Running the app demos

Flask to-do app:

```bash
uv run python Flask/app.py
```

Flask ML evaluation app:

```bash
uv run python Lab11/part1_flask/app.py
```

Streamlit RAG app:

```bash
uv run streamlit run Lab11/part2_RAG_StreamLit/RAG.py
```

YOLO multi-task demo:

```bash
uv run python Lab11/Lab11_HomeTask/yolo/app.py
```

## Notes and limitations

- Some notebooks expect local datasets under `Datasets/`, and that folder is ignored in `.gitignore`.
- The RAG prototype depends on a locally available GPT4All model file that is not committed.
- A number of files in the repository are course artifacts, exports, or archives rather than polished reusable modules.
- There is a committed SQLite database under `instance/database.db` for the Flask to-do example.
- The repository currently reads best as a documented learning portfolio rather than a single install-and-run product.

## Recommended way to read the repo

If you want to review the work in sequence:

1. Start with the top-level numbered lab notebooks for the course progression.
2. Move to `fast_lab_1.ipynb` and `fast_lab_2.ipynb` for more implementation-heavy experiments.
3. Review `Lab10/` for transformer and NLP work.
4. Review `Ensemble Report/` for the report-oriented material.
5. Finish with the application-style work in `Flask/` and `Lab11/`.

## Best standalone portfolio candidates

If this repository is later split into smaller, presentation-ready projects, the strongest candidates are:

1. `Lab11/Lab11_HomeTask/yolo/`
   Best standalone demo in the repo. It has a clear product shape, multiple computer vision tasks, API-style routes, and a presentable UI.
2. `Lab11/part2_RAG_StreamLit/`
   Strong portfolio topic because local-document RAG is easy to explain and relevant, though it needs cleanup around model paths and setup.
3. `Lab11/part1_flask/`
   Good smaller project showing the bridge between machine learning and web deployment, especially if expanded with better dataset handling and result persistence.

## Summary

This repository captures a broad machine learning learning journey: from data loading and preprocessing, through classical ML and unsupervised learning, into neural networks, NLP, RAG, and computer vision deployment demos. It is a useful archive of both coursework and practical experimentation, with the Lab 11 application projects standing out as the best material for standalone portfolio use.
