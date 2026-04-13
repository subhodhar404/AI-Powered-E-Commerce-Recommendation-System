# Hybrid E-Commerce Recommendation System

A Flask-based recommendation system for e-commerce products that combines content similarity, semantic similarity, collaborative filtering, and popularity-based ranking.

## Overview

This project started as a notebook experiment and was later organized into a Flask web application. The final version keeps the recommendation workflow simple to use while making the ranking logic more informative and explainable.

Users can:

- browse a dataset-backed product catalog
- search products by name or brand
- choose a product as the recommendation seed
- provide a target user id for collaborative filtering
- view hybrid recommendation results with signal-level scores

## Recommendation Pipeline

The final ranking blends four signals:

1. `ContentScore`  
   TF-IDF similarity over cleaned product metadata.

2. `SemanticScore`  
   Semantic similarity using `SentenceTransformer` when available, with a TF-IDF + SVD fallback for local environments.

3. `CollaborativeScore`  
   User-user collaborative filtering based on the notebook-style user-item matrix built from dataset-derived user ids.

4. `PopularityScore`  
   A lightweight prior based on rating and review volume.

These signals are combined into a weighted hybrid score and used to rank the final recommendations.

## Why This Project Is More Than a Basic Filter

A basic filter can show products from the same category, the same brand, or the highest-rated items. This project goes further by combining multiple recommendation signals and exposing the reasoning behind each result.

Each recommendation card includes:

- the final hybrid score
- the strongest contributing signal
- the number of active signals supporting the item
- the individual keyword, semantic, collaborative, and popularity values

## Project Structure

```text
app.py                 Flask routes and page rendering
engine.py              Recommendation engine and data pipeline
templates/             Jinja templates
mechine_learning/      Static assets
dataset.tsv            Source dataset
requirements.txt       Python dependencies
Procfile               Deployment entry
```

## Main Pages

- `/index.html` - homepage with a default dataset-backed catalog
- `/products.html` - searchable product catalog
- `/recommandation.html` - hybrid recommendation page
- `/about.html` - project summary page

## API Endpoints

- `/api/health` - health status
- `/api/products` - catalog data in JSON format
- `/api/recommendations` - recommendation results in JSON format

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open the app:

```text
http://127.0.0.1:5000
```

## Dataset Note

The application expects a `dataset.tsv` file in the project root.

The collaborative component follows the preprocessing style used in the notebook by extracting a user identifier from the dataset's `Uniq Id` field and building a user-item matrix from those values. That collaborative signal is then combined with content, semantic, and popularity-based scoring.

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- sentence-transformers
- HTML / CSS / Jinja

## Notes

- Semantic embeddings load lazily to keep startup time lower.
- If `SentenceTransformer` is not available, the application falls back to a local semantic approximation using TF-IDF and SVD.
- The core recommendation logic lives in `engine.py`.

## Future Improvements

- add stronger offline evaluation for collaborative recommendations
- store data in a database instead of a flat file
- add authentication and saved user histories
- compare multiple ranking strategies side by side

## License

This project is licensed under the MIT License.
