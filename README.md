# Hybrid E-Commerce Recommendation System

A Flask-based web application that generates product recommendations from an e-commerce dataset using a hybrid ranking pipeline built on keyword similarity, semantic similarity, preference-profile matching, and popularity signals.

## Overview

This project began as a notebook experiment and was later organized into a cleaner Flask application for deployment and portfolio use. The final system focuses on product-to-product recommendation rather than simple filtering, with an interface for browsing the catalog, selecting a seed product, and shaping results with optional liked items.

The application is designed to be:

- practical enough to run locally as a small web app
- structured enough to present in a portfolio
- clear enough to explain during a project review or interview

## What the App Does

- loads product data from a TSV file
- cleans and normalizes product metadata
- builds a searchable catalog
- recommends similar products from a selected item
- accepts optional liked items to create a lightweight preference profile
- exposes both HTML pages and JSON endpoints

## Recommendation Pipeline

The final ranking combines four signals:

1. `ContentScore`  
   TF-IDF similarity over cleaned product text.

2. `SemanticScore`  
   Semantic similarity using `SentenceTransformer` when available, with a TF-IDF + SVD fallback for local environments.

3. `ProfileScore`  
   Similarity to a user preference profile built from manually provided liked products.

4. `PopularityScore`  
   A lightweight prior based on rating and review volume.

These signals are merged into a weighted hybrid score, which is then used to rank the final recommendations.

## Why This Is More Than a Basic Filter

A basic filter can show products from the same category, brand, or rating range. This project goes further by combining multiple learned similarity signals and exposing the reasoning behind the ranking.

Each recommendation result includes:

- the final score
- the dominant signal behind the result
- the number of signals supporting that item
- the individual keyword, semantic, profile, and popularity components

## Application Structure

```text
app.py                 Flask routes and page rendering
engine.py              Recommendation engine and data pipeline
templates/             Jinja templates
mechine_learning/      Static assets (CSS, JS, images, fonts)
dataset.tsv            Source dataset
requirements.txt       Python dependencies
Procfile               Deployment entry for production hosts
```

## Main Pages

- `/index.html` - homepage with a dataset-backed sample catalog
- `/products.html` - searchable product catalog
- `/recommandation.html` - recommendation interface
- `/about.html` - project summary page

## API Endpoints

- `/api/health` - application health status
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

Open the app in your browser:

```text
http://127.0.0.1:5000
```

## Dataset

The application expects a `dataset.tsv` file in the project root.

The current implementation is built around a product-centric dataset, so the deployed version does not claim full user-user collaborative filtering. Instead, it uses a more reliable hybrid approach based on product text, semantic similarity, preference seeds, and popularity.

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- sentence-transformers
- HTML / CSS / Jinja

## Notes

- Semantic embeddings load lazily to keep startup lighter.
- If `SentenceTransformer` is unavailable, the app falls back to a local semantic approximation using TF-IDF and SVD.
- The recommendation engine currently lives in `engine.py`, which is the main backend logic file used by the Flask app.

## Future Improvements

- add a real user-interaction dataset for stronger evaluation
- store products in a database instead of a flat file
- add authentication and saved preferences
- compare recommendation strategies with offline metrics

## License

This project is licensed under the MIT License.
