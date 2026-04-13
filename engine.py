import difflib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional runtime dependency.
    SentenceTransformer = None


DATASET_PATH = Path("dataset.tsv")
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_HYBRID_WEIGHTS = {
    "ContentScore": 0.20,
    "SemanticScore": 0.35,
    "CollaborativeScore": 0.30,
    "PopularityScore": 0.15,
}


# Return an empty frame with the standard recommendation columns.
def empty_recommendation_frame():
    return pd.DataFrame(
        columns=[
            "ProdID",
            "RawProdID",
            "Name",
            "Brand",
            "ImageURL",
            "ReviewCount",
            "Rating",
            "ContentScore",
            "SemanticScore",
            "CollaborativeScore",
            "PopularityScore",
            "Score",
            "BestSignal",
            "SignalCount",
        ]
    )


# Extract a notebook-compatible user id from the raw row identifier.
def extract_user_id(raw_row_id):
    match = re.search(r"(\d+)", str(raw_row_id))
    return match.group(1) if match else ""


# Load the dataset, clean the fields, and build stable product and user ids.
def load_and_prepare_data(dataset_path=DATASET_PATH):
    df = pd.read_csv(dataset_path, sep="	")

    selected_columns = [
        "Uniq Id",
        "Product Id",
        "Product Rating",
        "Product Reviews Count",
        "Product Category",
        "Product Brand",
        "Product Name",
        "Product Image Url",
        "Product Description",
        "Product Tags",
    ]
    df = df[selected_columns].copy()

    df["Product Rating"] = pd.to_numeric(df["Product Rating"], errors="coerce").fillna(0)
    df["Product Reviews Count"] = pd.to_numeric(
        df["Product Reviews Count"], errors="coerce"
    ).fillna(0)
    df["Product Category"] = df["Product Category"].fillna("")
    df["Product Brand"] = df["Product Brand"].fillna("")
    df["Product Description"] = df["Product Description"].fillna("")
    df["Product Tags"] = df["Product Tags"].fillna("")

    column_name_mapping = {
        "Uniq Id": "RawRowID",
        "Product Id": "RawProdID",
        "Product Rating": "Rating",
        "Product Reviews Count": "ReviewCount",
        "Product Category": "Category",
        "Product Brand": "Brand",
        "Product Name": "Name",
        "Product Image Url": "ImageURL",
        "Product Description": "Description",
        "Product Tags": "Tags",
    }
    df = df.rename(columns=column_name_mapping)

    df["RawRowID"] = df["RawRowID"].astype(str).str.strip()
    df["RawProdID"] = df["RawProdID"].astype(str).str.strip()
    df["UserID"] = df["RawRowID"].apply(extract_user_id)
    df = df[
        (df["RawProdID"] != "")
        & (df["RawProdID"].str.lower() != "nan")
        & (df["Name"].astype(str).str.strip() != "")
        & (df["UserID"] != "")
    ].copy()

    product_keys = sorted(df["RawProdID"].unique().tolist())
    prod_map = {value: idx + 1 for idx, value in enumerate(product_keys)}
    df["ProdID"] = df["RawProdID"].map(prod_map).astype(int)

    stop_words = set(ENGLISH_STOP_WORDS)
    text_columns = ["Category", "Brand", "Description"]
    for col in text_columns:
        df[col] = df[col].apply(lambda text: fast_clean(text, stop_words))

    df["ImageURL"] = df["ImageURL"].apply(normalize_image_url)
    df["Tags"] = df[text_columns].apply(lambda row: ", ".join(row), axis=1)
    df = df.drop_duplicates(subset=["RawRowID", "RawProdID"]).reset_index(drop=True)
    return df


# Normalize text by lowercasing, stripping punctuation, and removing stop words.
def fast_clean(text, stop_words):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)


# Keep the first valid image URL from a raw image field.
def normalize_image_url(value):
    value = "" if pd.isna(value) else str(value).strip()
    if not value:
        return ""

    candidates = [part.strip() for part in value.split("|") if part.strip()]
    for candidate in candidates:
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate
    return candidates[0] if candidates else ""


# Build a product-level table from the raw review rows.
def build_item_frame(df):
    item_df = (
        df.sort_values(by=["ReviewCount", "Rating"], ascending=[False, False])
        .groupby("ProdID", as_index=False)
        .agg(
            {
                "RawProdID": "first",
                "Name": "first",
                "Brand": "first",
                "ImageURL": "first",
                "ReviewCount": "max",
                "Rating": "mean",
                "Category": "first",
                "Description": "first",
                "Tags": "first",
            }
        )
    )
    item_df["SemanticText"] = (
        item_df[["Name", "Brand", "Category", "Description", "Tags"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    item_df["PopularityScore"] = build_popularity_score(item_df)
    return item_df.reset_index(drop=True)


# Combine ratings and review volume into a simple popularity prior.
def build_popularity_score(item_df):
    rating_component = item_df["Rating"].fillna(0).clip(lower=0, upper=5) / 5.0
    review_component = np.log1p(item_df["ReviewCount"].fillna(0))
    max_review = review_component.max() if len(review_component) else 0
    if max_review > 0:
        review_component = review_component / max_review
    return (0.7 * rating_component + 0.3 * review_component).round(4)


# Return the strongest default products using popularity and rating signals.
def rating_based_recommendation(df, top_n=10):
    item_df = build_item_frame(df)
    return item_df.sort_values(
        by=["PopularityScore", "Rating", "ReviewCount"], ascending=[False, False, False]
    ).head(top_n)


# Resolve free-text input to the closest product name in the catalog.
def resolve_item_name(query, item_df):
    if not query:
        return ""

    query = query.strip()
    names = item_df["Name"].tolist()
    lower_to_original = {name.lower(): name for name in names}
    query_lower = query.lower()

    if query in names:
        return query
    if query_lower in lower_to_original:
        return lower_to_original[query_lower]

    contains_matches = [
        name
        for name in names
        if query_lower in name.lower() or name.lower() in query_lower
    ]
    if contains_matches:
        return contains_matches[0]

    close_matches = difflib.get_close_matches(query, names, n=1, cutoff=0.5)
    return close_matches[0] if close_matches else ""


# Resolve a requested user id against the collaborative model index.
def resolve_user_id(target_user_id, collaborative_model):
    user_item_matrix = collaborative_model.get("user_item_matrix")
    default_user_id = str(collaborative_model.get("default_user_id", "") or "")
    if user_item_matrix is None or user_item_matrix.empty:
        return ""

    available_ids = [str(user_id) for user_id in user_item_matrix.index.tolist()]
    available_lookup = set(available_ids)
    query = str(target_user_id).strip() if target_user_id is not None else ""

    if not query:
        return default_user_id or available_ids[0]
    if query in available_lookup:
        return query
    if query.endswith(".0") and query[:-2] in available_lookup:
        return query[:-2]

    digits = re.search(r"(\d+)", query)
    if digits and digits.group(1) in available_lookup:
        return digits.group(1)

    return default_user_id or available_ids[0]


# Build the TF-IDF model used for keyword-based similarity.
def build_content_model(df):
    item_df = build_item_frame(df)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(item_df["Tags"].fillna(""))
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return {
        "item_df": item_df,
        "vectorizer": vectorizer,
        "similarity_matrix": similarity_matrix,
    }


# Create a lightweight semantic fallback model with TF-IDF and SVD.
def build_lsa_embeddings(text_series):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        max_features=4000,
    )
    tfidf_matrix = vectorizer.fit_transform(text_series.tolist())

    if min(tfidf_matrix.shape) <= 2:
        embeddings = normalize(tfidf_matrix.toarray())
    else:
        n_components = max(2, min(64, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = normalize(svd.fit_transform(tfidf_matrix))
    return embeddings, "tfidf-lsa-fallback"


# Build the semantic model using SentenceTransformers or the local fallback.
def build_semantic_model(df, model_name=SEMANTIC_MODEL_NAME):
    item_df = build_item_frame(df)
    text_series = item_df["SemanticText"].fillna("")

    backend = "sentence-transformers"
    if SentenceTransformer is not None:
        try:
            encoder = SentenceTransformer(model_name)
            embeddings = encoder.encode(
                text_series.tolist(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            embeddings, backend = build_lsa_embeddings(text_series)
            model_name = "tfidf-lsa-fallback"
    else:
        embeddings, backend = build_lsa_embeddings(text_series)
        model_name = "tfidf-lsa-fallback"

    similarity_matrix = cosine_similarity(embeddings, embeddings)
    return {
        "item_df": item_df,
        "embeddings": embeddings,
        "similarity_matrix": similarity_matrix,
        "backend": backend,
        "model_name": model_name,
    }


# Build the collaborative model from the notebook-style user-item matrix.
def build_collaborative_model(df):
    item_df = build_item_frame(df)
    if df.empty:
        return {
            "item_df": item_df,
            "user_item_matrix": pd.DataFrame(),
            "similarity_matrix": np.empty((0, 0)),
            "default_user_id": "",
            "user_count": 0,
        }

    user_item_matrix = df.pivot_table(
        index="UserID", columns="ProdID", values="Rating", aggfunc="mean"
    ).fillna(0)
    similarity_matrix = cosine_similarity(user_item_matrix) if not user_item_matrix.empty else np.empty((0, 0))
    user_activity = df.groupby("UserID")["ProdID"].nunique().sort_values(ascending=False)
    default_user_id = str(user_activity.index[0]) if not user_activity.empty else ""
    return {
        "item_df": item_df,
        "user_item_matrix": user_item_matrix,
        "similarity_matrix": similarity_matrix,
        "default_user_id": default_user_id,
        "user_count": int(user_item_matrix.index.nunique()),
    }


# Shape a scored recommendation frame for a single ranking signal.
def create_scored_frame(item_df, selected_indices, score_map, score_column):
    if not selected_indices:
        return empty_recommendation_frame()

    recs = item_df.iloc[selected_indices][
        ["ProdID", "RawProdID", "Name", "Brand", "ImageURL", "ReviewCount", "Rating"]
    ].copy()
    recs["ContentScore"] = 0.0
    recs["SemanticScore"] = 0.0
    recs["CollaborativeScore"] = 0.0
    recs["PopularityScore"] = recs["ProdID"].map(
        item_df.set_index("ProdID")["PopularityScore"]
    ).fillna(0.0)
    recs[score_column] = recs["ProdID"].map(score_map).fillna(0.0)
    recs["Score"] = recs[score_column]
    recs["BestSignal"] = score_column.replace("Score", "")
    recs["SignalCount"] = (recs[[score_column]] > 0).sum(axis=1)
    return recs.reset_index(drop=True)


# Return recommendations based on TF-IDF keyword similarity.
def content_based_recommendations(df, item_name, top_n=10, content_model=None):
    if content_model is None:
        content_model = build_content_model(df)

    item_df = content_model["item_df"]
    resolved_name = resolve_item_name(item_name, item_df)
    if not resolved_name:
        return empty_recommendation_frame(), ""

    item_index = item_df[item_df["Name"] == resolved_name].index[0]
    similarity_matrix = content_model["similarity_matrix"]
    similarity_scores = list(enumerate(similarity_matrix[item_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

    selected_indices = [idx for idx, _ in similarity_scores]
    score_map = {item_df.iloc[idx]["ProdID"]: score for idx, score in similarity_scores}
    return create_scored_frame(item_df, selected_indices, score_map, "ContentScore"), resolved_name


# Return recommendations based on semantic embedding similarity.
def semantic_embedding_recommendations(df, item_name, top_n=10, semantic_model=None):
    if semantic_model is None:
        semantic_model = build_semantic_model(df)

    item_df = semantic_model["item_df"]
    resolved_name = resolve_item_name(item_name, item_df)
    if not resolved_name:
        return empty_recommendation_frame(), ""

    item_index = item_df[item_df["Name"] == resolved_name].index[0]
    similarity_matrix = semantic_model["similarity_matrix"]
    similarity_scores = list(enumerate(similarity_matrix[item_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

    selected_indices = [idx for idx, _ in similarity_scores]
    score_map = {item_df.iloc[idx]["ProdID"]: score for idx, score in similarity_scores}
    return create_scored_frame(item_df, selected_indices, score_map, "SemanticScore"), resolved_name


# Return recommendations from user-user similarity over the notebook-style interaction matrix.
def collaborative_filtering_recommendations(
    df,
    target_user_id,
    top_n=10,
    collaborative_model=None,
    min_rating=4,
):
    if collaborative_model is None:
        collaborative_model = build_collaborative_model(df)

    item_df = collaborative_model["item_df"]
    user_item_matrix = collaborative_model["user_item_matrix"]
    similarity_matrix = collaborative_model["similarity_matrix"]
    resolved_user_id = resolve_user_id(target_user_id, collaborative_model)

    if not resolved_user_id or user_item_matrix.empty:
        return empty_recommendation_frame(), ""

    target_user_index = user_item_matrix.index.get_loc(resolved_user_id)
    similar_user_indices = similarity_matrix[target_user_index].argsort()[::-1][1:]
    seen_items = set(df[df["UserID"] == resolved_user_id]["ProdID"])
    candidate_scores = {}

    for user_index in similar_user_indices:
        similarity_score = float(similarity_matrix[target_user_index][user_index])
        if similarity_score <= 0:
            continue

        similar_user_id = str(user_item_matrix.index[user_index])
        similar_user_rows = df[df["UserID"] == similar_user_id]
        for _, row in similar_user_rows.iterrows():
            prod_id = row["ProdID"]
            if prod_id in seen_items or row["Rating"] < min_rating:
                continue
            candidate_scores[prod_id] = candidate_scores.get(prod_id, 0.0) + similarity_score * float(row["Rating"])

    if not candidate_scores:
        return empty_recommendation_frame(), resolved_user_id

    max_score = max(candidate_scores.values())
    if max_score > 0:
        candidate_scores = {
            prod_id: score / max_score for prod_id, score in candidate_scores.items()
        }

    top_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    score_map = dict(top_items)
    selected_ids = [prod_id for prod_id, _ in top_items]
    selected_indices = item_df[item_df["ProdID"].isin(selected_ids)].index.tolist()
    recs = create_scored_frame(item_df, selected_indices, score_map, "CollaborativeScore")
    recs = recs.sort_values(by="CollaborativeScore", ascending=False).reset_index(drop=True)
    return recs, resolved_user_id


# Return a fallback list driven by rating and review-backed popularity.
def popularity_recommendations(df, top_n=10, item_df=None, excluded_names=None):
    item_df = item_df if item_df is not None else build_item_frame(df)
    excluded_names = set(excluded_names or [])
    recs = item_df[~item_df["Name"].isin(excluded_names)].copy()
    recs = recs.sort_values(
        by=["PopularityScore", "Rating", "ReviewCount"], ascending=[False, False, False]
    ).head(top_n)
    recs["ContentScore"] = 0.0
    recs["SemanticScore"] = 0.0
    recs["CollaborativeScore"] = 0.0
    recs["Score"] = recs["PopularityScore"]
    recs["BestSignal"] = "Popularity"
    recs["SignalCount"] = (recs[["PopularityScore"]] > 0).sum(axis=1)
    return recs[
        [
            "ProdID",
            "RawProdID",
            "Name",
            "Brand",
            "ImageURL",
            "ReviewCount",
            "Rating",
            "ContentScore",
            "SemanticScore",
            "CollaborativeScore",
            "PopularityScore",
            "Score",
            "BestSignal",
            "SignalCount",
        ]
    ].reset_index(drop=True)


# Combine content, semantic, collaborative, and popularity signals into one ranking.
def hybrid_recommendation(
    df,
    item_name,
    target_user_id=None,
    top_n=10,
    content_model=None,
    semantic_model=None,
    collaborative_model=None,
    weights=None,
):
    weights = weights or DEFAULT_HYBRID_WEIGHTS
    if content_model is None:
        content_model = build_content_model(df)
    if semantic_model is None:
        semantic_model = build_semantic_model(df)
    if collaborative_model is None:
        collaborative_model = build_collaborative_model(df)

    content_rec, resolved_seed_item = content_based_recommendations(
        df, item_name, top_n=top_n * 2, content_model=content_model
    )
    semantic_rec, _ = semantic_embedding_recommendations(
        df, item_name, top_n=top_n * 2, semantic_model=semantic_model
    )
    collaborative_rec, resolved_target_user = collaborative_filtering_recommendations(
        df,
        target_user_id,
        top_n=top_n * 2,
        collaborative_model=collaborative_model,
    )

    item_df = semantic_model["item_df"][
        [
            "ProdID",
            "RawProdID",
            "Name",
            "Brand",
            "ImageURL",
            "ReviewCount",
            "Rating",
            "PopularityScore",
        ]
    ].copy()

    seen_prod_ids = set(df[df["UserID"] == resolved_target_user]["ProdID"]) if resolved_target_user else set()
    seed_prod_ids = set(item_df[item_df["Name"] == resolved_seed_item]["ProdID"]) if resolved_seed_item else set()
    excluded_prod_ids = seen_prod_ids | seed_prod_ids
    excluded_names = item_df[item_df["ProdID"].isin(excluded_prod_ids)]["Name"].tolist()
    popularity_rec = popularity_recommendations(
        df, top_n=top_n * 2, item_df=semantic_model["item_df"], excluded_names=excluded_names
    )

    combined = item_df.merge(
        content_rec[["ProdID", "ContentScore"]], on="ProdID", how="left"
    ).merge(
        semantic_rec[["ProdID", "SemanticScore"]], on="ProdID", how="left"
    ).merge(
        collaborative_rec[["ProdID", "CollaborativeScore"]], on="ProdID", how="left"
    ).merge(
        popularity_rec[["ProdID", "PopularityScore"]], on="ProdID", how="left", suffixes=("", "_pop")
    )

    if "PopularityScore_pop" in combined.columns:
        combined["PopularityScore"] = combined["PopularityScore_pop"].fillna(
            combined["PopularityScore"]
        )
        combined = combined.drop(columns=["PopularityScore_pop"])

    score_columns = list(DEFAULT_HYBRID_WEIGHTS.keys())
    combined[score_columns] = combined[score_columns].fillna(0.0)
    if excluded_prod_ids:
        combined = combined[~combined["ProdID"].isin(excluded_prod_ids)].copy()
    combined = combined[combined[score_columns].sum(axis=1) > 0].copy()

    if combined.empty:
        return empty_recommendation_frame(), {
            "ResolvedSeedItem": resolved_seed_item,
            "ResolvedTargetUser": resolved_target_user,
        }

    combined["Score"] = sum(combined[col] * weights[col] for col in score_columns)
    best_signal_labels = {
        "ContentScore": "Keyword Similarity",
        "SemanticScore": "Semantic Embedding",
        "CollaborativeScore": "Collaborative Filtering",
        "PopularityScore": "Popularity",
    }
    combined["BestSignal"] = combined[score_columns].idxmax(axis=1).map(best_signal_labels)
    combined["SignalCount"] = (combined[score_columns] > 0).sum(axis=1)

    return (
        combined.sort_values(by=["Score", "SignalCount"], ascending=[False, False]).head(top_n),
        {
            "ResolvedSeedItem": resolved_seed_item,
            "ResolvedTargetUser": resolved_target_user,
        },
    )


# Keep evaluation disabled when the dataset cannot support honest offline metrics.
def evaluate_hybrid_recommender(*args, **kwargs):
    return {}


# Print a short dataset summary for local inspection.
def print_project_summary(df):
    print("Number of rows:", len(df))
    print("Number of products:", df["ProdID"].nunique())
    print("Number of users:", df["UserID"].nunique())
    print("Number of raw product ids:", df["RawProdID"].nunique())


# Find the dataset file from the common local project paths.
def discover_dataset_path():
    candidates = [
        DATASET_PATH,
        Path("dataset.tsv"),
        Path("data/dataset.tsv"),
        Path("dataset/data.tsv"),
    ]
    for path in candidates:
        if path.exists():
            return path

    for path in Path(".").rglob("*.tsv"):
        return path
    return DATASET_PATH


# Return the catalog page data with optional search and item limits.
def get_catalog(df, limit=None, query=None):
    if df.empty:
        return pd.DataFrame()

    catalog = build_item_frame(df).sort_values(
        by=["PopularityScore", "Rating", "ReviewCount"], ascending=[False, False, False]
    )

    if query:
        query = query.strip().lower()
        catalog = catalog[
            catalog["Name"].str.lower().str.contains(query, na=False)
            | catalog["Brand"].str.lower().str.contains(query, na=False)
            | catalog["Category"].str.lower().str.contains(query, na=False)
        ]

    if limit:
        catalog = catalog.head(limit)
    return catalog.reset_index(drop=True)


# Convert a recommendation frame into JSON-friendly product dictionaries.
def dataframe_to_products(df):
    if df.empty:
        return []

    products = []
    for _, row in df.iterrows():
        products.append(
            {
                "ProdID": int(row["ProdID"]),
                "RawProdID": str(row.get("RawProdID", "") or ""),
                "Name": str(row["Name"]),
                "Brand": "" if pd.isna(row.get("Brand")) else str(row.get("Brand")),
                "ImageURL": "" if pd.isna(row.get("ImageURL")) else str(row.get("ImageURL")),
                "ReviewCount": int(float(row.get("ReviewCount", 0) or 0)),
                "Rating": round(float(row.get("Rating", 0) or 0), 2),
                "Score": round(float(row.get("Score", 0) or 0), 4),
                "ContentScore": round(float(row.get("ContentScore", 0) or 0), 4),
                "SemanticScore": round(float(row.get("SemanticScore", 0) or 0), 4),
                "CollaborativeScore": round(float(row.get("CollaborativeScore", 0) or 0), 4),
                "PopularityScore": round(float(row.get("PopularityScore", 0) or 0), 4),
                "BestSignal": str(row.get("BestSignal", "") or ""),
                "SignalCount": int(float(row.get("SignalCount", 0) or 0)),
            }
        )
    return products


# Load the semantic model only when the app actually needs it.
def ensure_semantic_model(state):
    if state.get("semantic_model") is None and not state["data"].empty:
        semantic_model = build_semantic_model(state["data"])
        state["semantic_model"] = semantic_model
        state["semantic_backend"] = semantic_model["backend"]
        state["semantic_model_name"] = semantic_model["model_name"]
    return state.get("semantic_model")


# Build and cache the application state shared across requests.
@lru_cache(maxsize=1)
def build_app_state():
    dataset_path = discover_dataset_path()
    if not dataset_path.exists():
        return {
            "dataset_path": str(dataset_path),
            "data": pd.DataFrame(),
            "catalog": pd.DataFrame(),
            "content_model": None,
            "semantic_model": None,
            "collaborative_model": None,
            "semantic_backend": "not-loaded",
            "semantic_model_name": "",
            "metrics": {},
            "summary": {
                "rows": 0,
                "products": 0,
                "users": 0,
                "data_limitation": "Dataset not found.",
            },
        }

    data = load_and_prepare_data(dataset_path)
    catalog = get_catalog(data)
    content_model = build_content_model(data)
    collaborative_model = build_collaborative_model(data)

    return {
        "dataset_path": str(dataset_path),
        "data": data,
        "catalog": catalog,
        "content_model": content_model,
        "semantic_model": None,
        "collaborative_model": collaborative_model,
        "semantic_backend": "lazy-load",
        "semantic_model_name": "",
        "metrics": {},
        "summary": {
            "rows": int(len(data)),
            "products": int(data["ProdID"].nunique()),
            "users": int(data["UserID"].nunique()),
            "data_limitation": (
                "Collaborative filtering follows the notebook preprocessing by extracting user ids "
                "from the dataset's Uniq Id field, then combining that signal with content, semantic, and popularity scores."
            ),
        },
    }


if __name__ == "__main__":
    dataset_path = discover_dataset_path()

    if not dataset_path.exists():
        print("dataset.tsv file was not found in the current folder.")
        print("Keep dataset.tsv beside engine.py and run the script again.")
    else:
        data = load_and_prepare_data(dataset_path)
        print_project_summary(data)

        content_model = build_content_model(data)
        semantic_model = build_semantic_model(data)
        collaborative_model = build_collaborative_model(data)
        default_user_id = collaborative_model.get("default_user_id", "")
        sample_item = data["Name"].iloc[0]
        sample_recommendations, meta = hybrid_recommendation(
            data,
            sample_item,
            target_user_id=default_user_id,
            top_n=5,
            content_model=content_model,
            semantic_model=semantic_model,
            collaborative_model=collaborative_model,
        )
        print("\nSample final hybrid recommendations:")
        print(
            sample_recommendations[
                [
                    "Name",
                    "Score",
                    "ContentScore",
                    "SemanticScore",
                    "CollaborativeScore",
                    "PopularityScore",
                    "BestSignal",
                ]
            ]
        )
        print("\nResolved seed info:")
        print(meta)
