from flask import jsonify, Flask, render_template, request

from engine import (
    build_app_state,
    dataframe_to_products,
    ensure_semantic_model,
    get_catalog,
    hybrid_recommendation,
    parse_profile_inputs,
)


app = Flask(__name__, template_folder="templates", static_folder="mechine_learning")


def get_state():
    return build_app_state()


def get_default_item(state):
    catalog = state["catalog"]
    if catalog.empty:
        return ""
    return str(catalog["Name"].iloc[0])


def get_recommendation_context(state, selected_item, liked_items_text, top_n):
    recommendations = []
    recommendation_message = ""
    resolved_seed_item = ""
    resolved_profile_items = []
    semantic_backend = state.get("semantic_backend", "lazy-load")
    semantic_model_name = state.get("semantic_model_name", "")
    liked_item_names = parse_profile_inputs(liked_items_text)

    if not selected_item and liked_item_names:
        selected_item = liked_item_names[0]

    if state["data"].empty or not selected_item:
        return {
            "recommendations": recommendations,
            "recommendation_message": recommendation_message,
            "resolved_seed_item": resolved_seed_item,
            "resolved_profile_items": resolved_profile_items,
            "semantic_backend": semantic_backend,
            "semantic_model_name": semantic_model_name,
        }

    semantic_model = ensure_semantic_model(state)
    semantic_backend = state.get("semantic_backend", semantic_backend)
    semantic_model_name = state.get("semantic_model_name", semantic_model_name)

    recommendation_df, meta = hybrid_recommendation(
        state["data"],
        selected_item,
        liked_item_names=liked_item_names,
        top_n=top_n,
        content_model=state.get("content_model"),
        semantic_model=semantic_model,
    )
    recommendations = dataframe_to_products(recommendation_df)
    resolved_seed_item = meta.get("ResolvedSeedItem", "")
    resolved_profile_items = meta.get("ResolvedProfileItems", [])

    if recommendations:
        recommendation_message = (
            f"Showing {len(recommendations)} final hybrid recommendations for "
            f"'{resolved_seed_item or selected_item}'."
        )
    else:
        recommendation_message = (
            "No recommendations found. Try a clearer product name from the Browse Product page."
        )

    return {
        "recommendations": recommendations,
        "recommendation_message": recommendation_message,
        "resolved_seed_item": resolved_seed_item,
        "resolved_profile_items": resolved_profile_items,
        "semantic_backend": semantic_backend,
        "semantic_model_name": semantic_model_name,
    }


@app.route("/")
@app.route("/index.html")
def home():
    state = get_state()
    sample_products = dataframe_to_products(state["catalog"].head(6))
    default_item = get_default_item(state)
    return render_template(
        "index.html",
        summary=state["summary"],
        metrics=state["metrics"],
        dataset_path=state["dataset_path"],
        sample_products=sample_products,
        default_item=default_item,
        semantic_backend=state.get("semantic_backend", "lazy-load"),
        semantic_model_name=state.get("semantic_model_name", ""),
        data_ready=not state["data"].empty,
    )


@app.route("/products")
@app.route("/products.html")
def products_page():
    state = get_state()
    query = request.args.get("q", "").strip()
    products = dataframe_to_products(get_catalog(state["data"], limit=24, query=query))
    return render_template(
        "products.html",
        products=products,
        query=query,
        summary=state["summary"],
        default_item=get_default_item(state),
        data_ready=not state["data"].empty,
    )


@app.route("/recommandation")
@app.route("/recommandation.html")
def recommendation_page():
    state = get_state()
    default_item = get_default_item(state)
    selected_item = request.args.get("item_name", default_item).strip()
    liked_items_text = request.args.get("liked_items", "").strip()
    top_n = min(max(request.args.get("top_n", default=6, type=int), 1), 20)

    context = get_recommendation_context(state, selected_item, liked_items_text, top_n)
    return render_template(
        "recommandation.html",
        metrics=state["metrics"],
        summary=state["summary"],
        default_item=default_item,
        selected_item=selected_item,
        liked_items_text=liked_items_text,
        top_n=top_n,
        data_ready=not state["data"].empty,
        **context,
    )


@app.route("/about")
@app.route("/about.html")
def about_page():
    state = get_state()
    return render_template(
        "about.html",
        summary=state["summary"],
        metrics=state["metrics"],
        semantic_backend=state.get("semantic_backend", "lazy-load"),
        semantic_model_name=state.get("semantic_model_name", ""),
        data_ready=not state["data"].empty,
    )


@app.route("/authentication")
@app.route("/authentication.html")
def authentication_page():
    return render_template("authentication.html")


@app.route("/api/health")
def api_health():
    state = get_state()
    return jsonify(
        {
            "status": "ok",
            "data_ready": not state["data"].empty,
            "dataset_path": state["dataset_path"],
            "semantic_backend": state.get("semantic_backend", "lazy-load"),
            "data_limitation": state["summary"].get("data_limitation", ""),
        }
    )


@app.route("/api/products")
def api_products():
    state = get_state()
    query = request.args.get("q", "").strip()
    limit = min(max(request.args.get("limit", default=24, type=int), 1), 60)
    products = dataframe_to_products(get_catalog(state["data"], limit=limit, query=query))
    return jsonify(
        {
            "products": products,
            "count": len(products),
        }
    )


@app.route("/api/recommendations")
def api_recommendations():
    state = get_state()
    if state["data"].empty:
        return jsonify(
            {
                "products": [],
                "message": "Dataset not found. Keep dataset.tsv in the project root.",
            }
        )

    default_item = get_default_item(state)
    selected_item = request.args.get("item_name", default_item).strip()
    liked_items_text = request.args.get("liked_items", "").strip()
    top_n = min(max(request.args.get("top_n", default=6, type=int), 1), 20)

    context = get_recommendation_context(state, selected_item, liked_items_text, top_n)
    return jsonify(
        {
            "products": context["recommendations"],
            "selected_item": selected_item,
            "resolved_seed_item": context["resolved_seed_item"],
            "resolved_profile_items": context["resolved_profile_items"],
            "top_n": top_n,
            "metrics": state["metrics"],
            "message": context["recommendation_message"],
            "semantic_backend": context["semantic_backend"],
            "semantic_model_name": context["semantic_model_name"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
