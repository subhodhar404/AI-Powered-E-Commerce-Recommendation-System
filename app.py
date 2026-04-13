from flask import jsonify, Flask, render_template, request

from engine import (
    build_app_state,
    dataframe_to_products,
    ensure_semantic_model,
    get_catalog,
    hybrid_recommendation,
)


app = Flask(__name__, template_folder="templates", static_folder="mechine_learning")

DEFAULT_SEED_ITEM = "Garnier Whole Blends Repairing Shampoo Honey Treasures, For Damaged Hair, 22 fl. oz."
DEFAULT_COLLABORATIVE_USER_ID = "0"


def get_state():
    return build_app_state()


def get_default_item(state):
    catalog = state["catalog"]
    if catalog.empty:
        return ""
    names = set(catalog["Name"].tolist())
    if DEFAULT_SEED_ITEM in names:
        return DEFAULT_SEED_ITEM
    return str(catalog["Name"].iloc[0])


def get_default_user_id(state):
    collaborative_model = state.get("collaborative_model") or {}
    user_item_matrix = collaborative_model.get("user_item_matrix")
    if user_item_matrix is not None and DEFAULT_COLLABORATIVE_USER_ID in [str(user_id) for user_id in user_item_matrix.index.tolist()]:
        return DEFAULT_COLLABORATIVE_USER_ID
    return str(collaborative_model.get("default_user_id", "") or "")


def get_recommendation_context(state, selected_item, target_user_id, top_n):
    recommendations = []
    recommendation_message = ""
    resolved_seed_item = ""
    resolved_target_user = ""
    semantic_backend = state.get("semantic_backend", "lazy-load")
    semantic_model_name = state.get("semantic_model_name", "")

    if state["data"].empty or not selected_item:
        return {
            "recommendations": recommendations,
            "recommendation_message": recommendation_message,
            "resolved_seed_item": resolved_seed_item,
            "resolved_target_user": resolved_target_user,
            "semantic_backend": semantic_backend,
            "semantic_model_name": semantic_model_name,
        }

    semantic_model = ensure_semantic_model(state)
    semantic_backend = state.get("semantic_backend", semantic_backend)
    semantic_model_name = state.get("semantic_model_name", semantic_model_name)

    recommendation_df, meta = hybrid_recommendation(
        state["data"],
        selected_item,
        target_user_id=target_user_id,
        top_n=top_n,
        content_model=state.get("content_model"),
        semantic_model=semantic_model,
        collaborative_model=state.get("collaborative_model"),
    )
    recommendations = dataframe_to_products(recommendation_df)
    resolved_seed_item = meta.get("ResolvedSeedItem", "")
    resolved_target_user = meta.get("ResolvedTargetUser", "")

    if recommendations:
        recommendation_message = (
            f"Showing {len(recommendations)} hybrid recommendations for "
            f"user {resolved_target_user or target_user_id} and seed "
            f"'{resolved_seed_item or selected_item}'."
        )
    else:
        recommendation_message = (
            "No recommendations found. Try another product name or a different user id from the collaborative catalog."
        )

    return {
        "recommendations": recommendations,
        "recommendation_message": recommendation_message,
        "resolved_seed_item": resolved_seed_item,
        "resolved_target_user": resolved_target_user,
        "semantic_backend": semantic_backend,
        "semantic_model_name": semantic_model_name,
    }


@app.route("/")
@app.route("/index.html")
def home():
    state = get_state()
    sample_products = dataframe_to_products(state["catalog"].head(6))
    default_item = get_default_item(state)
    default_user_id = get_default_user_id(state)
    return render_template(
        "index.html",
        summary=state["summary"],
        metrics=state["metrics"],
        dataset_path=state["dataset_path"],
        sample_products=sample_products,
        default_item=default_item,
        default_user_id=default_user_id,
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
        default_user_id=get_default_user_id(state),
        data_ready=not state["data"].empty,
    )


@app.route("/recommandation")
@app.route("/recommandation.html")
def recommendation_page():
    state = get_state()
    default_item = get_default_item(state)
    default_user_id = get_default_user_id(state)
    selected_item = request.args.get("item_name", default_item).strip()
    target_user_id = request.args.get("target_user_id", default_user_id).strip()
    top_n = min(max(request.args.get("top_n", default=6, type=int), 1), 20)

    context = get_recommendation_context(state, selected_item, target_user_id, top_n)
    return render_template(
        "recommandation.html",
        metrics=state["metrics"],
        summary=state["summary"],
        default_item=default_item,
        default_user_id=default_user_id,
        selected_item=selected_item,
        target_user_id=target_user_id,
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
    default_user_id = get_default_user_id(state)
    selected_item = request.args.get("item_name", default_item).strip()
    target_user_id = request.args.get("target_user_id", default_user_id).strip()
    top_n = min(max(request.args.get("top_n", default=6, type=int), 1), 20)

    context = get_recommendation_context(state, selected_item, target_user_id, top_n)
    return jsonify(
        {
            "products": context["recommendations"],
            "selected_item": selected_item,
            "target_user_id": target_user_id,
            "resolved_seed_item": context["resolved_seed_item"],
            "resolved_target_user": context["resolved_target_user"],
            "top_n": top_n,
            "metrics": state["metrics"],
            "message": context["recommendation_message"],
            "semantic_backend": context["semantic_backend"],
            "semantic_model_name": context["semantic_model_name"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
