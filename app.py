import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_portfolio(path: str = "TWK_final_portfolio.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model & embeddings (cached so they survive re-runs)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_data
def build_embeddings(texts: tuple[str, ...]) -> np.ndarray:
    """Encode all project texts once and cache the result."""
    model = load_model()
    return model.encode(list(texts), show_progress_bar=False)


# ---------------------------------------------------------------------------
# Search / scoring
# ---------------------------------------------------------------------------

def search(
    query: str,
    selected_sector: str,
    projects: list[dict],
    embeddings: np.ndarray,
) -> list[dict]:
    model = load_model()
    query_emb = model.encode([query], show_progress_bar=False)

    similarities = cosine_similarity(query_emb, embeddings)[0]

    results = []
    for idx, project in enumerate(projects):
        sim = float(similarities[idx])

        if selected_sector == "All Sectors":
            score = sim
        else:
            if project["sector"] == selected_sector:
                score = (sim * 0.8) + 0.2
            else:
                score = sim * 0.8

        results.append({**project, "score": score})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------

CARD_CSS = """
<style>
.result-card {
    background-color: #1E1E1E;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 16px;
    color: #E0E0E0;
    font-family: sans-serif;
}
.result-card h3 {
    margin: 0 0 8px 0;
    color: #FFFFFF;
    font-size: 1.25rem;
}
.sector-badge {
    display: inline-block;
    background-color: #333;
    color: #BB86FC;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin-bottom: 12px;
}
.score-bar-container {
    background-color: #333;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    margin-top: 4px;
    margin-bottom: 16px;
}
.score-bar {
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, #BB86FC, #03DAC6);
}
.score-label {
    font-size: 0.85rem;
    color: #AAAAAA;
    margin-bottom: 2px;
}
.desc-text {
    font-size: 0.92rem;
    line-height: 1.6;
    color: #CCCCCC;
    margin-bottom: 16px;
}
.view-link {
    color: #03DAC6;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.9rem;
}
.view-link:hover {
    text-decoration: underline;
}
</style>
"""


def render_card(project: dict) -> str:
    pct = max(0.0, min(project["score"] * 100, 100.0))
    url = f"https://www.thewebkitchen.co.uk/web-design/{project['id']}/"
    return f"""
    <div class="result-card">
        <h3>{project["title"]}</h3>
        <span class="sector-badge">{project["sector"]}</span>
        <div class="score-label">Match Score: {pct:.1f}%</div>
        <div class="score-bar-container">
            <div class="score-bar" style="width:{pct:.1f}%"></div>
        </div>
        <div class="desc-text">{project["description"]}</div>
        <a class="view-link" href="{url}" target="_blank">View Project &rarr;</a>
    </div>
    """


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="TWK Smart Portfolio Search",
        page_icon="🔍",
        layout="centered",
    )

    # Inject card styles once
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    # --- Header ---
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
            <h1 style="color:#FFFFFF; margin-bottom:4px;">Smart Portfolio Search</h1>
            <p style="color:#AAAAAA; font-size:1rem;">
                Find the most relevant projects from
                <strong style="color:#BB86FC;">The Web Kitchen</strong> portfolio
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Load data & embeddings ---
    projects = load_portfolio()
    texts = tuple(
        f"{p['title']} | {p['sector']} | {p['description']}" for p in projects
    )
    embeddings = build_embeddings(texts)

    # --- Sector dropdown ---
    sectors = sorted({p["sector"] for p in projects})
    sector_options = ["All Sectors"] + sectors

    # --- Input controls ---
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Client Brief",
            placeholder="e.g. We need a modern website for a boarding school with virtual tours...",
        )
    with col2:
        selected_sector = st.selectbox("Sector", sector_options)

    # --- Session state for "See More" ---
    if "num_results" not in st.session_state:
        st.session_state.num_results = 5
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_sector" not in st.session_state:
        st.session_state.last_sector = ""

    # Reset counter when the search inputs change
    if query != st.session_state.last_query or selected_sector != st.session_state.last_sector:
        st.session_state.num_results = 5
        st.session_state.last_query = query
        st.session_state.last_sector = selected_sector

    # --- Run search ---
    if query.strip():
        results = search(query, selected_sector, projects, embeddings)
        visible = results[: st.session_state.num_results]

        st.markdown(
            f"<p style='color:#AAAAAA; margin-top:1rem;'>Showing <strong style='color:#FFFFFF;'>"
            f"{len(visible)}</strong> of <strong style='color:#FFFFFF;'>{len(results)}</strong> results</p>",
            unsafe_allow_html=True,
        )

        for project in visible:
            st.markdown(render_card(project), unsafe_allow_html=True)

        # "See More" button
        if st.session_state.num_results < len(results):
            if st.button("See More"):
                st.session_state.num_results += 5
                st.rerun()
    else:
        st.info("Enter a client brief above to search the portfolio.")


if __name__ == "__main__":
    main()
