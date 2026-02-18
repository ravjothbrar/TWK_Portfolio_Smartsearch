import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
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
# In-browser AI comparison (WebLLM via WebGPU)
# ---------------------------------------------------------------------------

WEBLLM_MODEL = "Qwen3-0.6B-q4f16_1-MLC"


def _js_escape(s: str) -> str:
    """Escape a Python string for safe embedding inside a JS template literal."""
    return (
        s.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
    )


def build_comparison_component(query: str, project: dict) -> str:
    """Return a self-contained HTML page that loads WebLLM in a Web Worker,
    runs Qwen3-0.6B entirely in-browser via WebGPU, and streams the comparison."""

    q = _js_escape(query)
    title = _js_escape(project["title"])
    sector = _js_escape(project["sector"])
    desc = _js_escape(project["description"])

    return f"""<!DOCTYPE html>
<html><head><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:sans-serif;background:#161616;color:#e0e0e0;padding:16px 20px}}
#status{{color:#aaa;font-size:.85rem;margin-bottom:8px}}
#bar-wrap{{background:#333;border-radius:6px;height:6px;width:100%;margin-bottom:14px;display:none}}
#bar-fill{{height:6px;border-radius:6px;background:linear-gradient(90deg,#BB86FC,#03DAC6);width:0%;transition:width .3s}}
#output{{font-size:.9rem;line-height:1.7;color:#ccc;white-space:pre-wrap}}
</style></head><body>
<div id="status">Checking WebGPU…</div>
<div id="bar-wrap"><div id="bar-fill"></div></div>
<div id="output"></div>
<script type="module">
import {{ CreateWebWorkerMLCEngine }} from "https://esm.run/@mlc-ai/web-llm";

const $ = id => document.getElementById(id);

/* ---- guard ---- */
if (!navigator.gpu) {{
  $("status").textContent = "WebGPU is not supported in this browser. Please use Chrome or Edge.";
  throw new Error("no WebGPU");
}}

/* ---- 3-line Web Worker (identical to chat.webllm.ai) ---- */
const workerBlob = new Blob([`
import {{ WebWorkerMLCEngineHandler }} from "https://esm.run/@mlc-ai/web-llm";
const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg) => {{ handler.onmessage(msg); }};
`], {{ type: "text/javascript" }});

const worker = new Worker(URL.createObjectURL(workerBlob), {{ type: "module" }});

/* ---- load engine (weights cached in browser Cache Storage) ---- */
$("status").textContent = "Loading Qwen3-0.6B — first visit downloads ~400 MB, then cached…";
$("bar-wrap").style.display = "block";

const engine = await CreateWebWorkerMLCEngine(worker, "{WEBLLM_MODEL}", {{
  initProgressCallback: (p) => {{
    $("status").textContent = p.text;
    if (p.progress != null) $("bar-fill").style.width = (p.progress * 100) + "%";
  }}
}});

$("bar-wrap").style.display = "none";
$("status").textContent = "Generating comparison…";

/* ---- inference ---- */
const completion = await engine.chat.completions.create({{
  messages: [
    {{ role: "system", content: "You are a concise web-design agency assistant. Compare a client brief with a portfolio project. Explain why the project is relevant — overlapping themes like industry, audience, features, design approach, tech requirements — and note meaningful differences. 3-5 short paragraphs, plain text, no markdown. /no_think" }},
    {{ role: "user", content: `CLIENT BRIEF:\\n`+`{q}`+`\\n\\nPORTFOLIO PROJECT:\\nTitle: {title}\\nSector: {sector}\\nDescription: {desc}` }}
  ],
  stream: true,
  temperature: 0.7,
  max_tokens: 512
}});

$("status").textContent = "";
const out = $("output");
for await (const chunk of completion) {{
  out.textContent += chunk.choices[0]?.delta?.content || "";
}}
</script></body></html>"""


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

def check_password() -> bool:
    """Show a password prompt and return True once the correct password is entered."""
    if st.session_state.get("authenticated"):
        return True
    pwd = st.text_input("Password", type="password", placeholder="Enter password")
    if pwd and pwd == "ravjoth":
        st.session_state.authenticated = True
        st.rerun()
    elif pwd:
        st.error("Incorrect password.")
    return False


def main() -> None:
    st.set_page_config(
        page_title="TWK Smart Portfolio Search",
        page_icon="🔍",
        layout="centered",
    )

    if not check_password():
        st.stop()

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

    # --- Session state ---
    if "num_results" not in st.session_state:
        st.session_state.num_results = 5
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_sector" not in st.session_state:
        st.session_state.last_sector = ""
    if "active_comparison" not in st.session_state:
        st.session_state.active_comparison = None

    # Reset when search inputs change
    if query != st.session_state.last_query or selected_sector != st.session_state.last_sector:
        st.session_state.num_results = 5
        st.session_state.active_comparison = None
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

            pid = project["id"]
            if st.session_state.active_comparison == pid:
                # Render the WebLLM component inline under this card
                html = build_comparison_component(query, project)
                components.html(html, height=420, scrolling=True)
            else:
                if st.button("Why this match?", key=f"compare_{pid}"):
                    st.session_state.active_comparison = pid
                    st.rerun()

        # "See More" button
        if st.session_state.num_results < len(results):
            if st.button("See More"):
                st.session_state.num_results += 5
                st.rerun()
    else:
        st.info("Enter a client brief above to search the portfolio.")


if __name__ == "__main__":
    main()
