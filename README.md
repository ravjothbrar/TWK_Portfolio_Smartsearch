# TWK Smart Portfolio Search

A semantic search tool for [The Web Kitchen](https://www.thewebkitchen.co.uk/) portfolio. Paste a client brief, get the most relevant past projects ranked by similarity.

**Live app:** [twkportfoliosmartsearch.streamlit.app](https://twkportfoliosmartsearch.streamlit.app/)

---

## What It Does

The tool lets you paste any client email or brief into a search box and instantly returns the most relevant projects from TWK's portfolio, ranked by a match score. Each result includes:

- A **match percentage** showing how closely the project relates to the brief
- A **10-line description** of the project highlighting unique implementations (e.g. 3D globes, interactive configurators)
- A **direct link** to the project on TWK's portfolio site
- An optional **sector filter** that boosts projects in the same industry

Latency is under 500ms after the initial load — all embeddings are cached in-browser on first visit.

---

## How It Was Built

### 1. Dataset compilation

The core challenge was extracting rich, queryable text from TWK's project history. The approach:

1. **Identified the best source documents** — Design Briefs and Research Reports from Google Drive, chosen for their context-rich content.
2. **Uploaded all documents to NotebookLM** (~150 documents, well within the 300 workspace limit).
3. **Filled gaps for older projects** — some lacked coherent documentation, so a supplementary report was compiled using web scrapers (text and visual) and AI agents to detail those projects in sufficient depth.
4. **Generated the dataset** — 6 carefully crafted prompts to NotebookLM produced the final `.json` file containing structured data for every project.

### 2. Application development

- **Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers` — chosen for its lightweight footprint and strong semantic performance.
- **Search logic:** Cosine similarity between the query embedding and pre-computed project embeddings, with an optional sector boost.
- **Framework:** Streamlit, with `@st.cache_resource` (model) and `@st.cache_data` (embeddings) to ensure everything is computed once and reused across interactions.
- **Deployment:** Streamlit Community Cloud (free tier).

---

## Project Structure

```
TWK_Portfolio_Smartsearch/
├── app.py                       # Streamlit application (single file)
├── requirements.txt             # Python dependencies
├── TWK_final_portfolio.json     # Portfolio dataset (88 projects)
└── README.md                    # This file
```

---

## Dataset Format

`TWK_final_portfolio.json` is an array of objects:

```json
{
  "id": "oxford-university-colleges",
  "title": "Oxford University Colleges",
  "sector": "Education",
  "description": "This project involved creating distinct digital identities for..."
}
```

| Field         | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `id`          | URL slug — used to build the link to `thewebkitchen.co.uk`        |
| `title`       | Project display name                                               |
| `sector`      | Industry category (Education, Corporate, Property, Finance, etc.)  |
| `description` | ~10-line summary focused on unique implementations and design work |

Current sectors: Corporate, Education, Finance, Media, Not-for-Profit, Property, Recruitment.

---

## How Search Works

1. On first load, the app encodes every project's `title | sector | description` into a 384-dimensional vector using `all-MiniLM-L6-v2`.
2. When a user submits a query, it gets encoded with the same model.
3. **Cosine similarity** is computed between the query vector and all project vectors.
4. **Scoring:**
   - If **"All Sectors"** is selected: `score = similarity`
   - If a **specific sector** is selected and the project **matches**: `score = (similarity * 0.8) + 0.2`
   - If a **specific sector** is selected and the project **does not match**: `score = similarity * 0.8`
5. Results are sorted by score (high to low) and displayed 5 at a time, with a "See More" button to load the next 5.

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/ravjothbrar/TWK_Portfolio_Smartsearch.git
cd TWK_Portfolio_Smartsearch

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`. The first run downloads the embedding model (~80 MB) and computes embeddings — subsequent runs are instant.

---

## Deployment (Streamlit Community Cloud)

The app is deployed for free on Streamlit Community Cloud.

### To redeploy or deploy a fork:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select the repository, branch (`main`), and main file (`app.py`).
4. Click **"Deploy"**.

The free tier provides ~1 GB RAM. The `all-MiniLM-L6-v2` model is ~80 MB, so it fits comfortably. First cold start after inactivity takes a few seconds; all subsequent interactions are fast.

---

## Maintaining the Dataset

When new projects are completed:

### Adding a new project

1. Upload the project's **Design Brief** and **Research Report** to the NotebookLM workspace.
2. Prompt NotebookLM to generate a new entry in the same JSON format (matching the `id`, `title`, `sector`, `description` structure).
3. Add the new object to `TWK_final_portfolio.json`.
4. Commit and push — the deployed app will automatically pick up the change.

### Updating an existing project

Edit the relevant object in `TWK_final_portfolio.json` directly. The embeddings are recomputed from the file contents on each cold start, so changes take effect immediately after the next app restart.

### Notes

- The `id` field must match the project's URL slug on `thewebkitchen.co.uk/web-design/{id}/`.
- Keep descriptions around 10 lines, focused on unique implementations, design decisions, and key functionality — this is what the embedding model uses for matching.
- If a project lacks documentation, the same web-scraping + AI-agent approach used for older projects can fill the gap.

---

## Dependencies

| Package               | Purpose                                |
| --------------------- | -------------------------------------- |
| `streamlit`           | Web application framework              |
| `sentence-transformers` | Embedding model (`all-MiniLM-L6-v2`) |
| `torch`               | Backend for sentence-transformers      |
| `numpy`               | Array operations                       |
| `scikit-learn`        | Cosine similarity computation          |
