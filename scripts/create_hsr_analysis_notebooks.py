from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


KERNEL = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


BOOTSTRAP = r"""
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from hsr_analysis.common import *

import numpy as np
import pandas as pd

ensure_project_structure(ROOT)
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 160)

print(f"Project root: {ROOT}")
"""


NOTEBOOKS: list[tuple[str, list]] = [
    (
        "00_project_setup_and_corpus_audit.ipynb",
        [
            md(
                """
                # 00 - Project Setup and Corpus Audit

                This notebook creates the reviewer-facing project structure, cleans the article table, writes the corpus audit, and exports the basic descriptive trend figures. It deliberately does not exclude reviews, editorials, bibliographies, anniversary items, or short-text records silently; it flags them so core and extended analyses can be compared later.
                """
            ),
            code(BOOTSTRAP),
            md("## Load and Clean the Corpus"),
            code(
                r"""
raw, source_path = load_article_source(ROOT)
print(f"Loaded {len(raw):,} rows from {source_path.relative_to(ROOT)}")

articles = build_articles_clean(raw)

exclusions_path = ROOT / "data/article_exclusions.csv"
if exclusions_path.exists():
    exclusions = pd.read_csv(exclusions_path)
    if not exclusions.empty and "article_id" in exclusions.columns:
        exclusions = exclusions.set_index("article_id")
        for col, target in [
            ("exclude_from_core", "corpus_inclusion_core"),
            ("exclude_from_extended", "corpus_inclusion_extended"),
        ]:
            if col in exclusions.columns:
                ids = exclusions.index[exclusions[col].fillna(False).astype(bool)]
                articles.loc[articles["article_id"].isin(ids), target] = False

write_csv(articles, ROOT / "outputs/tables/articles_clean.csv")
print(articles.shape)
display(articles.head(3))
"""
            ),
            md("## Corpus Audit"),
            code(
                r"""
def missing_count(series):
    return int(series.map(lambda x: compact_ws(x) == "").sum())

metrics = [
    ("n_records_total_raw", len(raw)),
    ("n_records_valid_year", len(articles)),
    ("n_unique_titles", articles["title"].map(compact_ws).str.lower().nunique()),
    ("n_duplicate_titles", int(articles["title"].map(compact_ws).str.lower().duplicated().sum())),
    ("n_missing_year_raw", int(pd.to_numeric(raw.get("year"), errors="coerce").isna().sum())),
    ("n_missing_authors", missing_count(articles["authors"])),
    ("n_missing_abstract_en", missing_count(articles["abstract_en"])),
    ("n_missing_abstract_de", missing_count(articles["abstract_de"])),
    ("n_missing_full_text", missing_count(articles["full_text"])),
    ("n_sufficient_text", int(articles["has_sufficient_text"].sum())),
    ("n_core_inclusion", int(articles["corpus_inclusion_core"].sum())),
    ("n_extended_inclusion", int(articles["corpus_inclusion_extended"].sum())),
    ("n_potential_reviews", int(articles["is_review"].sum())),
    ("n_potential_editorials_or_introductions", int(articles["is_editorial_or_intro"].sum())),
    ("n_potential_bibliographies", int(articles["is_bibliography"].sum())),
    ("n_potential_autobiographical_or_anniversary_items", int(articles["is_autobiographical_or_anniversary"].sum())),
]
audit = pd.DataFrame(metrics, columns=["metric", "value"])
write_csv(audit, ROOT / "outputs/tables/corpus_audit.csv")

by_year = articles.groupby("year").size().reset_index(name="n_articles")
by_language = articles.groupby("language", dropna=False).size().reset_index(name="n_articles")
by_journal = articles.groupby("journal", dropna=False).size().reset_index(name="n_articles")
by_type = articles.groupby("type_en", dropna=False).size().reset_index(name="n_articles")
by_period = articles.groupby("period", dropna=False).size().reset_index(name="n_articles")

write_csv(by_year, ROOT / "outputs/tables/corpus_by_year.csv")
write_csv(by_language, ROOT / "outputs/tables/corpus_by_language.csv")
write_csv(by_journal, ROOT / "outputs/tables/corpus_by_journal_or_supplement.csv")
write_csv(by_type, ROOT / "outputs/tables/corpus_by_type.csv")
write_csv(by_period, ROOT / "outputs/tables/corpus_by_period.csv")

audit_md = [
    "# HSR Corpus Audit",
    "",
    f"Source file: `{source_path.relative_to(ROOT)}`",
    "",
    "## Core Counts",
    "",
]
audit_md.extend(f"- `{row.metric}`: {row.value}" for row in audit.itertuples())
audit_md.extend(
    [
        "",
        "## Reviewer-Relevant Treatment",
        "",
        "- The modelling corpus is not filtered until each document has been assigned text, language, type, and uncertainty flags.",
        "- Reviews, editorials, introductions, bibliographies, supplements, and anniversary/autobiographical items are retained as flagged records.",
        "- Core and extended inclusion flags are written to `outputs/tables/articles_clean.csv` and can be toggled in later notebooks.",
        "- UMAP coordinates are used for visualization only; quantitative proximity is calculated in the embedding space or derived graph.",
    ]
)
write_markdown("\n".join(audit_md), ROOT / "outputs/diagnostics/corpus_audit.md")
display(audit)
"""
            ),
            md("## Basic Trends"),
            code(
                r"""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

articles["n_authors"] = articles["authors"].map(lambda x: len(split_author_string(x)))

basic_year = (
    articles.groupby("year")
    .agg(
        n_articles=("article_id", "nunique"),
        n_core_articles=("corpus_inclusion_core", "sum"),
        mean_authors=("n_authors", "mean"),
        share_english=("language", lambda x: (x == "en").mean()),
        share_german=("language", lambda x: (x == "de").mean()),
        share_reviews=("is_review", "mean"),
        share_editorials_or_intros=("is_editorial_or_intro", "mean"),
        share_bibliographies=("is_bibliography", "mean"),
    )
    .reset_index()
)
basic_period = (
    articles.groupby("period")
    .agg(
        n_articles=("article_id", "nunique"),
        n_core_articles=("corpus_inclusion_core", "sum"),
        mean_authors=("n_authors", "mean"),
        share_english=("language", lambda x: (x == "en").mean()),
        share_german=("language", lambda x: (x == "de").mean()),
    )
    .reset_index()
)
write_csv(basic_year, ROOT / "outputs/tables/basic_trends_by_year.csv")
write_csv(basic_period, ROOT / "outputs/tables/basic_trends_by_period.csv")

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(basic_year["year"], basic_year["n_articles"], color="#4c78a8")
ax.set_title("Articles per year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of articles")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_01_articles_per_year.png", dpi=220)
save_caption(ROOT, "fig_01_articles_per_year.png", "Annual article counts in the cleaned HSR corpus.")
plt.show()

lang_year = articles.pivot_table(index="year", columns="language", values="article_id", aggfunc="nunique", fill_value=0)
lang_share = lang_year.div(lang_year.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(11, 4))
lang_share[[c for c in ["en", "de", "fr"] if c in lang_share.columns]].plot.area(ax=ax, alpha=0.8)
ax.set_title("Language share over time")
ax.set_xlabel("Year")
ax.set_ylabel("Share of articles")
ax.legend(title="Language", loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_02_language_share_over_time.png", dpi=220)
save_caption(ROOT, "fig_02_language_share_over_time.png", "Share of HSR articles by publication language over time.")
plt.show()

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(basic_year["year"], basic_year["mean_authors"], color="#f58518", marker="o", markersize=3)
ax.set_title("Authorship size over time")
ax.set_xlabel("Year")
ax.set_ylabel("Mean number of authors")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_03_authorship_size_over_time.png", dpi=220)
save_caption(ROOT, "fig_03_authorship_size_over_time.png", "Mean number of listed authors per article by year.")
plt.show()

doc_flags = articles.melt(
    id_vars=["year", "article_id"],
    value_vars=["is_review", "is_editorial_or_intro", "is_bibliography", "is_autobiographical_or_anniversary"],
    var_name="document_flag",
    value_name="flagged",
)
doc_year = doc_flags.groupby(["year", "document_flag"])["flagged"].mean().reset_index()
fig, ax = plt.subplots(figsize=(11, 4))
sns.lineplot(data=doc_year, x="year", y="flagged", hue="document_flag", ax=ax)
ax.set_title("Flagged document types over time")
ax.set_xlabel("Year")
ax.set_ylabel("Share of articles")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_04_document_type_over_time.png", dpi=220)
save_caption(ROOT, "fig_04_document_type_over_time.png", "Share of records flagged as reviews, editorials/introductions, bibliographies, or anniversary/autobiographical items.")
plt.show()
"""
            ),
        ],
    ),
    (
        "01_people_names_and_roles.ipynb",
        [
            md(
                """
                # 01 - People, Names, and Roles

                This notebook creates the article-person table, name-disambiguation diagnostics, and editor-role templates. The outputs are designed to make uncertainty visible before any actor-level interpretation is attempted.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
articles_path = ROOT / "outputs/tables/articles_clean.csv"
if not articles_path.exists():
    raw, _ = load_article_source(ROOT)
    articles = build_articles_clean(raw)
    write_csv(articles, articles_path)
else:
    articles = pd.read_csv(articles_path)

corrections = load_name_corrections(ROOT)
article_persons = build_article_persons(articles, corrections)
write_csv(article_persons, ROOT / "outputs/tables/article_persons.csv")

variants, possible_duplicates, top_manual = name_diagnostics(article_persons)
write_csv(variants, ROOT / "outputs/tables/name_variants_detected.csv")
write_csv(possible_duplicates, ROOT / "outputs/tables/possible_duplicate_persons.csv")
write_csv(top_manual, ROOT / "outputs/tables/top_persons_manual_check.csv")

persons = (
    article_persons.groupby(["person_id", "person_name"])
    .agg(
        n_articles=("article_id", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_raw_name_variants=("person_raw", "nunique"),
        raw_name_variants=("person_raw", lambda x: " | ".join(sorted(set(map(compact_ws, x))))),
    )
    .reset_index()
)
persons["active_span"] = persons["last_year"] - persons["first_year"] + 1

roles_path = ROOT / "data/editors_roles.csv"
roles = pd.read_csv(roles_path)
if roles.empty:
    for col in ["is_editor_any", "is_managing_editor", "is_special_issue_editor", "is_board_member"]:
        persons[col] = False
    persons["editor_role_count"] = 0
    persons["editor_active_start"] = pd.NA
    persons["editor_active_end"] = pd.NA
    persons["editor_role_certainty"] = "unknown"
else:
    roles["person_name"] = roles["person_name"].map(compact_ws)
    role_summary = (
        roles.groupby("person_name")
        .agg(
            editor_role_count=("role_type", "count"),
            editor_active_start=("start_year", "min"),
            editor_active_end=("end_year", "max"),
            editor_role_certainty=("certainty", lambda x: "; ".join(sorted(set(map(compact_ws, x))) or ["unknown"])),
            role_types=("role_type", lambda x: " | ".join(sorted(set(map(compact_ws, x))))),
        )
        .reset_index()
    )
    persons = persons.merge(role_summary, on="person_name", how="left")
    persons["editor_role_count"] = persons["editor_role_count"].fillna(0).astype(int)
    persons["is_editor_any"] = persons["editor_role_count"] > 0
    persons["is_managing_editor"] = persons.get("role_types", "").fillna("").str.contains("managing_editor", regex=False)
    persons["is_special_issue_editor"] = persons.get("role_types", "").fillna("").str.contains("special_issue_editor|issue_editor|guest_editor", regex=True)
    persons["is_board_member"] = persons.get("role_types", "").fillna("").str.contains("editorial_board|consulting_editor|cooperating_editor", regex=True)
    persons["editor_role_certainty"] = persons["editor_role_certainty"].fillna("unknown")

write_csv(persons, ROOT / "outputs/tables/persons_clean.csv")

diagnostic_md = [
    "# Name and Role Diagnostics",
    "",
    f"- Article-person rows: {len(article_persons):,}",
    f"- Distinct normalized persons: {persons['person_id'].nunique():,}",
    f"- Persons with multiple raw variants: {int((variants.get('n_name_variants', pd.Series(dtype=int)) > 1).sum()) if not variants.empty else 0:,}",
    f"- Possible duplicate name groups: {len(possible_duplicates):,}",
    f"- Editor-role rows supplied manually: {len(roles):,}",
    "",
    "The top-50 recurring persons are written to `outputs/tables/top_persons_manual_check.csv` for manual review.",
]
write_markdown("\n".join(diagnostic_md), ROOT / "outputs/diagnostics/name_and_role_diagnostics.md")

display(persons.sort_values("n_articles", ascending=False).head(15))
display(possible_duplicates.head(15))
"""
            ),
        ],
    ),
    (
        "02_embeddings_qwen_and_diagnostics.ipynb",
        [
            md(
                """
                # 02 - Multilingual Embeddings and Confound Diagnostics

                This notebook computes or loads article embeddings. The default model follows the revised plan and uses `Qwen/Qwen3-Embedding-0.6B`, with a multilingual sentence-transformer fallback if the Qwen model cannot be loaded locally. Embeddings are cached, and all downstream notebooks read the cached `.npy` files.
                """
            ),
            code(BOOTSTRAP),
            md("## Configuration"),
            code(
                r"""
MAIN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ALT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

TEXT_COLUMN = "text_for_embedding"
MAX_CHARS = 6000
BATCH_SIZE = 16
COMPUTE_ALT_EMBEDDINGS = False
ALLOW_FALLBACK_MODEL = True

articles = pd.read_csv(ROOT / "outputs/tables/articles_clean.csv")
embedding_articles = articles[articles["corpus_inclusion_core"].fillna(False)].copy()
embedding_articles = embedding_articles[embedding_articles[TEXT_COLUMN].map(lambda x: len(compact_ws(x)) >= 120)].copy()
embedding_articles = embedding_articles.sort_values(["year", "article_id"]).reset_index(drop=True)
print(f"Embedding records: {len(embedding_articles):,}")
"""
            ),
            md("## Compute or Load Embeddings"),
            code(
                r"""
from pathlib import Path

def load_sentence_transformer(model_name):
    from sentence_transformers import SentenceTransformer
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except Exception:
        device = "cpu"
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    except TypeError:
        model = SentenceTransformer(model_name, device=device)
    return model

def compute_or_load_embeddings(df, model_name, output_path, index_path, batch_size=BATCH_SIZE):
    output_path = Path(output_path)
    index_path = Path(index_path)
    if output_path.exists() and index_path.exists():
        matrix = np.load(output_path)
        index = pd.read_csv(index_path)
        cached_model = index["embedding_model"].iloc[0] if "embedding_model" in index.columns and len(index) else model_name
        print(f"Loaded cached embeddings: {output_path.relative_to(ROOT)} {matrix.shape}")
        return matrix, index, cached_model

    texts = df[TEXT_COLUMN].map(lambda x: compact_ws(x)[:MAX_CHARS]).tolist()
    model = load_sentence_transformer(model_name)
    matrix = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    matrix = np.asarray(matrix, dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)
    index = df[["article_id", "year", "language", "title", "text_length_chars", "text_length_words"]].copy()
    index["embedding_row"] = np.arange(len(index))
    index["embedding_model"] = model_name
    index["text_column"] = TEXT_COLUMN
    index["max_chars"] = MAX_CHARS
    write_csv(index, index_path)
    return matrix, index, model_name

main_path = ROOT / "outputs/models/article_embeddings_main.npy"
main_index_path = ROOT / "outputs/tables/article_embedding_index.csv"

try:
    embeddings, embedding_index, model_used = compute_or_load_embeddings(
        embedding_articles,
        MAIN_EMBEDDING_MODEL,
        main_path,
        main_index_path,
    )
except Exception as exc:
    if not ALLOW_FALLBACK_MODEL:
        raise
    print(f"Main model failed: {exc}")
    print(f"Falling back to {FALLBACK_EMBEDDING_MODEL}")
    embeddings, embedding_index, model_used = compute_or_load_embeddings(
        embedding_articles,
        FALLBACK_EMBEDDING_MODEL,
        main_path,
        main_index_path,
    )

runs_path = ROOT / "outputs/tables/embedding_model_runs.csv"
run_row = pd.DataFrame(
    [
        {
            "embedding_slot": "main",
            "requested_model": MAIN_EMBEDDING_MODEL,
            "model_used": model_used,
            "n_articles": embeddings.shape[0],
            "n_dimensions": embeddings.shape[1],
            "text_column": TEXT_COLUMN,
            "max_chars": MAX_CHARS,
        }
    ]
)
write_csv(run_row, runs_path)
print(embeddings.shape)
display(embedding_index.head())
"""
            ),
            md("## Optional Alternative Embedding Model"),
            code(
                r"""
if COMPUTE_ALT_EMBEDDINGS:
    alt_embeddings, alt_index, alt_model_used = compute_or_load_embeddings(
        embedding_articles,
        ALT_EMBEDDING_MODEL,
        ROOT / "outputs/models/article_embeddings_alt.npy",
        ROOT / "outputs/tables/article_embedding_index_alt.csv",
    )
    alt_row = pd.DataFrame(
        [
            {
                "embedding_slot": "alt",
                "requested_model": ALT_EMBEDDING_MODEL,
                "model_used": alt_model_used,
                "n_articles": alt_embeddings.shape[0],
                "n_dimensions": alt_embeddings.shape[1],
                "text_column": TEXT_COLUMN,
                "max_chars": MAX_CHARS,
            }
        ]
    )
    previous = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    write_csv(pd.concat([previous, alt_row], ignore_index=True), runs_path)
else:
    print("Alternative embeddings are configured but not computed. Set COMPUTE_ALT_EMBEDDINGS = True for the robustness run.")
"""
            ),
            md("## Language and Length Diagnostics in the Embedding Space"),
            code(
                r"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

try:
    import umap
except ImportError as exc:
    raise ImportError("Install umap-learn to create diagnostic maps.") from exc

sns.set_theme(style="white")
diagnostic_mapper = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.10, metric="cosine", random_state=42)
diag_xy = diagnostic_mapper.fit_transform(embeddings)
diag = embedding_index.copy()
diag["umap_x"] = diag_xy[:, 0]
diag["umap_y"] = diag_xy[:, 1]
write_csv(diag, ROOT / "outputs/tables/embedding_diagnostic_coordinates.csv")

language_summary = (
    diag.groupby("language", dropna=False)
    .agg(
        n_articles=("article_id", "nunique"),
        mean_text_length_words=("text_length_words", "mean"),
        median_text_length_words=("text_length_words", "median"),
    )
    .reset_index()
)
write_csv(language_summary, ROOT / "outputs/tables/language_embedding_input_summary.csv")

fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(data=diag, x="umap_x", y="umap_y", hue="language", s=18, alpha=0.75, ax=ax)
ax.set_title("Language in embedding projection")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_language_in_embedding_space.png", dpi=220)
save_caption(ROOT, "fig_language_in_embedding_space.png", "Exploratory UMAP projection colored by article language; coordinates are not interpreted as metric distances.")
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(diag["umap_x"], diag["umap_y"], c=np.log1p(diag["text_length_words"]), s=18, alpha=0.75, cmap="viridis")
ax.set_title("Text length in embedding projection")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.colorbar(scatter, ax=ax, label="log(1 + words)")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_text_length_in_embedding_space.png", dpi=220)
save_caption(ROOT, "fig_text_length_in_embedding_space.png", "Exploratory UMAP projection colored by input text length.")
plt.show()
display(language_summary)
"""
            ),
        ],
    ),
    (
        "03_semantic_mapping_clustering_stability.ipynb",
        [
            md(
                """
                # 03 - Semantic Mapping, Clustering, and Stability

                This notebook treats UMAP as an exploratory visualization and performs clustering in higher-dimensional embedding/PCA space or on a kNN graph. It writes all cluster assignments and stability diagnostics so only robust semantic regions are interpreted later.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

articles = pd.read_csv(ROOT / "outputs/tables/articles_clean.csv")
embedding_index = pd.read_csv(ROOT / "outputs/tables/article_embedding_index.csv")
embeddings = np.load(ROOT / "outputs/models/article_embeddings_main.npy")
work = embedding_index.merge(articles, on="article_id", how="left", suffixes=("", "_article")).sort_values("embedding_row")
print(work.shape, embeddings.shape)
"""
            ),
            md("## UMAP Runs"),
            code(
                r"""
import umap

UMAP_GRID = [
    {"n_neighbors": 10, "min_dist": 0.05, "metric": "cosine"},
    {"n_neighbors": 15, "min_dist": 0.10, "metric": "cosine"},
    {"n_neighbors": 30, "min_dist": 0.10, "metric": "cosine"},
    {"n_neighbors": 50, "min_dist": 0.20, "metric": "cosine"},
]
SEEDS = [1, 7, 42, 100, 2026]

umap_rows = []
for grid_id, params in enumerate(UMAP_GRID):
    for seed in SEEDS:
        run_id = f"umap_g{grid_id}_seed{seed}"
        out_path = ROOT / f"outputs/models/umap_runs/{run_id}.npy"
        if out_path.exists():
            xy = np.load(out_path)
        else:
            reducer = umap.UMAP(n_components=2, random_state=seed, **params)
            xy = reducer.fit_transform(embeddings)
            np.save(out_path, xy.astype(np.float32))
        umap_rows.append({"umap_run_id": run_id, "seed": seed, **params, "path": str(out_path.relative_to(ROOT))})

umap_index = pd.DataFrame(umap_rows)
write_csv(umap_index, ROOT / "outputs/tables/umap_run_index.csv")

main_umap = np.load(ROOT / "outputs/models/umap_runs/umap_g1_seed42.npy")
umap_main = work[["article_id", "year", "language", "title"]].copy()
umap_main["umap_x"] = main_umap[:, 0]
umap_main["umap_y"] = main_umap[:, 1]
write_csv(umap_main, ROOT / "outputs/tables/article_umap_main.csv")

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(umap_main["umap_x"], umap_main["umap_y"], c=umap_main["year"], cmap="viridis", s=16, alpha=0.75)
ax.set_title("HSR semantic map, colored by year")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.colorbar(scatter, ax=ax, label="Year")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_semantic_map_by_year.png", dpi=220)
save_caption(ROOT, "fig_semantic_map_by_year.png", "Exploratory UMAP map colored by publication year; local neighborhoods and stable regions are interpreted, not point distances.")
plt.show()
display(umap_index.head())
"""
            ),
            md("## Cluster Solutions"),
            code(
                r"""
def label_summary(labels):
    labels = np.asarray(labels)
    non_noise = labels[labels != -1]
    return {
        "n_clusters": int(len(set(non_noise))),
        "n_noise": int((labels == -1).sum()),
        "noise_share": float((labels == -1).mean()),
    }

n_components = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
pca = PCA(n_components=n_components, random_state=42)
pca_embeddings = pca.fit_transform(embeddings)
np.save(ROOT / "outputs/models/article_embeddings_pca50.npy", pca_embeddings.astype(np.float32))

solutions = {}
solution_meta = []

try:
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=18, min_samples=5, metric="euclidean", prediction_data=False)
    labels = clusterer.fit_predict(pca_embeddings)
    solutions["hdbscan_pca50"] = labels
    solution_meta.append({"cluster_solution_id": "hdbscan_pca50", "model_name": "HDBSCAN on PCA(50)", "clustering_params": "min_cluster_size=18; min_samples=5"})
except Exception as exc:
    print(f"External hdbscan package skipped: {exc}")
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        clusterer = SklearnHDBSCAN(min_cluster_size=18, min_samples=5, metric="euclidean")
        labels = clusterer.fit_predict(pca_embeddings)
        solutions["hdbscan_sklearn_pca50"] = labels
        solution_meta.append({"cluster_solution_id": "hdbscan_sklearn_pca50", "model_name": "scikit-learn HDBSCAN on PCA(50)", "clustering_params": "min_cluster_size=18; min_samples=5"})
    except Exception as exc2:
        print(f"scikit-learn HDBSCAN skipped: {exc2}")

nn = NearestNeighbors(n_neighbors=min(20, len(work) - 1), metric="cosine")
nn.fit(embeddings)
distances, neighbors = nn.kneighbors(embeddings)
G = nx.Graph()
G.add_nodes_from(range(len(work)))
for i, (row_dist, row_neighbors) in enumerate(zip(distances, neighbors)):
    for d, j in zip(row_dist[1:], row_neighbors[1:]):
        G.add_edge(i, int(j), weight=float(1 - d))
communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)
graph_labels = np.full(len(work), -1, dtype=int)
for community_id, nodes in enumerate(communities):
    for node in nodes:
        graph_labels[node] = community_id
solutions["knn_louvain"] = graph_labels
solution_meta.append({"cluster_solution_id": "knn_louvain", "model_name": "kNN graph Louvain", "clustering_params": "k=20; cosine"})

for n_clusters in [18, 24, 30]:
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(pca_embeddings)
    key = f"kmeans_pca50_k{n_clusters}"
    solutions[key] = labels
    solution_meta.append({"cluster_solution_id": key, "model_name": "KMeans on PCA(50)", "clustering_params": f"k={n_clusters}"})

meta_rows = []
for item in solution_meta:
    labels = solutions[item["cluster_solution_id"]]
    summary = label_summary(labels)
    try:
        sil = silhouette_score(pca_embeddings, labels, metric="euclidean") if summary["n_clusters"] > 1 else np.nan
    except Exception:
        sil = np.nan
    meta_rows.append({**item, **summary, "silhouette_highdim": sil, "embedding_model": embedding_index["embedding_model"].iloc[0]})

cluster_index = pd.DataFrame(meta_rows)

hdbscan_candidates = [key for key in ["hdbscan_pca50", "hdbscan_sklearn_pca50"] if key in solutions]
if hdbscan_candidates:
    candidate = hdbscan_candidates[0]
    candidate_row = cluster_index.loc[cluster_index["cluster_solution_id"].eq(candidate)].iloc[0]
    preferred = candidate if candidate_row["noise_share"] < 0.65 and candidate_row["n_clusters"] >= 5 else "knn_louvain"
else:
    preferred = "knn_louvain"
cluster_index["is_main_solution"] = cluster_index["cluster_solution_id"].eq(preferred)
write_csv(cluster_index, ROOT / "outputs/tables/cluster_solutions_index.csv")

assignments = work[["article_id", "year", "language", "title", "embedding_row"]].copy()
for key, labels in solutions.items():
    assignments[key] = labels
assignments["cluster_id_main"] = solutions[preferred]
assignments["cluster_solution_main"] = preferred
write_csv(assignments, ROOT / "outputs/tables/article_cluster_assignments_all_models.csv")

print(f"Main cluster solution: {preferred}")
display(cluster_index)
"""
            ),
            md("## Stability Atlas and Confound Checks"),
            code(
                r"""
main_labels = assignments["cluster_id_main"].to_numpy()
stability_rows = []
for key, labels in solutions.items():
    stability_rows.append(
        {
            "cluster_solution_id": key,
            "ari_to_main": adjusted_rand_score(main_labels, labels),
            "nmi_to_main": normalized_mutual_info_score(main_labels, labels),
            **label_summary(labels),
        }
    )
stability = pd.DataFrame(stability_rows).merge(cluster_index, on="cluster_solution_id", how="left", suffixes=("", "_index"))
write_csv(stability, ROOT / "outputs/tables/topic_stability_metrics.csv")

fig, ax = plt.subplots(figsize=(8, 4))
stability_plot = stability.set_index("cluster_solution_id")[["ari_to_main", "nmi_to_main", "noise_share"]]
sns.heatmap(stability_plot, annot=True, cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_title("Model stability atlas")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_model_stability_atlas.png", dpi=220)
save_caption(ROOT, "fig_model_stability_atlas.png", "Comparison of clustering solutions against the selected main solution using ARI, NMI, and noise share.")
plt.show()

cluster_diag = (
    assignments.groupby("cluster_id_main")
    .agg(
        n_articles=("article_id", "nunique"),
        top_language=("language", lambda x: x.value_counts(dropna=False).index[0]),
        top_language_share=("language", lambda x: x.value_counts(normalize=True, dropna=False).iloc[0]),
        mean_text_length_words=("embedding_row", lambda rows: float(work.loc[rows.index, "text_length_words"].mean())),
        median_text_length_words=("embedding_row", lambda rows: float(work.loc[rows.index, "text_length_words"].median())),
    )
    .reset_index()
)
write_csv(cluster_diag, ROOT / "outputs/tables/language_cluster_diagnostics.csv")
write_csv(cluster_diag[["cluster_id_main", "n_articles", "mean_text_length_words", "median_text_length_words"]], ROOT / "outputs/tables/text_length_cluster_diagnostics.csv")

plot_df = umap_main.merge(assignments[["article_id", "cluster_id_main"]], on="article_id", how="left")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=plot_df, x="umap_x", y="umap_y", hue="cluster_id_main", palette="tab20", s=16, alpha=0.8, legend=False, ax=ax)
ax.set_title("Main semantic regions")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_semantic_regions_main.png", dpi=220)
save_caption(ROOT, "fig_semantic_regions_main.png", "Main model-induced semantic regions overlaid on the exploratory UMAP map.")
plt.show()

display(stability)
display(cluster_diag.sort_values("top_language_share", ascending=False).head(15))
"""
            ),
        ],
    ),
    (
        "04_topic_labeling_and_time_dynamics.ipynb",
        [
            md(
                """
                # 04 - Human-Supervised Topic Labels and Time Dynamics

                This notebook labels embedding-based clusters as candidate topics, writes evidence packs for manual review, compares unguided clusters with seed-topic lenses without making the seeds determinative, and exports the main topic-trend figures.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

articles = pd.read_csv(ROOT / "outputs/tables/articles_clean.csv")
assignments = pd.read_csv(ROOT / "outputs/tables/article_cluster_assignments_all_models.csv")
embedding_index = pd.read_csv(ROOT / "outputs/tables/article_embedding_index.csv")
embeddings = np.load(ROOT / "outputs/models/article_embeddings_main.npy")

topics_base = assignments[["article_id", "cluster_id_main", "cluster_solution_main", "embedding_row"]].merge(
    articles, on="article_id", how="left"
)
topics_base["cluster_id_main"] = topics_base["cluster_id_main"].astype(int)
print(topics_base.shape)
"""
            ),
            md("## Machine Labels and Manual Override Template"),
            code(
                r"""
label_rows = []
for cluster_id, group in topics_base.groupby("cluster_id_main"):
    terms = top_tfidf_terms(group["text_for_labeling"], n_terms=10)
    machine_label = " / ".join(terms[:4]) if terms else f"candidate topic {cluster_id}"
    label_rows.append(
        {
            "cluster_id_main": cluster_id,
            "topic_id": f"topic_{cluster_id}",
            "topic_label_machine": machine_label,
            "top_terms": "; ".join(terms),
            "n_articles": group["article_id"].nunique(),
            "first_year": group["year"].min(),
            "last_year": group["year"].max(),
            "topic_macro_category_machine": infer_macro_category(machine_label + " " + " ".join(terms)),
        }
    )

labels = pd.DataFrame(label_rows).sort_values(["cluster_id_main"])
override_path = ROOT / "outputs/labels/topic_label_overrides.csv"
if not override_path.exists():
    override_template = labels[["cluster_id_main", "topic_id", "topic_label_machine", "top_terms", "n_articles"]].copy()
    override_template["topic_label_human"] = ""
    override_template["topic_macro_category_human"] = ""
    override_template["label_status"] = "needs_human_review"
    override_template["label_rationale"] = ""
    write_csv(override_template, override_path)

overrides = pd.read_csv(override_path)
labels = labels.merge(
    overrides[["cluster_id_main", "topic_label_human", "topic_macro_category_human", "label_status", "label_rationale"]],
    on="cluster_id_main",
    how="left",
)
labels["topic_label_human"] = labels["topic_label_human"].map(compact_ws)
labels["topic_label_human"] = np.where(labels["topic_label_human"] != "", labels["topic_label_human"], labels["topic_label_machine"])
labels["topic_macro_category"] = labels["topic_macro_category_human"].map(compact_ws)
labels["topic_macro_category"] = np.where(
    labels["topic_macro_category"] != "",
    labels["topic_macro_category"],
    labels["topic_macro_category_machine"],
)
labels["label_status"] = labels["label_status"].fillna("machine_suggested")
labels["is_residual_topic"] = labels["cluster_id_main"].eq(-1) | labels["topic_label_human"].str.contains("residual|mixed|other", case=False, regex=True)

topic_stability = pd.read_csv(ROOT / "outputs/tables/topic_stability_metrics.csv")
main_solution = assignments["cluster_solution_main"].iloc[0]
main_stability_score = float(topic_stability.loc[topic_stability["cluster_solution_id"].eq(main_solution), "nmi_to_main"].fillna(1).iloc[0])
labels["topic_stability_score"] = main_stability_score

article_topics = topics_base[["article_id", "cluster_id_main"]].merge(labels, on="cluster_id_main", how="left")
article_topics["topic_probability_or_strength"] = np.nan
write_csv(labels, ROOT / "outputs/tables/topic_labels.csv")
write_csv(article_topics, ROOT / "outputs/tables/article_topics.csv")
display(labels.sort_values("n_articles", ascending=False).head(20))
"""
            ),
            md("## Topic Evidence Packs"),
            code(
                r"""
for cluster_id, group in topics_base.groupby("cluster_id_main"):
    label_row = labels[labels["cluster_id_main"].eq(cluster_id)].iloc[0]
    rows = group.sort_values("year")
    matrix = embeddings[rows["embedding_row"].to_numpy()]
    if len(rows) > 1:
        centroid = matrix.mean(axis=0, keepdims=True)
        sims = cosine_similarity(matrix, centroid).ravel()
        rows = rows.assign(_centroid_similarity=sims)
        representative = rows.sort_values("_centroid_similarity", ascending=False).head(15)
        marginal = rows.sort_values("_centroid_similarity", ascending=True).head(10)
    else:
        representative = rows
        marginal = rows

    period_dist = rows["period"].value_counts().to_string()
    language_dist = rows["language"].value_counts(dropna=False).to_string()
    type_dist = rows["type_en"].value_counts(dropna=False).head(10).to_string()
    author_counts = []
    for authors in rows["authors"].dropna():
        author_counts.extend(split_author_string(authors))
    author_summary = pd.Series(author_counts).value_counts().head(10).to_string() if author_counts else "No author data"
    issue_summary = rows["issue_title"].map(compact_ws).replace("", pd.NA).dropna().value_counts().head(10).to_string()
    if not issue_summary:
        issue_summary = "No recurring issue titles"

    pack = [
        f"# Evidence Pack: topic_{cluster_id}",
        "",
        f"- Machine label: {label_row['topic_label_machine']}",
        f"- Human label: {label_row['topic_label_human']}",
        f"- Macro category: {label_row['topic_macro_category']}",
        f"- Label status: {label_row['label_status']}",
        f"- Stability score inherited from main solution: {label_row['topic_stability_score']:.3f}",
        f"- Number of articles: {len(rows)}",
        "",
        "## Top Terms",
        "",
        label_row["top_terms"],
        "",
        "## Representative Titles",
        "",
    ]
    pack.extend(f"- {int(r.year)}: {compact_ws(r.title)}" for r in representative.itertuples())
    pack.extend(["", "## Marginal Titles", ""])
    pack.extend(f"- {int(r.year)}: {compact_ws(r.title)}" for r in marginal.itertuples())
    pack.extend(["", "## Period Distribution", "", "```", period_dist, "```"])
    pack.extend(["", "## Language Distribution", "", "```", language_dist, "```"])
    pack.extend(["", "## Document Types", "", "```", type_dist, "```"])
    pack.extend(["", "## Main Authors", "", "```", author_summary, "```"])
    pack.extend(["", "## Main Issues", "", "```", issue_summary, "```"])
    pack.extend(
        [
            "",
            "## Manual Interpretation",
            "",
            "Human label rationale: fill in `outputs/labels/topic_label_overrides.csv` and rerun this notebook.",
        ]
    )
    write_markdown("\n".join(pack), ROOT / f"outputs/diagnostics/topic_evidence/topic_{cluster_id}.md")

print(f"Wrote {labels.shape[0]} topic evidence packs.")
"""
            ),
            md("## Guided Seed Comparison as a Check, Not a Topic Model"),
            code(
                r"""
seed_topics = {
    "methods_and_methodology": ["method", "methods", "methodology", "qualitative", "quantitative", "grounded theory", "sampling"],
    "time_series_longitudinal": ["time series", "longitudinal", "panel", "sequence", "measurement", "classification"],
    "data_infrastructure": ["data", "database", "data archive", "coding", "digital history", "computing", "information system"],
    "digital_big_data_gis": ["big data", "computer", "digital", "gis", "geographic information", "algorithm"],
    "politics_elites_state": ["election", "voting", "party", "political", "elite", "crisis", "state"],
    "migration_borders_mobility": ["migration", "border", "migrant", "mobility", "regime"],
    "demography_family_population": ["family", "fertility", "demography", "population", "household"],
    "health_care_mortality": ["death", "mortality", "care", "health", "pandemic"],
    "sport_body_culture": ["sport", "football", "body", "leisure"],
    "conventions_law_economy": ["convention", "classification", "law", "economics of convention"],
    "environment_energy_sustainability": ["environment", "climate", "sustainability", "energy"],
    "science_technology_sts": ["science", "expertise", "knowledge", "technology", "sts"],
    "war_violence_security": ["war", "violence", "security", "risk", "genocide"],
    "culture_media_visuality": ["media", "visual", "image", "communication", "culture"],
}

seed_scores = topics_base[["article_id", "cluster_id_main", "text_for_labeling"]].copy()
text_lower = seed_scores["text_for_labeling"].map(lambda x: compact_ws(x).lower())
for seed_name, keywords in seed_topics.items():
    seed_scores[seed_name] = text_lower.map(lambda text: sum(text.count(keyword.lower()) for keyword in keywords))
seed_cols = list(seed_topics)
seed_scores["top_seed_topic"] = seed_scores[seed_cols].idxmax(axis=1)
seed_scores["top_seed_score"] = seed_scores[seed_cols].max(axis=1)

comparison = (
    seed_scores.groupby(["cluster_id_main", "top_seed_topic"])
    .agg(n_articles=("article_id", "nunique"), mean_seed_score=("top_seed_score", "mean"))
    .reset_index()
    .sort_values(["cluster_id_main", "n_articles"], ascending=[True, False])
)
comparison = comparison.merge(labels[["cluster_id_main", "topic_id", "topic_label_human"]], on="cluster_id_main", how="left")
write_csv(comparison, ROOT / "outputs/tables/guided_vs_unguided_topic_comparison.csv")
display(comparison.groupby("cluster_id_main").head(3))
"""
            ),
            md("## Topic Trends, Sensitivity, and Innovation Metrics"),
            code(
                r"""
topic_articles = articles.merge(article_topics[["article_id", "topic_id", "topic_label_human", "topic_macro_category", "topic_stability_score"]], on="article_id", how="inner")

def topic_shares(df, group_col):
    counts = df.groupby([group_col, "topic_id", "topic_label_human", "topic_macro_category"], dropna=False).size().reset_index(name="n_articles")
    totals = df.groupby(group_col).size().rename("n_total").reset_index()
    out = counts.merge(totals, on=group_col)
    out["topic_share"] = out["n_articles"] / out["n_total"]
    return out

year_trends = topic_shares(topic_articles, "year")
year_trends = year_trends.sort_values(["topic_id", "year"])
year_trends["rolling_share_5y"] = year_trends.groupby("topic_id")["topic_share"].transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
period_trends = topic_shares(topic_articles, "period")

write_csv(year_trends, ROOT / "outputs/tables/topic_trends_by_year.csv")
write_csv(period_trends, ROOT / "outputs/tables/topic_trends_by_period.csv")

def summarize_condition(name, mask):
    subset = topic_articles[mask].copy()
    if subset.empty:
        return pd.DataFrame()
    out = topic_shares(subset, "period")
    out["condition"] = name
    return out

sensitivity = pd.concat(
    [
        summarize_condition("core", topic_articles["corpus_inclusion_core"].fillna(False)),
        summarize_condition("extended", topic_articles["corpus_inclusion_extended"].fillna(False)),
        summarize_condition("core_without_review_editorial_biblio", topic_articles["corpus_inclusion_core"].fillna(False) & ~topic_articles[["is_review", "is_editorial_or_intro", "is_bibliography"]].any(axis=1)),
    ],
    ignore_index=True,
)
write_csv(sensitivity, ROOT / "outputs/tables/topic_trend_sensitivity.csv")

macro_year = topic_articles.groupby(["year", "topic_macro_category"]).size().reset_index(name="n_articles")
macro_pivot = macro_year.pivot_table(index="year", columns="topic_macro_category", values="n_articles", fill_value=0)
fig, ax = plt.subplots(figsize=(12, 5))
macro_pivot.plot.area(ax=ax, alpha=0.85)
ax.set_title("Macro topic streams")
ax.set_xlabel("Year")
ax.set_ylabel("Number of articles")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Macro category")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_05_macro_topic_streams.png", dpi=220)
save_caption(ROOT, "fig_05_macro_topic_streams.png", "Changing distribution of broad thematic fields in HSR; categories are human-supervised labels on model-induced clusters.")
plt.show()

heatmap = period_trends.pivot_table(index="topic_label_human", columns="period", values="topic_share", fill_value=0)
fig, ax = plt.subplots(figsize=(12, max(6, 0.28 * len(heatmap))))
sns.heatmap(heatmap, cmap="mako", ax=ax)
ax.set_title("Topic prominence by period")
ax.set_xlabel("")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_06_topic_period_heatmap.png", dpi=220)
save_caption(ROOT, "fig_06_topic_period_heatmap.png", "Period heatmap of candidate topic shares; only stable and reviewed topics should be emphasized in the main text.")
plt.show()

top_topics = topic_articles["topic_label_human"].value_counts().head(8).index
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=year_trends[year_trends["topic_label_human"].isin(top_topics)], x="year", y="rolling_share_5y", hue="topic_label_human", ax=ax)
ax.set_title("Selected topic trajectories")
ax.set_xlabel("Year")
ax.set_ylabel("Five-year rolling share")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Topic")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_07_selected_topic_trajectories.png", dpi=220)
save_caption(ROOT, "fig_07_selected_topic_trajectories.png", "Five-year rolling shares for the most frequent candidate topics.")
plt.show()

innovation_rows = []
for topic_id, group in topic_articles.groupby("topic_id"):
    by_year = group.groupby("year").size()
    first_year = int(by_year.index.min())
    peak_year = int(by_year.idxmax())
    active_years = set(by_year.index.astype(int))
    possible_years = set(range(first_year, int(topic_articles["year"].max()) + 1))
    persistence = len(active_years & possible_years) / max(len(possible_years), 1)
    top_issues = group["issue_id"].value_counts().head(2).sum()
    issue_dependency = top_issues / len(group)
    innovation_rows.append(
        {
            "topic_id": topic_id,
            "topic_label_human": group["topic_label_human"].iloc[0],
            "first_year": first_year,
            "first_period": group.loc[group["year"].eq(first_year), "period"].iloc[0],
            "peak_year": peak_year,
            "peak_share": float(year_trends.loc[(year_trends["topic_id"].eq(topic_id)) & (year_trends["year"].eq(peak_year)), "topic_share"].max()),
            "n_years_active": len(active_years),
            "persistence_score": persistence,
            "issue_dependency_score": issue_dependency,
            "is_stable_core": persistence >= 0.25 and issue_dependency < 0.55 and group["period"].nunique() >= 3,
            "is_emerging_topic": first_year >= 2010 and persistence >= 0.10,
            "is_declining_topic": peak_year < 2005 and persistence < 0.35,
            "is_episode_topic": issue_dependency >= 0.55 and persistence < 0.25,
        }
    )
innovation = pd.DataFrame(innovation_rows).sort_values(["first_year", "topic_label_human"])
write_csv(innovation, ROOT / "outputs/tables/topic_innovation_metrics.csv")

fig, ax = plt.subplots(figsize=(12, max(5, 0.25 * len(innovation))))
plot_innov = innovation.sort_values("first_year")
ax.scatter(plot_innov["first_year"], plot_innov["topic_label_human"], s=(plot_innov["persistence_score"] * 350 + 20), c=plot_innov["issue_dependency_score"], cmap="viridis", alpha=0.8)
ax.set_title("Topic entry and persistence")
ax.set_xlabel("First year")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_08_topic_entry_and_persistence_timeline.png", dpi=220)
save_caption(ROOT, "fig_08_topic_entry_and_persistence_timeline.png", "First appearance of candidate topics; marker size indicates persistence and color indicates issue dependency.")
plt.show()

surprise = topics_base.merge(article_topics[["article_id", "topic_label_human"]], on="article_id", how="left")
surprise["expected_topic"] = ""
surprise["assigned_topic"] = surprise["topic_label_human"]
surprise["reason_for_surprise"] = ""
surprise["manual_comment"] = ""
write_csv(surprise[["article_id", "title", "year", "authors", "expected_topic", "assigned_topic", "reason_for_surprise", "manual_comment"]].head(100), ROOT / "outputs/tables/surprise_cases.csv")

try:
    import plotly.express as px

    umap_main = pd.read_csv(ROOT / "outputs/tables/article_umap_main.csv")
    interactive = umap_main.merge(
        topic_articles[
            [
                "article_id",
                "authors",
                "issue_title",
                "type_en",
                "topic_label_human",
                "topic_macro_category",
            ]
        ],
        on="article_id",
        how="left",
    )
    interactive["issue_title"] = interactive["issue_title"].map(compact_ws)
    fig = px.scatter(
        interactive,
        x="umap_x",
        y="umap_y",
        color="topic_label_human",
        hover_data={
            "title": True,
            "year": True,
            "authors": True,
            "issue_title": True,
            "language": True,
            "type_en": True,
            "topic_macro_category": True,
            "umap_x": False,
            "umap_y": False,
        },
        title="Interactive HSR Semantic Map",
        width=1100,
        height=800,
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.75})
    fig.write_html(ROOT / "outputs/figures/interactive_hsr_map.html", include_plotlyjs="cdn")
    print("Wrote outputs/figures/interactive_hsr_map.html")
except Exception as exc:
    print(f"Interactive map skipped: {exc}")
display(innovation.head(20))
"""
            ),
        ],
    ),
    (
        "05_issues_and_editorial_positioning.ipynb",
        [
            md(
                """
                # 05 - Issues and Editorial Positioning

                This notebook treats issues as their own analytical unit and adds editorial-role information where manually supplied. It phrases special issues as curatorial concentrations, not causal interventions.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

articles = pd.read_csv(ROOT / "outputs/tables/articles_clean.csv")
article_topics = pd.read_csv(ROOT / "outputs/tables/article_topics.csv")
article_persons = pd.read_csv(ROOT / "outputs/tables/article_persons.csv")
persons = pd.read_csv(ROOT / "outputs/tables/persons_clean.csv")
embedding_index = pd.read_csv(ROOT / "outputs/tables/article_embedding_index.csv")
embeddings = np.load(ROOT / "outputs/models/article_embeddings_main.npy")
roles = pd.read_csv(ROOT / "data/editors_roles.csv")

topic_articles = articles.merge(article_topics[["article_id", "topic_id", "topic_label_human", "topic_macro_category"]], on="article_id", how="left")
embedding_lookup = dict(zip(embedding_index["article_id"], embedding_index["embedding_row"]))
first_author_year = article_persons.groupby("person_id")["year"].min().to_dict()
editor_names = set(roles["person_name"].map(compact_ws)) if not roles.empty else set()
"""
            ),
            md("## Issue-Level Metrics"),
            code(
                r"""
issue_rows = []
for issue_id, group in topic_articles.groupby("issue_id"):
    topic_counts = group["topic_id"].value_counts(dropna=False)
    topic_probs = topic_counts / topic_counts.sum()
    hhi = float((topic_probs ** 2).sum())
    main_topic = topic_counts.index[0] if len(topic_counts) else pd.NA
    main_topic_row = group[group["topic_id"].eq(main_topic)].head(1)
    main_label = main_topic_row["topic_label_human"].iloc[0] if len(main_topic_row) else pd.NA
    main_macro = main_topic_row["topic_macro_category"].iloc[0] if len(main_topic_row) else pd.NA

    rows = [embedding_lookup.get(aid) for aid in group["article_id"] if aid in embedding_lookup]
    if len(rows) > 1:
        mat = embeddings[rows]
        centroid = mat.mean(axis=0, keepdims=True)
        coherence = float(cosine_similarity(mat, centroid).mean())
    else:
        coherence = np.nan

    issue_persons = article_persons[article_persons["article_id"].isin(group["article_id"])]
    issue_year = int(group["year"].median())
    unique_persons = issue_persons["person_id"].dropna().unique()
    if len(unique_persons):
        new_share = np.mean([first_author_year.get(pid, issue_year) >= issue_year for pid in unique_persons])
        recurring_share = np.mean([first_author_year.get(pid, issue_year) < issue_year for pid in unique_persons])
    else:
        new_share = np.nan
        recurring_share = np.nan

    editor_authored_articles = set(issue_persons.loc[issue_persons["person_name"].isin(editor_names), "article_id"])
    share_editor_authored = len(editor_authored_articles) / group["article_id"].nunique() if group["article_id"].nunique() else np.nan
    issue_title = compact_ws(group["issue_title"].dropna().iloc[0]) if group["issue_title"].notna().any() else ""
    enough_articles_for_concentration = group["article_id"].nunique() >= 4
    special_likely = bool(issue_title) or (enough_articles_for_concentration and hhi >= 0.45)
    basis = "issue_title" if issue_title else "topic_concentration" if enough_articles_for_concentration and hhi >= 0.45 else "none"

    issue_rows.append(
        {
            "issue_id": issue_id,
            "year": issue_year,
            "volume": group["volume"].dropna().iloc[0] if group["volume"].notna().any() else pd.NA,
            "issue": group["issue"].dropna().iloc[0] if group["issue"].notna().any() else pd.NA,
            "journal": group["journal"].dropna().iloc[0] if group["journal"].notna().any() else pd.NA,
            "issue_title": issue_title,
            "n_articles": group["article_id"].nunique(),
            "n_core_articles": int(group["corpus_inclusion_core"].sum()),
            "main_topic": main_label,
            "main_topic_id": main_topic,
            "main_macro_category": main_macro,
            "topic_concentration": hhi,
            "semantic_coherence": coherence,
            "share_editorial_or_intro": float(group["is_editorial_or_intro"].mean()),
            "share_new_authors": new_share,
            "share_recurring_authors": recurring_share,
            "share_editor_authored": share_editor_authored,
            "is_special_issue_likely": special_likely,
            "issue_detection_basis": basis,
            "manual_issue_status": "unclear",
        }
    )

issues = pd.DataFrame(issue_rows).sort_values(["year", "volume", "issue"])
corrections_path = ROOT / "data/issue_corrections.csv"
corrections = pd.read_csv(corrections_path)
if not corrections.empty:
    issues = issues.merge(corrections, on="issue_id", how="left", suffixes=("", "_manual"))
    issues["manual_issue_status"] = np.where(issues["manual_issue_status_manual"].map(compact_ws) != "", issues["manual_issue_status_manual"], issues["manual_issue_status"])
    issues["issue_title"] = np.where(issues["issue_title_corrected"].map(compact_ws) != "", issues["issue_title_corrected"], issues["issue_title"])
    issues = issues.drop(columns=[c for c in issues.columns if c.endswith("_manual") or c == "issue_title_corrected"], errors="ignore")

write_csv(issues, ROOT / "outputs/tables/issues_clean.csv")
display(issues.sort_values(["is_special_issue_likely", "topic_concentration"], ascending=False).head(20))
"""
            ),
            md("## Special Issue Pre/Post Patterns"),
            code(
                r"""
prepost_rows = []
for row in issues[issues["is_special_issue_likely"]].itertuples():
    if pd.isna(row.main_topic_id):
        continue
    pre = topic_articles[(topic_articles["year"].between(row.year - 5, row.year - 1))]
    issue_year = topic_articles[topic_articles["year"].eq(row.year)]
    post = topic_articles[(topic_articles["year"].between(row.year + 1, row.year + 5))]
    def share(df):
        return float(df["topic_id"].eq(row.main_topic_id).mean()) if len(df) else np.nan
    prepost_rows.append(
        {
            "issue_id": row.issue_id,
            "year": row.year,
            "issue_title": row.issue_title,
            "main_topic": row.main_topic,
            "topic_share_pre_5y": share(pre),
            "topic_share_issue_year": share(issue_year),
            "topic_share_post_5y": share(post),
            "persistence_after_issue": share(post),
            "interpretation_guardrail": "curatorial concentration, not causal proof",
        }
    )
prepost = pd.DataFrame(prepost_rows)
write_csv(prepost, ROOT / "outputs/tables/special_issue_topic_prepost.csv")

fig, ax = plt.subplots(figsize=(12, 5))
plot_issues = issues[issues["is_special_issue_likely"]].copy()
ax.scatter(plot_issues["year"], plot_issues["topic_concentration"], s=plot_issues["n_articles"] * 15, c=plot_issues["semantic_coherence"], cmap="viridis", alpha=0.75)
ax.set_title("Editorial interventions timeline")
ax.set_xlabel("Year")
ax.set_ylabel("Issue topic concentration")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_09_editorial_interventions_timeline.png", dpi=220)
save_caption(ROOT, "fig_09_editorial_interventions_timeline.png", "Likely special issues and topic concentration over time; the figure indicates curatorial concentration, not causal effects.")
plt.show()
display(prepost.head(20))
"""
            ),
            md("## Editor Topic Positions"),
            code(
                r"""
if roles.empty:
    editor_positions = pd.DataFrame(columns=["person_id", "person_name", "n_articles", "dominant_topic", "topic_entropy_norm", "editor_role_certainty"])
    editor_matched = pd.DataFrame(columns=["topic_id", "topic_label_human", "editor_share", "matched_noneditor_share"])
    editor_ranking = pd.DataFrame()
    write_csv(editor_positions, ROOT / "outputs/tables/editor_topic_positions.csv")
    write_csv(editor_matched, ROOT / "outputs/tables/editor_vs_matched_noneditor_topic_distribution.csv")
    write_csv(editor_ranking, ROOT / "outputs/tables/editor_brokerage_ranking.csv")
    print("No editor roles supplied yet. Fill data/editors_roles.csv and rerun this notebook.")
else:
    role_names = set(roles["person_name"].map(compact_ws))
    editor_persons = article_persons[article_persons["person_name"].isin(role_names)]
    editor_articles = editor_persons.merge(topic_articles[["article_id", "topic_id", "topic_label_human", "year", "language", "type_en", "text_length_words"]], on="article_id", how="left")
    rows = []
    for person_name, group in editor_articles.groupby("person_name"):
        topic_entropy = normalized_entropy(group["topic_id"], n_categories=topic_articles["topic_id"].nunique())
        dominant = group["topic_label_human"].value_counts().index[0] if group["topic_label_human"].notna().any() else pd.NA
        rows.append(
            {
                "person_name": person_name,
                "n_articles": group["article_id"].nunique(),
                "first_year": group["year"].min(),
                "last_year": group["year"].max(),
                "dominant_topic": dominant,
                "n_topics": group["topic_id"].nunique(),
                "topic_entropy_norm": topic_entropy,
            }
        )
    editor_positions = pd.DataFrame(rows)
    if not editor_positions.empty:
        editor_positions = editor_positions.merge(persons, on="person_name", how="left", suffixes=("", "_person"))
    write_csv(editor_positions, ROOT / "outputs/tables/editor_topic_positions.csv")

    if editor_articles.empty:
        editor_matched = pd.DataFrame(columns=["topic_id", "topic_label_human", "n_editor_articles", "editor_share", "n_noneditor_articles", "matched_noneditor_share"])
    else:
        topic_dist_editor = editor_articles.groupby(["topic_id", "topic_label_human"]).size().reset_index(name="n_editor_articles")
        topic_dist_editor["editor_share"] = topic_dist_editor["n_editor_articles"] / max(topic_dist_editor["n_editor_articles"].sum(), 1)
        non_editor_articles = topic_articles[~topic_articles["article_id"].isin(editor_articles["article_id"])]
        topic_dist_non = non_editor_articles.groupby(["topic_id", "topic_label_human"]).size().reset_index(name="n_noneditor_articles")
        topic_dist_non["matched_noneditor_share"] = topic_dist_non["n_noneditor_articles"] / max(topic_dist_non["n_noneditor_articles"].sum(), 1)
        editor_matched = topic_dist_editor.merge(topic_dist_non, on=["topic_id", "topic_label_human"], how="outer").fillna(0)
    write_csv(editor_matched, ROOT / "outputs/tables/editor_vs_matched_noneditor_topic_distribution.csv")

    editor_ranking = editor_positions.sort_values(["topic_entropy_norm", "n_articles"], ascending=False) if not editor_positions.empty else editor_positions
    write_csv(editor_ranking, ROOT / "outputs/tables/editor_brokerage_ranking.csv")

    umap_main = pd.read_csv(ROOT / "outputs/tables/article_umap_main.csv")
    plot_articles = umap_main.merge(topic_articles[["article_id", "topic_label_human"]], on="article_id", how="left")
    editor_article_ids = set(editor_articles["article_id"])
    plot_articles["is_editor_authored"] = plot_articles["article_id"].isin(editor_article_ids)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_articles["umap_x"], plot_articles["umap_y"], s=12, alpha=0.25, color="lightgray")
    highlighted = plot_articles[plot_articles["is_editor_authored"]]
    ax.scatter(highlighted["umap_x"], highlighted["umap_y"], s=28, alpha=0.85, color="#d62728")
    ax.set_title("Editor-authored articles in topic space")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/figures/fig_14_editor_positions_in_topic_space.png", dpi=220)
    save_caption(ROOT, "fig_14_editor_positions_in_topic_space.png", "Editorial actors overlaid on the exploratory semantic map; this indicates positioning, not causal influence.")
    plt.show()

display(editor_positions.head(20))
"""
            ),
        ],
    ),
    (
        "06_actor_profiles_networks_typology.ipynb",
        [
            md(
                """
                # 06 - Actor Profiles, Networks, and Typology

                This notebook builds person-topic profiles and network metrics for recurring contributors. The typology is heuristic and should only be interpreted for manually checked recurring names.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

articles = pd.read_csv(ROOT / "outputs/tables/articles_clean.csv")
article_topics = pd.read_csv(ROOT / "outputs/tables/article_topics.csv")
article_persons = pd.read_csv(ROOT / "outputs/tables/article_persons.csv")
persons = pd.read_csv(ROOT / "outputs/tables/persons_clean.csv")
embedding_index = pd.read_csv(ROOT / "outputs/tables/article_embedding_index.csv")
embeddings = np.load(ROOT / "outputs/models/article_embeddings_main.npy")

person_articles = article_persons.merge(
    article_topics[["article_id", "topic_id", "topic_label_human", "topic_macro_category"]],
    on="article_id",
    how="left",
).merge(articles[["article_id", "year", "issue_id"]], on="article_id", how="left", suffixes=("", "_article"))
embedding_lookup = dict(zip(embedding_index["article_id"], embedding_index["embedding_row"]))
"""
            ),
            md("## Person-Topic Profiles"),
            code(
                r"""
profile_rows = []
person_topic_counts = (
    person_articles.groupby(["person_id", "person_name", "topic_id", "topic_label_human"])
    .size()
    .reset_index(name="n_articles_in_topic")
)
person_totals = person_articles.groupby("person_id")["article_id"].nunique().rename("n_articles_total").reset_index()
person_topic_profiles_long = person_topic_counts.merge(person_totals, on="person_id", how="left")
person_topic_profiles_long["topic_share_person"] = person_topic_profiles_long["n_articles_in_topic"] / person_topic_profiles_long["n_articles_total"]
write_csv(person_topic_profiles_long, ROOT / "outputs/tables/person_topic_matrix_long.csv")

for person_id, group in person_articles.groupby("person_id"):
    group = group.drop_duplicates("article_id")
    topic_counts = group["topic_id"].value_counts()
    dominant_topic = group["topic_label_human"].value_counts().index[0] if group["topic_label_human"].notna().any() else pd.NA
    dominant_share = float(topic_counts.iloc[0] / topic_counts.sum()) if len(topic_counts) else np.nan
    entropy_norm = normalized_entropy(group["topic_id"], n_categories=article_topics["topic_id"].nunique())
    rows = [embedding_lookup.get(aid) for aid in group["article_id"] if aid in embedding_lookup]
    dispersion = cosine_centroid_dispersion(embeddings[rows]) if len(rows) else np.nan
    macro = group["topic_macro_category"].fillna("")
    profile_rows.append(
        {
            "person_id": person_id,
            "person_name": group["person_name"].iloc[0],
            "n_articles": group["article_id"].nunique(),
            "first_year": group["year"].min(),
            "last_year": group["year"].max(),
            "active_span": group["year"].max() - group["year"].min() + 1,
            "n_topics": group["topic_id"].nunique(),
            "dominant_topic": dominant_topic,
            "dominant_topic_share": dominant_share,
            "topic_entropy_norm": entropy_norm,
            "effective_n_topics": float(np.exp(shannon_entropy(group["topic_id"]))),
            "method_share": float(macro.eq("methods_and_methodology").mean()),
            "data_infrastructure_share": float(macro.eq("data_and_infrastructure").mean()),
            "substantive_topic_share": float((~macro.isin(["methods_and_methodology", "data_and_infrastructure", ""])).mean()),
            "semantic_dispersion_highdim": dispersion,
        }
    )
profiles = pd.DataFrame(profile_rows).merge(
    persons[["person_id", "is_editor_any", "is_special_issue_editor", "is_managing_editor", "editor_role_certainty"]],
    on="person_id",
    how="left",
)
profiles[["is_editor_any", "is_special_issue_editor", "is_managing_editor"]] = profiles[["is_editor_any", "is_special_issue_editor", "is_managing_editor"]].fillna(False)
profiles["editor_role_certainty"] = profiles["editor_role_certainty"].fillna("unknown")
write_csv(profiles, ROOT / "outputs/tables/person_topic_profiles.csv")
display(profiles.sort_values("n_articles", ascending=False).head(20))
"""
            ),
            md("## Networks"),
            code(
                r"""
G = nx.Graph()
for article_id, group in person_articles.groupby("article_id"):
    people = sorted(set(group["person_id"].dropna()))
    for person in people:
        G.add_node(person)
    for a, b in combinations(people, 2):
        if G.has_edge(a, b):
            G[a][b]["weight"] += 1
        else:
            G.add_edge(a, b, weight=1)

if G.number_of_nodes():
    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True) if G.number_of_edges() else {n: 0 for n in G.nodes}
    pagerank = nx.pagerank(G, weight="weight") if G.number_of_edges() else {n: 1 / G.number_of_nodes() for n in G.nodes}
    components = {node: cid for cid, comp in enumerate(nx.connected_components(G)) for node in comp}
else:
    degree = weighted_degree = betweenness = pagerank = components = {}

participation = {}
for person_id, group in person_articles.groupby("person_id"):
    counts = group["topic_id"].value_counts()
    total = counts.sum()
    participation[person_id] = float(1 - ((counts / total) ** 2).sum()) if total else 0.0

network_metrics = pd.DataFrame(
    [
        {
            "person_id": node,
            "degree_centrality": degree.get(node, 0),
            "weighted_degree": weighted_degree.get(node, 0),
            "betweenness_centrality": betweenness.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "participation_coefficient_across_topics": participation.get(node, 0),
            "brokerage_score": betweenness.get(node, 0) * participation.get(node, 0),
            "network_component": components.get(node, -1),
        }
        for node in set(list(degree.keys()) + list(profiles["person_id"]))
    ]
).merge(profiles[["person_id", "person_name", "n_articles"]], on="person_id", how="left")
write_csv(network_metrics, ROOT / "outputs/tables/person_network_metrics.csv")

fig, ax = plt.subplots(figsize=(9, 7))
core_nodes = [n for n, d in degree.items() if d >= 2]
sub = G.subgraph(core_nodes).copy()
if sub.number_of_nodes():
    pos = nx.spring_layout(sub, seed=42, k=0.35)
    article_count_lookup = profiles.drop_duplicates("person_id").set_index("person_id")["n_articles"].to_dict()
    sizes = [max(20, article_count_lookup.get(n, 1) * 18) for n in sub.nodes]
    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.18, width=0.8)
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=sizes, node_color="#4c78a8", alpha=0.75)
ax.set_title("Co-authorship network core")
ax.axis("off")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_11_coauthorship_network_core.png", dpi=220)
save_caption(ROOT, "fig_11_coauthorship_network_core.png", "Core co-authorship network; network measures are sensitive to name disambiguation and corpus coverage.")
plt.show()
display(network_metrics.sort_values("brokerage_score", ascending=False).head(20))
"""
            ),
            md("## Actor Typology"),
            code(
                r"""
typed = profiles.merge(
    network_metrics[["person_id", "betweenness_centrality", "participation_coefficient_across_topics", "brokerage_score"]],
    on="person_id",
    how="left",
)
typed[["betweenness_centrality", "participation_coefficient_across_topics", "brokerage_score"]] = typed[["betweenness_centrality", "participation_coefficient_across_topics", "brokerage_score"]].fillna(0)

entropy_q67 = typed["topic_entropy_norm"].quantile(0.67)
broker_q67 = typed["brokerage_score"].quantile(0.67)
disp_q67 = typed["semantic_dispersion_highdim"].quantile(0.67)

def assign_type(row):
    if row["n_articles"] < 2:
        return "One-time contributors"
    if bool(row.get("is_editor_any", False)) and (row["active_span"] >= 10 or row["n_articles"] >= 3):
        return "Institutional anchors"
    if row["n_articles"] >= 2 and row["method_share"] + row["data_infrastructure_share"] >= 0.5:
        return "Methods and data architects"
    if row["n_articles"] >= 3 and row["dominant_topic_share"] >= 0.60 and row["topic_entropy_norm"] <= 0.35:
        return "Thematic specialists"
    if row["n_articles"] >= 3 and row["topic_entropy_norm"] >= entropy_q67 and row["brokerage_score"] >= broker_q67 and row["semantic_dispersion_highdim"] <= disp_q67:
        return "Bridge builders"
    if row["n_articles"] >= 3 and row["topic_entropy_norm"] >= entropy_q67 and row["semantic_dispersion_highdim"] >= disp_q67:
        return "Wide-ranging contributors"
    return "Recurring contributors"

typed["actor_type"] = typed.apply(assign_type, axis=1)
typed["typology_uncertainty"] = np.where(typed["n_articles"] < 3, "high: fewer than three articles", "moderate")
typed["topic_entrepreneur_candidate"] = False
write_csv(typed, ROOT / "outputs/tables/person_typology.csv")

sensitivity = typed[["person_id", "person_name", "actor_type", "n_articles", "topic_entropy_norm", "brokerage_score", "semantic_dispersion_highdim", "typology_uncertainty"]].copy()
write_csv(sensitivity, ROOT / "outputs/tables/person_typology_sensitivity.csv")

plot_typed = typed[typed["n_articles"] >= 2].copy()
fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(
    data=plot_typed,
    x="topic_entropy_norm",
    y="brokerage_score",
    hue="actor_type",
    style="is_editor_any",
    size="n_articles",
    sizes=(30, 350),
    alpha=0.8,
    ax=ax,
)
ax.set_title("Actor typology matrix")
ax.set_xlabel("Thematic breadth")
ax.set_ylabel("Brokerage score")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_13_actor_typology_matrix.png", dpi=220)
save_caption(ROOT, "fig_13_actor_typology_matrix.png", "Recurring contributors positioned by thematic breadth and brokerage; types are heuristic and uncertainty depends on name and role validation.")
plt.show()
display(typed["actor_type"].value_counts())
"""
            ),
        ],
    ),
    (
        "07_article_type_and_method_labels.ipynb",
        [
            md(
                """
                # 07 - Article Type and Method Labels

                This notebook creates transparent rule-based prelabels for contribution type and methodological approach. It also exports a stratified validation sample for human review or later LLM-assisted labeling.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import re

articles = pd.read_csv(ROOT / "outputs/tables/articles_clean.csv")
article_topics = pd.read_csv(ROOT / "outputs/tables/article_topics.csv")
df = articles.merge(article_topics[["article_id", "topic_id", "topic_label_human", "topic_macro_category"]], on="article_id", how="left")
df["label_text"] = df[["title", "abstract_en", "abstract_de", "keywords_en", "type_en"]].fillna("").agg(" ".join, axis=1).map(lambda x: x.lower())
"""
            ),
            code(
                r"""
def contribution_type(text, row):
    if bool(row.get("is_editorial_or_intro", False)):
        return "editorial_introduction"
    if bool(row.get("is_review", False)) or bool(row.get("is_bibliography", False)):
        return "review_or_bibliography"
    if re.search(r"\bdatabase\b|\bdata archive\b|\binfrastructure\b|\bdocumentation\b|\bdataset\b", text):
        return "research_infrastructure_or_data"
    if re.search(r"\bmethod\b|\bmethodology\b|\bmeasurement\b|\bcoding\b|\bsampling\b|\bclassification\b", text):
        return "methodological"
    if re.search(r"\bempirical\b|\banalysis\b|\bstudy\b|\bsurvey\b|\binterview\b|\bcase study\b|\bdata\b", text):
        return "empirical_analysis"
    if re.search(r"\btheory\b|\bconceptual\b|\bconcept\b|\bframework\b|\bcritique\b|\bnormative\b", text):
        return "conceptual_theoretical"
    return "mixed_or_unclear"

def method_approach(text):
    if re.search(r"\bregression\b|\bstatistic\b|\bquantitative\b|\bsurvey\b|\bpanel\b|\btime series\b|\bmodel\b", text):
        return "quantitative"
    if re.search(r"\binterview\b|\bethnograph\b|\bqualitative\b|\bgrounded theory\b|\bfocus group\b", text):
        return "qualitative"
    if re.search(r"\bmixed methods\b|\bmixed-method\b", text):
        return "mixed_methods"
    if re.search(r"\bdigital\b|\bcomputational\b|\balgorithm\b|\bgis\b|\bnetwork\b|\btext mining\b|\bmachine learning\b", text):
        return "computational_digital"
    if re.search(r"\barchive\b|\bhistorical narrative\b|\bsource\b|\bcase history\b", text):
        return "historical_narrative_or_archival"
    if re.search(r"\bformal\b|\bmathematical\b|\bgame theory\b", text):
        return "formal_theoretical"
    if re.search(r"\bconceptual\b|\bnormative\b|\btheoretical\b|\btheory\b", text):
        return "conceptual_normative"
    if re.search(r"\binfrastructure\b|\bdocumentation\b|\bdatabase\b|\bdata archive\b", text):
        return "infrastructure_documentation"
    return "unclear"

labels = df[["article_id", "year", "period", "language", "type_en", "topic_id", "topic_label_human", "topic_macro_category"]].copy()
labels["contribution_type_rule"] = df.apply(lambda row: contribution_type(row["label_text"], row), axis=1)
labels["method_approach_rule"] = df["label_text"].map(method_approach)
labels["contribution_type_human"] = ""
labels["method_approach_human"] = ""
labels["label_status"] = "rule_based_preset"
write_csv(labels, ROOT / "outputs/tables/article_type_labels.csv")

sample_size = max(100, int(0.05 * len(labels)))
sample_size = min(sample_size, len(labels))
strata = labels[["period", "language", "topic_macro_category", "contribution_type_rule"]].fillna("unknown").agg("|".join, axis=1)
try:
    if strata.value_counts().min() >= 2 and strata.nunique() < sample_size:
        sample, _ = train_test_split(labels, train_size=sample_size, random_state=42, stratify=strata)
    else:
        sample = labels.sample(n=sample_size, random_state=42)
except ValueError:
    sample = labels.sample(n=sample_size, random_state=42)
sample = sample.sort_values(["period", "topic_macro_category", "year"])
sample["validator_1_contribution_type"] = ""
sample["validator_1_method_approach"] = ""
sample["validator_1_notes"] = ""
write_csv(sample, ROOT / "outputs/labels/article_type_validation_sample.csv")

validation_path = ROOT / "outputs/labels/article_type_validation_sample.csv"
validated = pd.read_csv(validation_path)
results = []
if "validator_1_contribution_type" in validated.columns and validated["validator_1_contribution_type"].map(compact_ws).ne("").any():
    checked = validated[validated["validator_1_contribution_type"].map(compact_ws).ne("")]
    results.append({"metric": "n_validated_contribution_type", "value": len(checked)})
    results.append({"metric": "rule_human_agreement_contribution_type", "value": float((checked["validator_1_contribution_type"] == checked["contribution_type_rule"]).mean())})
if "validator_1_method_approach" in validated.columns and validated["validator_1_method_approach"].map(compact_ws).ne("").any():
    checked = validated[validated["validator_1_method_approach"].map(compact_ws).ne("")]
    results.append({"metric": "n_validated_method_approach", "value": len(checked)})
    results.append({"metric": "rule_human_agreement_method_approach", "value": float((checked["validator_1_method_approach"] == checked["method_approach_rule"]).mean())})
write_csv(pd.DataFrame(results, columns=["metric", "value"]), ROOT / "outputs/tables/article_type_validation_results.csv")
display(labels.head())
"""
            ),
            code(
                r"""
type_year = labels.groupby(["year", "contribution_type_rule"]).size().reset_index(name="n")
type_year["share"] = type_year["n"] / type_year.groupby("year")["n"].transform("sum")
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=type_year, x="year", y="share", hue="contribution_type_rule", ax=ax)
ax.set_title("Contribution type over time")
ax.set_xlabel("Year")
ax.set_ylabel("Share")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Contribution type")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_16_article_type_over_time.png", dpi=220)
save_caption(ROOT, "fig_16_article_type_over_time.png", "Rule-based contribution-type prelabels over time; final use requires validation.")
plt.show()

method_year = labels.groupby(["year", "method_approach_rule"]).size().reset_index(name="n")
method_year["share"] = method_year["n"] / method_year.groupby("year")["n"].transform("sum")
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=method_year, x="year", y="share", hue="method_approach_rule", ax=ax)
ax.set_title("Methodological approach over time")
ax.set_xlabel("Year")
ax.set_ylabel("Share")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Method")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_17_methodological_approach_over_time.png", dpi=220)
save_caption(ROOT, "fig_17_methodological_approach_over_time.png", "Rule-based method-approach prelabels over time; ambiguous records require human review.")
plt.show()
"""
            ),
        ],
    ),
    (
        "08_citations_internationalization_and_summary.ipynb",
        [
            md(
                """
                # 08 - Citations, Internationalization, External Context, and Summary

                This notebook keeps citation and diversity/internationalization analyses modular. It writes conservative outputs when citation or affiliation coverage is incomplete and generates the final analysis summary and reviewer-defense checklist.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
import matplotlib.pyplot as plt
import seaborn as sns

articles = pd.read_csv(ROOT / "outputs/tables/articles_clean.csv")
citations_path = ROOT / "citation_metadata_links.csv"
citations = pd.read_csv(citations_path) if citations_path.exists() else pd.DataFrame()
print(f"Citation rows: {len(citations):,}")
"""
            ),
            md("## Citation and Internationalization Metrics"),
            code(
                r"""
if citations.empty:
    citation_language = pd.DataFrame(columns=["LA", "n_records", "mean_citations", "median_citations"])
    affiliations = pd.DataFrame(columns=["citation_id", "affiliation", "country", "year"])
else:
    citations["Z9_numeric"] = pd.to_numeric(citations.get("Z9"), errors="coerce")
    citation_language = (
        citations.groupby("LA", dropna=False)
        .agg(
            n_records=("citation_id", "nunique"),
            mean_citations=("Z9_numeric", "mean"),
            median_citations=("Z9_numeric", "median"),
            n_matched_to_metadata=("meta_handle", lambda x: x.notna().sum()),
        )
        .reset_index()
    )

    country_aliases = {
        "Zealand": "New Zealand",
        "Tobago": "Trinidad and Tobago",
        "Republic": "Czech Republic",
        "Korea": "South Korea",
        "Emirates": "United Arab Emirates",
        "England": "United Kingdom",
        "Scotland": "United Kingdom",
        "Wales": "United Kingdom",
        "USA": "United States",
        "Africa": "South Africa",
    }
    affiliation_rows = []
    for row in citations.itertuples():
        c1 = compact_ws(getattr(row, "C1", ""))
        if not c1:
            continue
        parts = [part.strip() for part in c1.split("[") if part.strip()]
        for part in parts:
            country = part.split()[-1].strip(".,;)")
            country = country_aliases.get(country, country)
            affiliation_rows.append({"citation_id": row.citation_id, "affiliation": part, "country": country, "year": getattr(row, "year", pd.NA)})
    affiliations = pd.DataFrame(affiliation_rows)

write_csv(citation_language, ROOT / "outputs/tables/citation_language_metrics.csv")
write_csv(affiliations, ROOT / "outputs/tables/affiliations_extracted_from_citations.csv")

if not affiliations.empty:
    country_year = affiliations.groupby(["year", "country"]).size().reset_index(name="n_affiliations")
    totals = country_year.groupby("year")["n_affiliations"].transform("sum")
    country_year["share"] = country_year["n_affiliations"] / totals
    top_countries = affiliations["country"].value_counts().head(8).index
    plot_country = country_year[country_year["country"].isin(top_countries)]
else:
    country_year = pd.DataFrame(columns=["year", "country", "n_affiliations", "share"])
    plot_country = country_year

language_year = articles.groupby(["year", "language"]).size().reset_index(name="n_articles")
language_year["share"] = language_year["n_articles"] / language_year.groupby("year")["n_articles"].transform("sum")
internationalization = {
    "n_articles": len(articles),
    "n_citation_records": len(citations),
    "n_affiliation_rows_extracted": len(affiliations),
    "citation_metadata_coverage_share": float(citations["meta_handle"].notna().mean()) if not citations.empty and "meta_handle" in citations.columns else np.nan,
}
write_csv(pd.DataFrame([internationalization]), ROOT / "outputs/tables/internationalization_metrics.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.lineplot(data=language_year, x="year", y="share", hue="language", ax=axes[0])
axes[0].set_title("Publication language")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Share")
if not plot_country.empty:
    sns.lineplot(data=plot_country, x="year", y="share", hue="country", ax=axes[1])
axes[1].set_title("Affiliation countries from citation metadata")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Share of extracted affiliations")
axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/fig_20_language_and_affiliation_over_time.png", dpi=220)
save_caption(ROOT, "fig_20_language_and_affiliation_over_time.png", "Publication language and extracted affiliation countries over time; affiliation coverage depends on available citation metadata.")
plt.show()
display(citation_language)
"""
            ),
            md("## Optional External Map Placeholders"),
            code(
                r"""
external_sample_path = ROOT / "outputs/tables/external_literature_sample.csv"
radiation_path = ROOT / "outputs/tables/radiation_power_clusters.csv"
if not external_sample_path.exists():
    write_csv(
        pd.DataFrame(
            columns=[
                "external_id",
                "title",
                "year",
                "source",
                "abstract",
                "embedding_status",
                "notes",
            ]
        ),
        external_sample_path,
    )
if not radiation_path.exists():
    write_csv(
        pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_label",
                "n_hsr_articles",
                "n_external_records",
                "interpretation_status",
                "notes",
            ]
        ),
        radiation_path,
    )
write_markdown(
    "# External Context Module\n\nNo external background map has been computed in this notebook. Use this module only when OpenAlex, Crossref, Web of Science, Scopus, or another comparison sample has sufficient coverage. Until then, phrase this as exploratory external semantic positioning, not radiation power.",
    ROOT / "outputs/diagnostics/external_context_status.md",
)
"""
            ),
            md("## Final Summary and Reviewer Defense Checklist"),
            code(
                r"""
def read_optional(path):
    path = ROOT / path
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

audit = read_optional("outputs/tables/corpus_audit.csv")
labels = read_optional("outputs/tables/topic_labels.csv")
innovation = read_optional("outputs/tables/topic_innovation_metrics.csv")
typology = read_optional("outputs/tables/person_typology.csv")
figures = sorted([p.name for p in (ROOT / "outputs/figures").glob("*.png")] + [p.name for p in (ROOT / "outputs/figures").glob("*.html")])

metric = dict(zip(audit.get("metric", []), audit.get("value", []))) if not audit.empty else {}
persistent_topics = innovation[innovation.get("is_stable_core", pd.Series(dtype=bool)).fillna(False)] if not innovation.empty else pd.DataFrame()

summary = [
    "# HSR Analysis Summary",
    "",
    f"- Corpus records with valid years: {metric.get('n_records_valid_year', 'not computed')}",
    f"- Articles with sufficient text: {metric.get('n_sufficient_text', 'not computed')}",
    f"- Core inclusion records: {metric.get('n_core_inclusion', 'not computed')}",
    f"- Candidate topics/clusters: {len(labels) if not labels.empty else 'not computed'}",
    f"- Persistent topics by operational heuristic: {len(persistent_topics) if not persistent_topics.empty else 'not computed'}",
    f"- Actor types assigned: {typology['actor_type'].nunique() if not typology.empty and 'actor_type' in typology else 'not computed'}",
    "",
    "## Generated Figures",
    "",
]
summary.extend(f"- `{figure}`" for figure in figures)
summary.extend(
    [
        "",
        "## Open Manual Steps",
        "",
        "- Fill `outputs/labels/topic_label_overrides.csv` with human topic labels and rationales.",
        "- Review `outputs/tables/top_persons_manual_check.csv` and add corrections to `data/name_corrections.csv`.",
        "- Fill `data/editors_roles.csv` with historically sourced editor and issue-editor roles.",
        "- Validate `outputs/labels/article_type_validation_sample.csv` before interpreting article-type trends.",
        "- Decide whether the optional external-context module has enough coverage to enter the main article.",
    ]
)
write_markdown("\n".join(summary), ROOT / "outputs/HSR_analysis_summary.md")

checklist = [
    "# Reviewer Defense Checklist",
    "",
    "## Why Embeddings Instead of LDA/STM?",
    "The pipeline uses multilingual sentence embeddings because the corpus spans German, English, and a small number of French records, and because the project aims to map semantic neighborhoods rather than infer latent topics as ontological entities. Clusters are described as candidate topics or semantic regions.",
    "",
    "## How Stable Are the Clusters?",
    "Cluster solutions are compared in `outputs/tables/topic_stability_metrics.csv` and visualized in `fig_model_stability_atlas.png`. Only stable regions should be discussed as robust findings.",
    "",
    "## How Was Guided Confirmation Avoided?",
    "Seed topics are used only in `outputs/tables/guided_vs_unguided_topic_comparison.csv` as a comparison against unguided embedding clusters. They do not determine the main cluster assignment.",
    "",
    "## How Were Language and Length Effects Checked?",
    "Language and text-length diagnostics are written to `outputs/tables/language_cluster_diagnostics.csv`, `outputs/tables/text_length_cluster_diagnostics.csv`, `fig_language_in_embedding_space.png`, and `fig_text_length_in_embedding_space.png`.",
    "",
    "## How Were Reviews, Editorials, and Bibliographies Treated?",
    "They are flagged in `outputs/tables/articles_clean.csv` and can be included or excluded through core/extended sensitivity analyses. They are not silently removed.",
    "",
    "## How Were Historical Editor Roles Reconstructed?",
    "Editor roles come from the manual, source-bearing file `data/editors_roles.csv`. If the file is empty or uncertain, editor analyses are treated as approximations.",
    "",
    "## Why Is Editorial Positioning Not a Causal Claim?",
    "The notebooks calculate co-occurrence, semantic position, issue concentration, and pre/post patterns. These are consistent with curatorial signatures but do not establish editor intention or causal mechanisms.",
    "",
    "## How Were LLM Labels Validated?",
    "The default notebook path uses transparent machine labels from evidence packs and requires manual overrides in `outputs/labels/topic_label_overrides.csv`. Any future LLM labeling should be saved with prompts and validation status.",
    "",
    "## Core vs Exploratory Analyses",
    "Core analyses: corpus audit, embeddings, clustering/stability, human-supervised labels, topic trends, person-topic profiles, issue concentration. Exploratory analyses: external field mapping, radiation-power language, and any diversity inference beyond aggregate language/affiliation patterns.",
    "",
    "## Claims to Avoid",
    "Avoid claims that models discover true topics, that UMAP point distances are metric intellectual distances, that editors caused topics, or that name-based demographic inference is definitive.",
]
write_markdown("\n".join(checklist), ROOT / "outputs/reviewer_defense_checklist.md")

print("Wrote outputs/HSR_analysis_summary.md and outputs/reviewer_defense_checklist.md")
"""
            ),
        ],
    ),
]


def write_notebook(filename: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"].update(KERNEL)
    nb["cells"] = cells
    path = NOTEBOOK_DIR / filename
    nbf.write(nb, path)
    print(path.relative_to(ROOT))


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in NOTEBOOKS:
        write_notebook(filename, cells)


if __name__ == "__main__":
    main()
