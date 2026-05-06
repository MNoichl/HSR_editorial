from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUTPUT_SUBDIRS = [
    "outputs",
    "outputs/tables",
    "outputs/figures",
    "outputs/models",
    "outputs/models/umap_runs",
    "outputs/labels",
    "outputs/diagnostics",
    "outputs/diagnostics/topic_evidence",
    "outputs/prompts",
    "data",
    "data/validation",
]

ARTICLE_SOURCE_CANDIDATES = [
    "data/HSR_full_data.csv",
    "HSR_full_data.csv",
    "intermediate_datasets/hsr_metadata_with_texts.csv",
    "HSR/hsr_metadata_fulltexts/hsr.csv",
]

MANUAL_TEMPLATE_COLUMNS = {
    "data/name_corrections.csv": [
        "raw_name",
        "corrected_name",
        "certainty",
        "notes",
    ],
    "data/editors_roles.csv": [
        "person_name",
        "role_type",
        "start_year",
        "end_year",
        "source",
        "certainty",
        "notes",
    ],
    "data/issue_editor_corrections.csv": [
        "issue_id",
        "issue_editor_raw",
        "person_name",
        "certainty",
        "source",
        "notes",
    ],
    "data/issue_corrections.csv": [
        "issue_id",
        "manual_issue_status",
        "issue_title_corrected",
        "certainty",
        "source",
        "notes",
    ],
    "data/article_exclusions.csv": [
        "article_id",
        "exclude_from_core",
        "exclude_from_extended",
        "reason",
        "notes",
    ],
}

PERIODS = [
    (1976, 1989, "1976-1989: founding and quantitative-historical phase"),
    (1990, 1999, "1990-1999: consolidation and transformation"),
    (2000, 2009, "2000-2009: internationalization and methodological expansion"),
    (2010, 2019, "2010-2019: special-issue expansion and thematic pluralization"),
    (2020, 2026, "2020-2026: contemporary reorientation and digital/knowledge infrastructures"),
]

MACRO_CATEGORY_KEYWORDS = {
    "methods_and_methodology": [
        "method",
        "methodology",
        "qualitative",
        "quantitative",
        "survey",
        "interview",
        "sampling",
        "measurement",
        "classification",
        "sequence",
        "longitudinal",
        "statistics",
        "model",
    ],
    "data_and_infrastructure": [
        "data",
        "database",
        "archive",
        "infrastructure",
        "coding",
        "digital",
        "computer",
        "gis",
        "information system",
        "openalex",
    ],
    "politics_elites_state": [
        "politic",
        "elite",
        "state",
        "party",
        "election",
        "voting",
        "government",
        "parliament",
        "regime",
    ],
    "migration_borders_mobility": [
        "migration",
        "migrant",
        "mobility",
        "border",
        "asylum",
        "refugee",
        "diaspora",
    ],
    "demography_family_population": [
        "demography",
        "family",
        "fertility",
        "population",
        "household",
        "marriage",
        "birth",
    ],
    "health_care_mortality": [
        "health",
        "care",
        "mortality",
        "death",
        "pandemic",
        "disease",
        "medicine",
    ],
    "sport_body_culture": [
        "sport",
        "football",
        "body",
        "leisure",
        "games",
    ],
    "conventions_law_economy": [
        "convention",
        "law",
        "econom",
        "capitalism",
        "market",
        "classification",
        "institution",
    ],
    "science_technology_sts": [
        "science",
        "technology",
        "expertise",
        "knowledge",
        "sts",
        "innovation",
    ],
    "environment_energy_sustainability": [
        "environment",
        "climate",
        "sustainability",
        "energy",
        "ecology",
    ],
    "war_violence_security": [
        "war",
        "violence",
        "security",
        "risk",
        "genocide",
        "military",
    ],
    "culture_media_visuality": [
        "culture",
        "media",
        "visual",
        "image",
        "communication",
        "memory",
        "discourse",
    ],
}


def project_root_from_notebook() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


def ensure_project_structure(root: Path) -> None:
    for relative in OUTPUT_SUBDIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative, columns in MANUAL_TEMPLATE_COLUMNS.items():
        path = root / relative
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)


def first_existing(root: Path, candidates: Iterable[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exists: " + ", ".join(candidates))


def load_article_source(root: Path) -> tuple[pd.DataFrame, Path]:
    path = first_existing(root, ARTICLE_SOURCE_CANDIDATES)
    df = pd.read_csv(path)
    return df, path


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text.strip()


def compact_ws(text: object) -> str:
    return re.sub(r"\s+", " ", as_text(text)).strip()


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def slugify(text: object, max_len: int = 80) -> str:
    cleaned = strip_accents(compact_ws(text).lower())
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return cleaned[:max_len] or "unknown"


def stable_hash(parts: Iterable[object], length: int = 12) -> str:
    payload = "||".join(compact_ws(part) for part in parts)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:length]


def period_from_year(year: object) -> str:
    value = pd.to_numeric(pd.Series([year]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "unknown"
    year_int = int(value)
    for start, end, label in PERIODS:
        if start <= year_int <= end:
            return label
    return "outside_target_range"


def make_article_id(row: pd.Series) -> str:
    handle = compact_ws(row.get("handle", ""))
    if handle:
        return "hsr_" + slugify(handle.replace("document/", "document_"), max_len=64)
    document_id = compact_ws(row.get("document_id", ""))
    if document_id:
        return "hsr_document_" + slugify(document_id, max_len=64)
    return "hsr_" + stable_hash([row.get("year"), row.get("title"), row.get("authors")])


def make_issue_id(row: pd.Series) -> str:
    journal = slugify(row.get("journal", "hsr"), max_len=30)
    year = compact_ws(row.get("year", "unknown")) or "unknown"
    volume = compact_ws(row.get("volume", "unknown")) or "unknown"
    issue = compact_ws(row.get("issue", "unknown")) or "unknown"
    return f"{journal}_{slugify(year)}_v{slugify(volume)}_i{slugify(issue)}"


def text_length_words(text: object) -> int:
    value = compact_ws(text)
    if not value:
        return 0
    return len(re.findall(r"\b\w+\b", value))


def join_nonempty(parts: Iterable[object], separator: str = "\n\n") -> str:
    return separator.join(part for part in (compact_ws(p) for p in parts) if part)


def document_flags(row: pd.Series) -> dict[str, bool]:
    title = compact_ws(row.get("title", "")).lower()
    issue_title = compact_ws(row.get("issue_title", "")).lower()
    type_en = compact_ws(row.get("type_en", "")).lower()
    journal = compact_ws(row.get("journal", "")).lower()
    text = " ".join([title, issue_title, type_en, journal])

    is_review = bool(re.search(r"\breview\b|\breviews\b|rezension|book review", text))
    is_editorial_or_intro = bool(
        re.search(r"\beditorial\b|\bintroduction\b|\bintroductory\b|preface|foreword|einleitung", text)
    )
    is_bibliography = bool(re.search(r"bibliograph|bibliography|bibliografie", text))
    is_anniversary = bool(
        re.search(r"anniversary|jubila|fifty|50th|autobiograph|autobiograf|gedenk|memorial", text)
    )
    is_supplement = "supplement" in text or "beiheft" in text
    return {
        "is_supplement": is_supplement,
        "is_review": is_review,
        "is_editorial_or_intro": is_editorial_or_intro,
        "is_bibliography": is_bibliography,
        "is_autobiographical_or_anniversary": is_anniversary,
    }


def build_articles_clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "document_id" not in df.columns and "handle" in df.columns:
        df.insert(0, "document_id", df["handle"].map(lambda x: compact_ws(x).replace("document/", "")))

    df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")
    df = df[df["year"].notna()].copy()
    df["article_id"] = df.apply(make_article_id, axis=1)
    df["decade"] = (df["year"].astype(int) // 10 * 10).astype(str) + "s"
    df["period"] = df["year"].map(period_from_year)
    df["issue_id"] = df.apply(make_issue_id, axis=1)

    for col in [
        "title",
        "authors",
        "abstract_en",
        "abstract_de",
        "keywords_en",
        "keywords_de",
        "subjects_en",
        "subjects_de",
        "full_text",
        "language",
        "journal",
        "type_en",
        "volume",
        "issue",
        "issue_title",
        "doi",
    ]:
        if col not in df.columns:
            df[col] = ""

    df["text_for_embedding"] = df.apply(
        lambda row: join_nonempty(
            [
                row.get("title"),
                row.get("abstract_en"),
                row.get("abstract_de"),
                row.get("keywords_en"),
                row.get("keywords_de"),
                row.get("subjects_en"),
                row.get("subjects_de"),
            ]
        ),
        axis=1,
    )
    df["text_for_embedding_full"] = df.apply(
        lambda row: join_nonempty([row.get("text_for_embedding"), compact_ws(row.get("full_text"))[:15000]]),
        axis=1,
    )
    df["text_for_labeling"] = df.apply(
        lambda row: join_nonempty([row.get("text_for_embedding"), compact_ws(row.get("full_text"))[:3500]]),
        axis=1,
    )

    df["text_length_chars"] = df["text_for_embedding"].map(len)
    df["text_length_words"] = df["text_for_embedding"].map(text_length_words)
    df["full_text_length_chars"] = df["full_text"].map(lambda x: len(as_text(x)))
    df["has_sufficient_text"] = (df["text_length_chars"] >= 120) | (df["full_text_length_chars"] >= 1000)

    flags = pd.DataFrame([document_flags(row) for _, row in df.iterrows()], index=df.index)
    df = pd.concat([df, flags], axis=1)
    df["corpus_inclusion_core"] = df["has_sufficient_text"]
    df["corpus_inclusion_extended"] = df["has_sufficient_text"]

    preferred = [
        "article_id",
        "year",
        "decade",
        "period",
        "title",
        "authors",
        "abstract_en",
        "abstract_de",
        "full_text",
        "text_for_embedding",
        "text_for_embedding_full",
        "text_for_labeling",
        "language",
        "journal",
        "type_en",
        "keywords_en",
        "volume",
        "issue",
        "issue_id",
        "issue_title",
        "is_supplement",
        "is_review",
        "is_editorial_or_intro",
        "is_bibliography",
        "is_autobiographical_or_anniversary",
        "has_sufficient_text",
        "text_length_chars",
        "text_length_words",
        "full_text_length_chars",
        "corpus_inclusion_core",
        "corpus_inclusion_extended",
        "doi",
        "handle",
        "document_id",
    ]
    return df[[col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]]


def load_name_corrections(root: Path) -> dict[str, str]:
    path = root / "data/name_corrections.csv"
    if not path.exists():
        return {}
    corrections = pd.read_csv(path)
    if corrections.empty:
        return {}
    corrections = corrections.dropna(subset=["raw_name", "corrected_name"], how="any")
    return {
        compact_ws(row["raw_name"]).lower(): compact_ws(row["corrected_name"])
        for _, row in corrections.iterrows()
        if compact_ws(row["raw_name"]) and compact_ws(row["corrected_name"])
    }


def split_author_string(authors: object) -> list[str]:
    text = compact_ws(authors)
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"\s*;\s*", text) if part.strip()]
    return [part for part in parts if part and part.lower() != "[anonymous]"]


def normalize_person_name(name: object, corrections: dict[str, str] | None = None) -> tuple[str, str]:
    corrections = corrections or {}
    raw = compact_ws(name)
    corrected = corrections.get(raw.lower(), raw)
    corrected = re.sub(r"\s+", " ", corrected).strip(" ;")
    person_id = "person_" + slugify(corrected, max_len=70)
    return corrected, person_id


def build_article_persons(articles: pd.DataFrame, corrections: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for _, article in articles.iterrows():
        authors = split_author_string(article.get("authors", ""))
        for position, raw_name in enumerate(authors, start=1):
            person_name, person_id = normalize_person_name(raw_name, corrections)
            rows.append(
                {
                    "article_id": article["article_id"],
                    "person_raw": raw_name,
                    "person_name": person_name,
                    "person_id": person_id,
                    "position_in_author_list": position,
                    "role": "author",
                    "name_disambiguation_status": "corrected" if raw_name != person_name else "raw_normalized",
                    "year": article.get("year"),
                    "title": article.get("title"),
                    "issue_id": article.get("issue_id"),
                }
            )
    return pd.DataFrame(rows)


def name_diagnostics(article_persons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if article_persons.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    variants = (
        article_persons.groupby("person_id")
        .agg(
            person_name=("person_name", "first"),
            n_name_variants=("person_raw", "nunique"),
            raw_variants=("person_raw", lambda x: " | ".join(sorted(set(map(compact_ws, x))))),
            n_articles=("article_id", "nunique"),
        )
        .reset_index()
        .sort_values(["n_name_variants", "n_articles"], ascending=False)
    )

    tmp = article_persons[["person_name", "person_id", "article_id"]].drop_duplicates().copy()
    tmp["name_ascii"] = tmp["person_name"].map(lambda x: strip_accents(compact_ws(x)).lower())
    tmp["last_key"] = tmp["name_ascii"].map(lambda x: x.split(",")[0].strip() if "," in x else x.split()[-1] if x else "")
    tmp["first_initial"] = tmp["name_ascii"].map(
        lambda x: (x.split(",")[1].strip()[:1] if "," in x and len(x.split(",")) > 1 else x[:1])
    )
    dup = (
        tmp.groupby(["last_key", "first_initial"])
        .agg(
            possible_person_ids=("person_id", lambda x: " | ".join(sorted(set(x)))),
            possible_names=("person_name", lambda x: " | ".join(sorted(set(x)))),
            n_person_ids=("person_id", "nunique"),
            n_articles=("article_id", "nunique"),
        )
        .reset_index()
    )
    possible_duplicates = dup[dup["n_person_ids"] > 1].sort_values(
        ["n_person_ids", "n_articles"], ascending=False
    )

    top = (
        article_persons.groupby(["person_id", "person_name"])
        .agg(
            n_articles=("article_id", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            raw_variants=("person_raw", lambda x: " | ".join(sorted(set(map(compact_ws, x))))),
        )
        .reset_index()
        .sort_values("n_articles", ascending=False)
        .head(50)
    )
    top["manual_check_status"] = ""
    top["notes"] = ""
    return variants, possible_duplicates, top


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_markdown(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def shannon_entropy(values: pd.Series) -> float:
    counts = values.value_counts()
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(-(probs * np.log(probs)).sum())


def normalized_entropy(values: pd.Series, n_categories: int | None = None) -> float:
    entropy = shannon_entropy(values)
    k = n_categories or values.nunique()
    if k <= 1:
        return 0.0
    return float(entropy / np.log(k))


def cosine_centroid_dispersion(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    centroid = matrix.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * centroid_norm
    denom[denom == 0] = np.nan
    sims = matrix.dot(centroid) / denom
    return float(np.nanmean(1 - sims))


def infer_macro_category(label_or_text: object) -> str:
    text = compact_ws(label_or_text).lower()
    scores = {
        category: sum(1 for keyword in keywords if keyword in text)
        for category, keywords in MACRO_CATEGORY_KEYWORDS.items()
    }
    best_category, best_score = max(scores.items(), key=lambda item: item[1])
    return best_category if best_score > 0 else "other_or_residual"


def top_tfidf_terms(texts: Iterable[str], n_terms: int = 12) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from sklearn.feature_extraction.text import TfidfVectorizer

    cleaned = [compact_ws(text) for text in texts if compact_ws(text)]
    if not cleaned:
        return []
    extra_stopwords = {
        "aber",
        "about",
        "after",
        "all",
        "also",
        "als",
        "and",
        "are",
        "auf",
        "aus",
        "bei",
        "between",
        "bis",
        "can",
        "das",
        "dem",
        "den",
        "der",
        "des",
        "die",
        "dieser",
        "dieses",
        "durch",
        "ein",
        "eine",
        "einem",
        "einen",
        "einer",
        "eines",
        "for",
        "from",
        "fur",
        "gesis",
        "has",
        "have",
        "historical",
        "hsr",
        "https",
        "ist",
        "mit",
        "nach",
        "not",
        "oder",
        "research",
        "sind",
        "social",
        "ssoar",
        "that",
        "the",
        "this",
        "und",
        "von",
        "was",
        "werden",
        "were",
        "www",
        "zur",
    }
    stopwords = sorted(set(ENGLISH_STOP_WORDS).union(extra_stopwords))
    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.75,
        token_pattern=r"(?u)\b[^\W\d_][\w-]{2,}\b",
    )
    matrix = vectorizer.fit_transform(cleaned)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    order = scores.argsort()[::-1][:n_terms]
    return terms[order].tolist()


def save_caption(root: Path, figure_name: str, caption: str) -> Path:
    path = root / "outputs/figures/figure_captions.csv"
    row = pd.DataFrame([{"figure": figure_name, "caption": caption}])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["figure"] != figure_name]
        row = pd.concat([existing, row], ignore_index=True)
    return write_csv(row, path)
