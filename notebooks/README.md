# HSR Analysis Notebooks

Run the notebooks in this order:

1. `00_project_setup_and_corpus_audit.ipynb`
2. `01_people_names_and_roles.ipynb`
3. `02_embeddings_qwen_and_diagnostics.ipynb`
4. `03_semantic_mapping_clustering_stability.ipynb`
5. `04_topic_labeling_and_time_dynamics.ipynb`
6. `05_issues_and_editorial_positioning.ipynb`
7. `06_actor_profiles_networks_typology.ipynb`
8. `07_article_type_and_method_labels.ipynb`
9. `08_citations_internationalization_and_summary.ipynb`

The default embedding model is `Qwen/Qwen3-Embedding-0.6B`. Embeddings and all intermediate tables are cached under `outputs/`. Manual review files live in `data/` and `outputs/labels/`.

