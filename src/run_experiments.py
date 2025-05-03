# src/run_experiments.py
from ordinal_probe import DimensionalityReducer, OrdinalCluster, OrdinalEvaluator
from utils import load_ordinal_terms, load_embedding_model, get_embeddings_for_terms
import numpy as np
import pandas as pd

DATA_PATH = '/content/drive/MyDrive/ordered_terms/data/ordinal_terms.csv'

EMBEDDING_MODELS = [
    "Word2Vec", "GloVe", "fastText", "BPEmb",
    "BERT", "RoBERTa", "GPT-2"
]

REDUCTION_METHOD = 'PCA'
CLUSTER_LINKAGES = ['single', 'complete', 'average']
CLUSTER_DIVISORS = [2, 3, 4]

def main():
    ordinal_terms = load_ordinal_terms(DATA_PATH)
    evaluator = OrdinalEvaluator()
    results_list = []

    for model_name in EMBEDDING_MODELS:
        print(f"\nEvaluating embeddings from model: {model_name}")
        model = load_embedding_model(model_name)
        
        for category, terms in ordinal_terms.items():
            print(f"  Processing category: {category}")
            embeddings = get_embeddings_for_terms(model, terms, model_name)
            if embeddings.size == 0:
                print(f"    Skipping '{category}' as no embeddings were loaded.")
                continue

            n_terms = embeddings.shape[0]
            true_order = list(range(n_terms))

            reducer = DimensionalityReducer(n_components=2)
            reduced_embeddings = reducer.fit_transform(embeddings)
            predicted_order = list(np.argsort(reduced_embeddings[:, 0]))

            for linkage in CLUSTER_LINKAGES:
                for divisor in CLUSTER_DIVISORS:
                    n_clusters = max(2, n_terms // divisor)
                    if n_clusters >= n_terms:
                        continue  # Skip trivial singleton clustering
                    
                    cluster_model = OrdinalCluster(linkage=linkage, n_clusters=n_clusters)
                    clusters = cluster_model.fit_predict(reduced_embeddings)
                    actual_clusters = len(np.unique(clusters))

                    rho = evaluator.spearman_rank(true_order, predicted_order)
                    mae = evaluator.mean_abs_error(true_order, predicted_order)
                    cem = evaluator.cem(true_order, predicted_order, num_classes=n_terms)

                    # build true class labels by grouping ranks into bins of size=divisor
                    true_classes = [i // divisor for i in true_order]

                    # now compute purity against those classes
                    purity = evaluator.purity(clusters, true_classes)
                    results_list.append({
                        "Embedding Model": model_name,
                        "Category": category,
                        "Reduction Method": REDUCTION_METHOD,
                        "Cluster Linkage": linkage,
                        "Cluster Divisor": divisor,
                        "NumTerms": n_terms,
                        "NumClusters": actual_clusters,
                        "Spearman": rho,
                        "MAE": mae,
                        "CEM": cem,
                        "Purity": purity
                    })

    results_df = pd.DataFrame(results_list)
    results_df.to_csv("experiment_results_cluster_sweep.csv", index=False)
    print("\nResults saved to 'experiment_results_cluster_sweep.csv'.")
    print(results_df.head())

if __name__ == "__main__":
    main()
