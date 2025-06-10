import numpy as np
from neural_net import get_net_scores, train_net, prepare_loader, SimpleDenseNet
from pep_utils import load_peptide_data, get_peptide_distances, pep_train_test_split
from plot import plot_boxplot, plot_roc_curve, plot_esm_amino_acids_embeddings, plot_2dim_reduction
from esm_embeddings import get_esm_embeddings, get_esm_model
from clustering import kmeans_clustering, tsne_dim_reduction
from itertools import product
import numpy as np
from sklearn.metrics import roc_auc_score



def simple_score(p_train, n_train, p_test, n_test):
    """
    Distance-based baseline.
    For every peptide in test we compute its mean Euclidean distance
    to all positive-training embeddings and all negative-training
    embeddings.  The final score is a signed log-fold-difference:

        score  =  log1p(dist_to_neg) – log1p(dist_to_pos)

    So a higher score -> more “positive-like”, because
    distance-to-negatives is large while distance-to-positives is small.

    :param p_train: Positive-class train embeddings.
    :param n_train: Negative-class train embeddings.
    :param p_test: Positive-class test embeddings.
    :param n_test: Negative-class test embeddings.
    :return: Scores for the positive-test set, Scores for the negative-test set (np.ndarray, np.ndarray).
    """
    positive_mean_distances_pos = get_peptide_distances(p_test, p_train, reduce_func=np.mean)
    negative_mean_distances_pos = get_peptide_distances(n_test, p_train, reduce_func=np.mean)
    positive_mean_distances_neg = get_peptide_distances(p_test, n_train, reduce_func=np.mean)
    negative_mean_distances_neg = get_peptide_distances(n_test, n_train, reduce_func=np.mean)

    p_score = np.log1p(positive_mean_distances_neg) - np.log1p(positive_mean_distances_pos)
    n_score = np.log1p(negative_mean_distances_neg) - np.log1p(negative_mean_distances_pos)
    return p_score, n_score




if __name__ == '__main__':
    # Fixed parameters
    chosen_embedding_size = 1280
    chosen_embedding_layer = 33
    chosen_test_size = 0.25

    print("🔹 Loading peptide data")
    positive_pep, negative_pep, doubt_lables = load_peptide_data()

    print("🔹 Loading ESM-2 model")
    model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=chosen_embedding_size)

    print("🔹 Getting ESM-2 amino acid embeddings for heatmap")
    positive_aa_embeddings = get_esm_embeddings(positive_pep[0:1], model_esm, alphabet_esm, batch_converter_esm, device_esm,
                                                embedding_layer=chosen_embedding_layer, sequence_embedding=False)[0]
    negative_aa_embeddings = get_esm_embeddings(negative_pep[0:1], model_esm, alphabet_esm, batch_converter_esm, device_esm,
                                                embedding_layer=chosen_embedding_layer, sequence_embedding=False)[0]

    print("🔹 Plotting amino acid embedding heatmaps")
    plot_esm_amino_acids_embeddings(positive_aa_embeddings, out_file_path="positive_heatmap.png")
    plot_esm_amino_acids_embeddings(negative_aa_embeddings, out_file_path="negative_heatmap.png")

    print("🔹 Getting sequence-level ESM-2 embeddings")
    positive_esm_emb = get_esm_embeddings(positive_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm,
                                          embedding_layer=chosen_embedding_layer, sequence_embedding=True)
    negative_esm_emb = get_esm_embeddings(negative_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm,
                                          embedding_layer=chosen_embedding_layer, sequence_embedding=True)

    print("🔹 Clustering and TSNE")
    all_esm_embeddings = negative_esm_emb + positive_esm_emb
    all_labels = [0] * len(negative_esm_emb) + [1] * len(positive_esm_emb)

    k_means_labels = kmeans_clustering(all_esm_embeddings, k=2)
    coords_2d = tsne_dim_reduction(all_esm_embeddings, dim=2)

    plot_2dim_reduction(coords_2d, [["N", "P"][i] for i in all_labels], out_file_path="2d_true_labels.png")
    plot_2dim_reduction(coords_2d, k_means_labels, out_file_path="2d_k_means.png")

    print("🔹 Splitting into train and test sets")
    positive_train, negative_train, is_doubt_train, positive_test, negative_test, is_doubt_test = pep_train_test_split(
        positive_esm_emb, negative_esm_emb, doubt_lables, test_size=chosen_test_size)

    print("🔹 Calculating baseline Euclidean distance score")
    positive_score, negative_score = simple_score(positive_train, negative_train, positive_test, negative_test)
    plot_boxplot({"Positive Test": positive_score, "Negative Test": negative_score}, out_file_path="baseline_boxplot.png")
    plot_roc_curve([0] * len(negative_score) + [1] * len(positive_score),
                   np.concatenate([negative_score, positive_score]), out_file_path="baseline_roc_curve.png")

    # 🔍 Grid Search
    print("🔍 Starting grid search over neural network hyperparameters")

    batch_sizes = [32, 64]
    epochs_list = [30, 50]
    lrs = [1e-3, 1e-4]
    hidden_dims = [64, 128]
    dropouts = [0.2, 0.4]

    best_auc = 0
    best_config = None

    for batch_size, epochs, lr, hidden_dim, dropout in product(batch_sizes, epochs_list, lrs, hidden_dims, dropouts):
        print(f"\n🧪 Trying configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, hidden_dim={hidden_dim}, dropout={dropout}")

        net_dataloader = prepare_loader(positive_train, negative_train, batch_size=batch_size)
        network = SimpleDenseNet(esm_emb_dim=chosen_embedding_size, hidden_dim=hidden_dim, dropout=dropout)
        trained_network = train_net(network, net_dataloader, num_epochs=epochs, lr=lr)

        positive_score = get_net_scores(trained_net=trained_network, esm_seq_embeddings=positive_test)
        negative_score = get_net_scores(trained_net=trained_network, esm_seq_embeddings=negative_test)

        y_true = [0] * len(negative_score) + [1] * len(positive_score)
        y_pred = np.concatenate([negative_score, positive_score])
        auc = roc_auc_score(y_true, y_pred)
        print(f"📈 AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_config = {
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "hidden_dim": hidden_dim,
                "dropout": dropout
            }

        plot_boxplot({"Positive Test": positive_score, "Negative Test": negative_score},
                     out_file_path=f"grid_boxplot_bs{batch_size}_hd{hidden_dim}_dp{dropout}.png")
        plot_roc_curve(y_true, y_pred, out_file_path=f"grid_roc_bs{batch_size}_hd{hidden_dim}_dp{dropout}.png")

    print("\n✅ Grid Search Complete")
    print(f"🏆 Best AUC: {best_auc:.4f} with configuration: {best_config}")
