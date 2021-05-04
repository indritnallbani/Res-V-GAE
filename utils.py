from torch_geometric.utils import to_dense_adj
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def get_roc_score(z, x, edges_pos, edges_neg):
   

    z=z.cpu()

    adj_orig = to_dense_adj(x)[0]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(z, z.T)

    preds = []
    pos = []
    for e in edges_pos.T:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg.T:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score
