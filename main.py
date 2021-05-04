 
import os.path as os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges,to_dense_adj,add_self_loops
from model import Encoder_VGAE,Encoder_GAE
from utils import get_roc_score
import statistics


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGAE', help='GAE, VGAE')
parser.add_argument('--dev', type=str, default='cuda', help='Use cuda or cpu')
parser.add_argument('--dataset', type=str, default='Cora',help='Cora, CiteSeer, Pubmed')
parser.add_argument('--runs', type=int, default=10, help='Number of runs')
parser.add_argument('--depth', type=int, default=1, help='Number of layers')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--act', type=str, default="sigmoid", help='activation function')

parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--res', type=str, default="True", help='Residual connection')
args = parser.parse_args()

#download datasets from pytorch-geometric
path = os.join(os.dirname(os.realpath(__file__)), '..', 'data', args.dataset)
dataset = Planetoid(path, args.dataset)

dev = torch.device(args.dev)



        
if args.model== VGAE:
    model = VGAE(Encoder_VGAE(dataset.num_features, args.hidden1, args.hidden2, args.depth,args.res)).to(dev)
else:
    model = GAE(Encoder_GAE(dataset.num_features, args.hidden1, args.hidden2, args.depth,args.res)).to(dev)

auc_score_list= []
ap_score_list = []

print("Dataset: ",args.dataset," Model: ", args.model, ", Residual :",args.res,", Layer depth:", args.depth," ")


for i in range(1, args.runs +1):
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)

    x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    adj_train = train_pos_edge_index
    adj_train_dense = to_dense_adj(adj_train)[0]
    adj_train_dense=adj_train_dense

    norm = adj_train_dense.shape[0] * adj_train_dense.shape[0] / float((adj_train_dense.shape[0] * adj_train_dense.shape[0] - adj_train_dense.sum()) * 2)


    z_final= None
    for epoch in range(1, 200):

        model.train()
        optimizer.zero_grad()

        z = model.encode(x, train_pos_edge_index)

        # loss from binary cross entropy
        loss = norm* model.recon_loss(z, train_pos_edge_index)
        
        if args.model == VGAE:
            loss = loss + (model.kl_loss()/ data.num_nodes) 

        loss.backward()
        optimizer.step()

        #last embedding
        z_final=z



        auc, ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)

    model.eval()
    with torch.no_grad():
        auc_score, ap_score = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
        auc_score_list.append(auc_score*100)
        ap_score_list.append(ap_score*100)

       

print('AUC score: {:.2f} +/-{:.2f}'.format(statistics.mean(auc_score_list), statistics.stdev(auc_score_list)))
print('AP score: {:.2f} +/- {:.2f}'.format(statistics.mean(ap_score_list), statistics.stdev(ap_score_list)))
