**Res(V)GAE: Going Deeper with Residual Connections**

 In this paper, we study the effects
of adding residual connections to graph autoencoders with multiple graph convolutional layers and propose Residual (Variational) Graph Autoencoder (Res(V)GAE), a deep (variational) graph autoencoder model with multiple residual connections. We show that residual connections improve the average precision of the graph autoencoders when we increase the number of graph convolutional layers. Experimental results suggest that our proposed model with residual connections outperforms the models without residual connections for the link prediction task.

Finally, we have contributed twofold: firstly we study the effectiveness of adding residual connections to deep graph models and we introduce our own deep learning model called Res-VGAE. We have reported the results of models from one to eight residual connections on the link prediction task. The results show that  we are able to increase the accuracy results using AP and AUC metrics when compared with other similar models. The dataset used are Cora, Citeseer, and Pubmed.


**Model Architecture**

Model architecture of the GAE, ResGAE, VGAE, and ResVGAE. Residual connections start after the first Hidden Layer (HL) since the input and the
output size of layers with residual connections must be the same. The encoder takes the adjacency matrix A and the feature matrix X as inputs and outputs
the node embeddings Z. The decoder takes as input the embedding matrix Z and outputs the reconstructed adjacency matrix Aˆ. The blocks in blue indicate
the graph convolutional layers that embed 32-dimensional node feature vectors into a matrix. Similarly, yellow blocks constitute the graph convolutional
layers that embed 16-dimensional hidden layer features into the output Z matrix. The upper and lower branches of the encoder represent variational graph
autoencoder and graph autoencoder architectures, respectively.

![res](https://user-images.githubusercontent.com/34435436/154451968-fe9a7345-e3c2-4d6f-8297-fdabf7625758.png)

**Results**

We use AP and AUC scores to report the average precision of 10 runs with random train, validation, and test splits of the same size, and all the models are trained for 200 epochs. The validation and test sets contain 5\% and 10\% of the total edges, respectively.
The embedding dimensions of the node features for the hidden layers is 32 and the embedding dimension of node features for the output layer is 16. The models are optimized using Adam optimizer with a learning rate of 0.01.

Experiments indicate that, for shallow models, all proposed models achieve similar average precision scores. Here we see that all models with one layer embed the node features in a very similar way. The score differs when we use models with deeper networks. In models with eight layers, we see that models with residual connections have higher average precision scores than models without residual connections.

![Screenshot from 2022-02-17 13-08-53](https://user-images.githubusercontent.com/34435436/154453727-c44014ab-7928-4c1a-a719-84e8546c7a28.png)
