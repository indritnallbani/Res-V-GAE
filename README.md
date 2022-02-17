**Res(V)GAE: Going Deeper with Residual Connections**

**Model Architecture**

Model architecture of the GAE, ResGAE, VGAE, and ResVGAE. Residual connections start after the first Hidden Layer (HL) since the input and the
output size of layers with residual connections must be the same. The encoder takes the adjacency matrix A and the feature matrix X as inputs and outputs
the node embeddings Z. The decoder takes as input the embedding matrix Z and outputs the reconstructed adjacency matrix Aˆ. The blocks in blue indicate
the graph convolutional layers that embed 32-dimensional node feature vectors into a matrix. Similarly, yellow blocks constitute the graph convolutional
layers that embed 16-dimensional hidden layer features into the output Z matrix. The upper and lower branches of the encoder represent variational graph
autoencoder and graph autoencoder architectures, respectively.

![res](https://user-images.githubusercontent.com/34435436/154451968-fe9a7345-e3c2-4d6f-8297-fdabf7625758.png)

**Results**

![Screenshot from 2022-02-17 13-08-53](https://user-images.githubusercontent.com/34435436/154453727-c44014ab-7928-4c1a-a719-84e8546c7a28.png)
