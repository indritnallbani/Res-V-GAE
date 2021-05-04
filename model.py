
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

#Encoder
class Encoder_VGAE(torch.nn.Module):
	def __init__(self, in_channels, hidden1, hidden2, depth,res):
		super(Encoder_VGAE, self).__init__()

		self.conv1 = GCNConv(in_channels,hidden1)
		self.residual = Residual(hidden1, hidden1)
		self.depth = depth
		self.residual=res

		self.conv_mu = GCNConv(hidden1, hidden2)
		self.conv_logstd = GCNConv(hidden1, hidden2)

	def forward(self, x, edge_index):
		#Input layer
		x = F.relu(self.conv1(x, edge_index))

		#Hidden layers according to the desired depth
		for i in range(1, self.depth + 1):
			x = self.residual(x,edge_index)

		self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)



class Encoder_GAE(torch.nn.Module):
	def __init__(self, in_channels, hidden1, hidden2, depth,res):
		super(Encoder_GAE, self).__init__()

		#first layer
		self.conv1 = GCNConv(in_channels,hidden1)
		self.layers = Residual(hidden1, hidden1,res)
		self.depth = depth
		self.residual=res

		#lastlayer
		self.convx = GCNConv(hidden1, hidden2)
		
	def forward(self, x, edge_index):
		x = F.relu(self.conv1(x, edge_index))

		for i in range(1, self.depth + 1):

			x = self.layers(x,edge_index)

		return self.convx(x, edge_index)


class Residual(torch.nn.Module):
	def __init__(self,hidden1, hidden2,res):
		super(Residual,self).__init__()

		self.conv = GCNConv(hidden1,hidden1)
		self.res=res

	def forward(self, x, edge_index):

		output = F.relu(self.conv(x, edge_index))

		if self.res == 'True':
			return output + x
		else:
			return output



