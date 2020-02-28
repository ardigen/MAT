"""
Code based on:
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
"""

import numpy as np
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.Descriptors as Descriptors
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import rdkit.Chem.rdChemReactions as rdRxns
import copy

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import pairwise_distances

att_dtype = np.float32

class Graph():
	'''Describes an undirected graph class'''
	def __init__(self):
		self.nodes = []
		self.num_nodes = 0
		self.edges = []
		self.num_edges = 0
		self.N_features = 0
		self.bondtype_list_order = []
		self.atomtype_list_order = []
		return

	def nodeAttributes(self):
		'''Returns 2D array where (#, :) contains attributes of node #'''
		return (np.vstack([x.attributes for x in self.nodes]))

	def edgeAttributes(self):
		'''Returns 2D array where (#, :) contains attributes of edge #'''
		return (np.vstack([x.attributes for x in self.edges]))

	def edgeAttributesAtt(self):
		return (np.vstack([x.attributesAtt for x in self.edges]))

	def nodeNeighbors(self):
		return [x.neighbors for x in self.nodes]

	def clone(self):
		'''clone() method to trick Theano'''
		return copy.deepcopy(self)

	# Add the AdjTensor with edge info
	def getAdjTensor(self, maxNodes):
		adjTensor = np.zeros([maxNodes, maxNodes, self.edgeFeatureDim + 1])
		for edge in self.edges:
			(i, j) = edge.ends
			adjTensor[i, j, 0] = 1.0
			adjTensor[j, i, 0] = 1.0
			adjTensor[i, j, 1:] = edge.features
			adjTensor[j, i, 1:] = edge.features
		return adjTensor

	def dump_as_tensor(self):
		'''Method to represent attributed graph as a giant tensor

		The tensor is N_node x N_node x N_attributes.

		For a given node, A_i,i,: is a vector of that node's features, followed
		  by zeros where edge attributes would be.
		For a pair of nodes i and j, A_i,j,: is a vector of node j's features,
		  followed by the edge attributes connecting the two nodes.

		This representation is not as efficient as it could be, but we need to
		  pack the whole graph information into a single tensor in order to use
		  Keras/Theano easily'''

		# Bad input handling
		if not self.nodes:
			raise(ValueError, 'Error generating tensor for graph with no nodes')
		if not self.edges:
			raise(ValueError, 'Need at least one bond!')

		N_nodes = len(self.nodes)
		N_features = sizeAttributeVector(molecular_attributes = self.molecular_attributes, formal_charge_one_hot=self.formal_charge_one_hot)
		tensor = np.zeros((N_nodes, N_nodes, N_features))

		# Special case of no bonds (e.g., methane)
		if not self.edges:
			nodeAttributes = np.vstack([x.attributes for x in self.nodes])
			for i, node in enumerate(self.nodes):
				tensor[i, i, 0:len(nodeAttributes[i])] = nodeAttributes[i]
			return tensor

		edgeAttributes = np.vstack([x.attributes for x in self.edges])
		nodeAttributes = np.vstack([x.attributes for x in self.nodes])
		nodeNeighbors = self.nodeNeighbors()
		# Assign diagonal entries
		for i, node in enumerate(self.nodes):
			tensor[i, i, :] = np.concatenate((nodeAttributes[i], np.zeros_like(edgeAttributes[0])))
		# Assign bonds now
		for e, edge in enumerate(self.edges):
			(i, j) = edge.connects
			tensor[i, j, :] = np.concatenate((nodeAttributes[j], edgeAttributes[e]))
			tensor[j, i, :] = np.concatenate((nodeAttributes[i], edgeAttributes[e]))

		return tensor

	def dump_as_matrices(self):
		# Bad input handling
		if not self.nodes:
			raise(ValueError, 'Error generating tensor for graph with no nodes')

		N_nodes = len(self.nodes)
		F_a, F_b = sizeAttributeVectorsAtt(self.bondtype_list_order, molecular_attributes = self.molecular_attributes, formal_charge_one_hot=self.formal_charge_one_hot)

		mat_features = np.zeros((N_nodes, F_a), dtype = np.float32)
		mat_adjacency = np.zeros((N_nodes, N_nodes), dtype = np.float32)
		mat_specialbondtypes = np.zeros((N_nodes, F_b), dtype = np.float32)
		adjTensor = np.zeros([F_b, N_nodes, N_nodes], dtype = np.float32)

		if self.edges:
			edgeAttributes = np.vstack([x.attributes for x in self.edges])
		else:
			edgeAttributes = []
		nodeAttributes = np.vstack([x.attributes for x in self.nodes])


		for i, node in enumerate(self.nodes):
			mat_features[i, :] = nodeAttributes[i]
			mat_adjacency[i, i] = 1.0 # include self terms

		for e, edge in enumerate(self.edges):
			(i, j) = edge.connects
			mat_adjacency[i, j] = 1.0
			mat_adjacency[j, i] = 1.0

			# Keep track of extra special bond types - which are nothing more than
			# bias terms specific to the bond type because they are all one-hot encoded
			mat_specialbondtypes[i, :] += edgeAttributes[e]
			mat_specialbondtypes[j, :] += edgeAttributes[e]

		for edge in self.edges:
			(i, j) = edge.connects
			adjTensor[0:, i, j] = edge.attributes
			adjTensor[0:, j, i] = edge.attributes

		return (mat_features, mat_adjacency, adjTensor) # replace mat_specialbondtypes bt adjTensor

	def dump_as_matrices_Att(self):
		# Bad input handling
		if not self.nodes:
			raise(ValueError, 'Error generating tensor for graph with no nodes')

		N_nodes = len(self.nodes)
		F_a, F_b = sizeAttributeVectors(self.bondtype_list_order,
										self.atomtype_list_order, molecular_attributes = self.molecular_attributes, formal_charge_one_hot=self.formal_charge_one_hot)
		F_a, F_bAtt = sizeAttributeVectorsAtt(self.bondtype_list_order, self.atomtype_list_order,
											  molecular_attributes=self.molecular_attributes, formal_charge_one_hot=self.formal_charge_one_hot)

		mat_features = np.zeros((N_nodes, F_a), dtype = np.float32)
		mat_adjacency = np.zeros((N_nodes, N_nodes), dtype = np.float32)
		mat_positions = np.zeros((N_nodes, 3), dtype = np.float32)
		mat_specialbondtypes = np.zeros((N_nodes, F_b), dtype = np.float32)
		adjTensor = np.zeros([F_b, N_nodes, N_nodes], dtype = np.float32)
		adjTensorAtt = np.zeros([F_bAtt, N_nodes, N_nodes], dtype = np.float32)

		adjTensor_OrderAtt = np.zeros([5, N_nodes, N_nodes], dtype = np.float32)
		adjTensor_AromAtt = np.zeros([3, N_nodes, N_nodes], dtype = np.float32)
		adjTensor_ConjAtt = np.zeros([3, N_nodes, N_nodes], dtype = np.float32)
		adjTensor_RingAtt = np.zeros([3, N_nodes, N_nodes], dtype=np.float32)

		if self.edges:
			edgeAttributes = np.vstack([x.attributes for x in self.edges])
		else:
			edgeAttributes = []
		nodeAttributes = np.vstack([x.attributes for x in self.nodes])

		for i, node in enumerate(self.nodes):
			mat_features[i, :] = nodeAttributes[i]
			mat_adjacency[i, i] = 1.0 # include self terms
			mat_positions[i, :] = node.pos
			adjTensorAtt[0:len(self.atomtype_list_order), i, i] = node.attributesAtt
			adjTensor_OrderAtt[0, i, i] = 1
			adjTensor_AromAtt[0,i,i] = 1
			adjTensor_ConjAtt[0, i, i] = 1
			adjTensor_RingAtt[0, i, i] = 1
		mat_distances = pairwise_distances(mat_positions)

		for e, edge in enumerate(self.edges):
			(i, j) = edge.connects
			mat_adjacency[i, j] = 1.0
			mat_adjacency[j, i] = 1.0

			# Keep track of extra special bond types - which are nothing more than
			# bias terms specific to the bond type because they are all one-hot encoded
			mat_specialbondtypes[i, :] += edgeAttributes[e]
			mat_specialbondtypes[j, :] += edgeAttributes[e]

		for edge in self.edges:
			(i, j) = edge.connects
			adjTensorAtt[len(self.atomtype_list_order):, i, j] = edge.attributesAtt
			adjTensorAtt[len(self.atomtype_list_order):, j, i] = edge.attributesAtt


			adjTensor_OrderAtt[1:, i, j] = edge.orderAtt
			adjTensor_OrderAtt[1:, j, i] = edge.orderAtt
			adjTensor_AromAtt[1:, i, j] = edge.aromAtt
			adjTensor_AromAtt[1:, j, i] = edge.aromAtt
			adjTensor_ConjAtt[1:, i, j] = edge.conjAtt
			adjTensor_ConjAtt[1:, j, i] = edge.conjAtt
			adjTensor_RingAtt[1:, i, j] = edge.ringAtt
			adjTensor_RingAtt[1:, j, i] = edge.ringAtt

		return (mat_features, mat_adjacency, adjTensorAtt, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt, mat_distances)


	def dump_as_matrices_Att_dummyNode(self):
		# Bad input handling
		if not self.nodes:
			raise(ValueError, 'Error generating tensor for graph with no nodes')

		N_nodes = len(self.nodes) + 1
		F_a, F_b = sizeAttributeVectors(self.bondtype_list_order,
										self.atomtype_list_order, molecular_attributes = self.molecular_attributes, formal_charge_one_hot=self.formal_charge_one_hot)
		F_a, F_bAtt = sizeAttributeVectorsAtt(self.bondtype_list_order, self.atomtype_list_order,
											  molecular_attributes=self.molecular_attributes, formal_charge_one_hot=self.formal_charge_one_hot)

		mat_features = np.zeros((N_nodes, F_a+1), dtype = np.float32)
		mat_adjacency = np.zeros((N_nodes, N_nodes), dtype = np.float32)
		mat_positions = np.zeros((N_nodes, 3), dtype = np.float32)
		mat_specialbondtypes = np.zeros((N_nodes, F_b), dtype = np.float32)
		adjTensor = np.zeros([F_b, N_nodes, N_nodes], dtype = np.float32)
		adjTensorAtt = np.zeros([F_bAtt, N_nodes, N_nodes], dtype = np.float32)

		adjTensor_OrderAtt = np.zeros([5, N_nodes, N_nodes], dtype = np.float32)
		adjTensor_AromAtt = np.zeros([3, N_nodes, N_nodes], dtype = np.float32)
		adjTensor_ConjAtt = np.zeros([3, N_nodes, N_nodes], dtype = np.float32)
		adjTensor_RingAtt = np.zeros([3, N_nodes, N_nodes], dtype=np.float32)

		if self.edges:
			edgeAttributes = np.vstack([x.attributes for x in self.edges])
		else:
			edgeAttributes = []
		nodeAttributes = np.vstack([x.attributes for x in self.nodes])

		for i, node in enumerate(self.nodes):
			i = i+1
			mat_features[i, 1:] = nodeAttributes[i-1]
			mat_adjacency[i, i] = 1.0 # include self terms
			mat_positions[i, :] = node.pos
			adjTensorAtt[0:len(self.atomtype_list_order), i, i] = node.attributesAtt
			adjTensor_OrderAtt[0, i, i] = 1
			adjTensor_AromAtt[0,i,i] = 1
			adjTensor_ConjAtt[0, i, i] = 1
			adjTensor_RingAtt[0, i, i] = 1
		mat_distances = pairwise_distances(mat_positions)

		# ADD DUMMY NODE
		mat_features[0,0] = 1
		mat_distances[0,:] = 1e6
		mat_distances[:,0] = 1e6

		for e, edge in enumerate(self.edges):
			(i, j) = edge.connects
			i, j = i+1, j+1
			mat_adjacency[i, j] = 1.0
			mat_adjacency[j, i] = 1.0

			# Keep track of extra special bond types - which are nothing more than
			# bias terms specific to the bond type because they are all one-hot encoded
			mat_specialbondtypes[i, :] += edgeAttributes[e]
			mat_specialbondtypes[j, :] += edgeAttributes[e]

		for edge in self.edges:
			(i, j) = edge.connects
			i, j = i+1, j+1
			adjTensorAtt[len(self.atomtype_list_order):, i, j] = edge.attributesAtt
			adjTensorAtt[len(self.atomtype_list_order):, j, i] = edge.attributesAtt


			adjTensor_OrderAtt[1:, i, j] = edge.orderAtt
			adjTensor_OrderAtt[1:, j, i] = edge.orderAtt
			adjTensor_AromAtt[1:, i, j] = edge.aromAtt
			adjTensor_AromAtt[1:, j, i] = edge.aromAtt
			adjTensor_ConjAtt[1:, i, j] = edge.conjAtt
			adjTensor_ConjAtt[1:, j, i] = edge.conjAtt
			adjTensor_RingAtt[1:, i, j] = edge.ringAtt
			adjTensor_RingAtt[1:, j, i] = edge.ringAtt

		return (mat_features, mat_adjacency, adjTensorAtt, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt, mat_distances)

class Node():
	'''Describes an attributed node in an undirected graph'''
	def __init__(self, i = None, attributes = np.array([], dtype = att_dtype),
				 attributesAtt = np.array([], dtype = att_dtype), pos=None):
		self.i = i
		self.attributes = attributes # 1D array
		self.neighbors = [] # (atom index, bond index)
		self.attributesAtt = attributesAtt
		self.pos = pos
		return

class Edge():
	'''Describes an attributed edge in an undirected graph'''
	def __init__(self, connects = (), i = None, attributes = np.array([], dtype = att_dtype),
				 attributesAtt = np.array([], dtype = att_dtype), orderAtt = np.array([], dtype = att_dtype),
				 aromAtt=np.array([], dtype=att_dtype), conjAtt = np.array([], dtype = att_dtype),
				 ringAtt = np.array([], dtype = att_dtype)):
		self.i = i
		self.attributes = attributes # 1D array
		self.attributesAtt = attributesAtt
		self.connects = connects # (atom index, atom index)
		self.orderAtt = orderAtt
		self.aromAtt = aromAtt
		self.conjAtt = conjAtt
		self.ringAtt = ringAtt
		return

def molToGraph(rdmol, bondtype_list_order, atomtype_list_order, molecular_attributes = False, formal_charge_one_hot=False):
	'''Converts an RDKit molecule to an attributed undirected graph'''
	# Initialize
	graph = Graph()
	graph.molecular_attributes = molecular_attributes
	graph.bondtype_list_order = bondtype_list_order
	bond_list = bondtype_list_order
	graph.atomtype_list_order = atomtype_list_order
	graph.formal_charge_one_hot = formal_charge_one_hot

	# Calculate atom-level molecule descriptors
	attributes = [[] for i in rdmol.GetAtoms()]
	if molecular_attributes:
		labels = []
		[attributes[i].append(x[0]) \
			for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
		labels.append('Crippen contribution to logp')

		[attributes[i].append(x[1]) \
			for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
		labels.append('Crippen contribution to mr')

		[attributes[i].append(x) \
			for (i, x) in enumerate(rdMolDescriptors._CalcTPSAContribs(rdmol))]
		labels.append('TPSA contribution')

		[attributes[i].append(x) \
			for (i, x) in enumerate(rdMolDescriptors._CalcLabuteASAContribs(rdmol)[0])]
		labels.append('Labute ASA contribution')

		[attributes[i].append(x) \
			for (i, x) in enumerate(EState.EStateIndices(rdmol))]
		labels.append('EState Index')

		rdPartialCharges.ComputeGasteigerCharges(rdmol)
		[attributes[i].append(float(a.GetProp('_GasteigerCharge'))) \
			for (i, a) in enumerate(rdmol.GetAtoms())]
		labels.append('Gasteiger partial charge')

		# Gasteiger partial charges sometimes gives NaN
		for i in range(len(attributes)):
			if np.isnan(attributes[i][-1]) or np.isinf(attributes[i][-1]):
				attributes[i][-1] = 0.0

		[attributes[i].append(float(a.GetProp('_GasteigerHCharge'))) \
			for (i, a) in enumerate(rdmol.GetAtoms())]
		labels.append('Gasteiger hydrogen partial charge')

		# Gasteiger partial charges sometimes gives NaN
		for i in range(len(attributes)):
			if np.isnan(attributes[i][-1]) or np.isinf(attributes[i][-1]):
				attributes[i][-1] = 0.0

	# Add bonds
	for bond in rdmol.GetBonds():
		edge = Edge()
		edge.i = bond.GetIdx()
		edge.attributes = bondAttributes(bond)
		edge.orderAtt = list(oneHotVector(bond.GetBondTypeAsDouble(), [1.0, 1.5, 2.0, 3.0]))
		edge.aromAtt = list(oneHotVector(bond.GetIsAromatic(), [1.0, 0.0]))
		edge.conjAtt = list(oneHotVector(bond.GetIsConjugated(), [1.0, 0.0]))
		edge.ringAtt = list(oneHotVector(bond.IsInRing(), [1.0, 0.0]))

		BeginAtom, EndAtom = bond.GetBeginAtom(), bond.GetEndAtom()
		begin_idx, end_idx = BeginAtom.GetAtomicNum(), EndAtom.GetAtomicNum()
		if begin_idx < end_idx:
			bond_type = str(begin_idx) + '_' + str(end_idx)
		else:
			bond_type= str(end_idx) + '_' + str(begin_idx)

		bond_attributes = []
		bond_attributes = bond_attributes + list(oneHotVector(bond_type, bondtype_list_order))
		edge.attributesAtt = np.array(bond_attributes, dtype=att_dtype)

		edge.connects = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
		graph.edges.append(edge)

	# Add atoms
	conf = rdmol.GetConformer() # needed to calculate positions
	for k, atom in enumerate(rdmol.GetAtoms()):
		node = Node()
		node.i = atom.GetIdx()
		node.attributes = atomAttributes(atom, extra_attributes = attributes[k], formal_charge_one_hot=formal_charge_one_hot)
		node_type = atom.GetAtomicNum()
		pos = conf.GetAtomPosition(k)
		node.pos = [pos.x, pos.y, pos.z]
		node_attributesAtt = []
		node_attributesAtt = node_attributesAtt + list(oneHotVector(node_type, atomtype_list_order))
		node.attributesAtt = np.array(node_attributesAtt, dtype=att_dtype)
		for neighbor in atom.GetNeighbors():
			node.neighbors.append((
				neighbor.GetIdx(),
				rdmol.GetBondBetweenAtoms(
					atom.GetIdx(),
					neighbor.GetIdx()
				).GetIdx()
			))
		graph.nodes.append(node)
	# Add counts, for convenience
	graph.num_edges = len(graph.edges)
	graph.num_nodes = len(graph.nodes)
	return graph

def bondAttributes(bond):
	'''Returns a numpy array of attributes for an RDKit bond

	From Neural FP defaults:
	The bond features were a concatenation of whether the bond type was single, double, triple,
	or aromatic, whether the bond was conjugated, and whether the bond was part of a ring.
	'''
	# Initialize
	attributes = []
	# Add bond type
	attributes += oneHotVector(
		bond.GetBondTypeAsDouble(),
		[1.0, 1.5, 2.0, 3.0]
	)
	# Add if is aromatic
	attributes.append(bond.GetIsAromatic())
	# Add if bond is conjugated
	attributes.append(bond.GetIsConjugated())
	# Add if bond is part of ring
	attributes.append(bond.IsInRing())

	# NEED THIS FOR TENSOR REPRESENTATION - 1 IF THERE IS A BOND
	attributes.append(1)

	return np.array(attributes, dtype = att_dtype)

def atomAttributes(atom, extra_attributes = [], formal_charge_one_hot=False):
	'''Returns a numpy array of attributes for an RDKit atom

	From ECFP defaults:
	<IdentifierConfiguration>
        <Property Name="AtomicNumber" Value="1"/>
        <Property Name="HeavyNeighborCount" Value="1"/>
        <Property Name="HCount" Value="1"/>
        <Property Name="FormalCharge" Value="1"/>
        <Property Name="IsRingAtom" Value="1"/>
    </IdentifierConfiguration>
    '''
	# Initialize
	attributes = []
	# Add atomic number (todo: finish)
	attributes += oneHotVector(
		atom.GetAtomicNum(),
		[5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
	)
	# Add heavy neighbor count
	attributes += oneHotVector(
		len(atom.GetNeighbors()),
		[0, 1, 2, 3, 4, 5]
	)
	# Add hydrogen count
	attributes += oneHotVector(
		atom.GetTotalNumHs(),
		[0, 1, 2, 3, 4]
	)
	# Add formal charge
	if formal_charge_one_hot:
		attributes += oneHotVector(
			atom.GetFormalCharge(),
			[-1, 0, 1]
		)
	else:
		attributes.append(atom.GetFormalCharge())
	
	# Add boolean if in ring
	attributes.append(atom.IsInRing())
	# Add boolean if aromatic atom
	attributes.append(atom.GetIsAromatic())

	attributes += extra_attributes

	return np.array(attributes, dtype = att_dtype)

def oneHotVector(val, lst):
	'''Converts a value to a one-hot vector based on options in lst'''
	if val not in lst:
		val = lst[-1]
	return map(lambda x: x == val, lst)

def sizeAttributeVector(molecular_attributes = False, formal_charge_one_hot=False):
	m = AllChem.MolFromSmiles('CC')
	AllChem.Compute2DCoords(m)
	g = molToGraph(m, molecular_attributes = molecular_attributes, formal_charge_one_hot=formal_charge_one_hot)
	a = g.nodes[0]
	b = g.edges[0]
	return len(a.attributes) + len(b.attributes)

def sizeAttributeVectors(bondtype_list_order, atomtype_list_order, molecular_attributes = False, formal_charge_one_hot=False):
	m = AllChem.MolFromSmiles('CC')
	AllChem.Compute2DCoords(m)
	g = molToGraph(m, bondtype_list_order, atomtype_list_order, molecular_attributes = molecular_attributes, formal_charge_one_hot=formal_charge_one_hot)
	a = g.nodes[0]
	b = g.edges[0]
	return len(a.attributes), len(b.attributes)

def sizeAttributeVectorsAtt(bondtype_list_order, atomtype_list_order, molecular_attributes = False, formal_charge_one_hot=False):
	m = AllChem.MolFromSmiles('CC')
	AllChem.Compute2DCoords(m)
	g = molToGraph(m, bondtype_list_order,atomtype_list_order, molecular_attributes = molecular_attributes, formal_charge_one_hot=formal_charge_one_hot)
	a = g.nodes[0]
	b = g.edges[0]
	return len(a.attributes), len(b.attributesAtt)+len(a.attributesAtt)

def fillBondType(rdmol, bondtype_list):

	# Add bonds
	for bond in rdmol.GetBonds():
		BeginAtom, EndAtom = bond.GetBeginAtom(), bond.GetEndAtom()
		begin_idx, end_idx = BeginAtom.GetAtomicNum(), EndAtom.GetAtomicNum()
		if begin_idx < end_idx:
			bond_type = str(begin_idx) + '_' + str(end_idx)
		else:
			bond_type = str(end_idx) + '_' + str(begin_idx)
		if bond_type in bondtype_list:
			pass
		else:
			bondtype_list.append(bond_type)
	for atom in rdmol.GetAtoms():
		atom_num = atom.GetAtomicNum()
		bond_type = str(atom_num) + '_' + str(atom_num)
		if bond_type in bondtype_list:
			pass
		else:
			bondtype_list.append(bond_type)
	return(bondtype_list)

def fillBondType_dic(rdmol, bondtype_dic):
	# Add bonds
	for bond in rdmol.GetBonds():
		BeginAtom, EndAtom = bond.GetBeginAtom(), bond.GetEndAtom()
		begin_idx, end_idx = BeginAtom.GetAtomicNum(), EndAtom.GetAtomicNum()
		if begin_idx < end_idx:
			bond_type = str(begin_idx) + '_' + str(end_idx)
		else:
			bond_type = str(end_idx) + '_' + str(begin_idx)
		if bond_type in bondtype_dic.keys():
			bondtype_dic[bond_type] += 1
		else:
			bondtype_dic[bond_type] = 1
	for atom in rdmol.GetAtoms():
		atom_num = atom.GetAtomicNum()
		bond_type = str(atom_num) + '_' + str(atom_num)
		if bond_type in bondtype_dic.keys():
			bondtype_dic[bond_type] += 1
		else:
			bondtype_dic[bond_type] = 1
	return(bondtype_dic)

def fillAtomType_dic(rdmol, atomtype_dic):
	for atom in rdmol.GetAtoms():
		atom_num = atom.GetAtomicNum()
		if atom_num in atomtype_dic:
			atomtype_dic[atom_num] += 1
		else:
			atomtype_dic[atom_num] = 1
	return(atomtype_dic)
