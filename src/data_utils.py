"""
Code based on:
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
"""

import numpy as np
import pandas as pd
import torch
import csv
from rdkit.Chem import MolFromSmiles, SDMolSupplier
from torch.utils.data import Dataset
import os
import scipy
from sklearn.utils import shuffle
import operator
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem

from .neural_fp import *

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def load_data_from_df(dataset_path, target, repeats=1, 
                      shuffle_repeats=True, sdf_file=None, 
                      dummyNode=False, formal_charge_one_hot=False):    
    data_df = pd.read_csv(dataset_path)
    
    data_x = data_df.iloc[:, 0].values
    data_y = data_df.iloc[:, 1].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)
    
    if repeats > 1:
        data_x = np.repeat(data_x, repeats=repeats)
        data_y = np.repeat(data_y, repeats=repeats)
        
        if shuffle_repeats:
            new_order = shuffle(list(range(data_x.shape[0])))

            data_x = data_x[new_order]
            data_y = data_y[new_order]
    
    x_all, y_all, target, mol_sizes = load_data_from_smiles(data_x, data_y, target, sdf_file=sdf_file, 
                                                            dummyNode=dummyNode, formal_charge_one_hot=formal_charge_one_hot)

    return (x_all, y_all, target, mol_sizes)


def load_data_from_smiles(smiles, labels, target, 
                          bondtype_freq=20, atomtype_freq=10, sdf_file=None,
                          dummyNode=False, formal_charge_one_hot=False):
    bondtype_dic = {}
    atomtype_dic = {}
    for smile in smiles:
        try:
            mol = MolFromSmiles(smile)
            bondtype_dic = fillBondType_dic(mol, bondtype_dic)
            atomtype_dic = fillAtomType_dic(mol, atomtype_dic)
        except AttributeError:
            pass
        else:
            pass

    sorted_bondtype_dic = sorted(bondtype_dic.items(), key=operator.itemgetter(1))
    sorted_bondtype_dic.reverse()
    bondtype_list_order = [ele[0] for ele in sorted_bondtype_dic]
    bondtype_list_number = [ele[1] for ele in sorted_bondtype_dic]

    filted_bondtype_list_order = []
    for i in range(0, len(bondtype_list_order)):
        if bondtype_list_number[i] > bondtype_freq:
            filted_bondtype_list_order.append(bondtype_list_order[i])
    filted_bondtype_list_order.append('Others')

    sorted_atom_types_dic = sorted(atomtype_dic.items(), key=operator.itemgetter(1))
    sorted_atom_types_dic.reverse()
    atomtype_list_order = [ele[0] for ele in sorted_atom_types_dic]
    atomtype_list_number = [ele[1] for ele in sorted_atom_types_dic]

    filted_atomtype_list_order = []
    for i in range(0, len(atomtype_list_order)):
        if atomtype_list_number[i] > atomtype_freq:
            filted_atomtype_list_order.append(atomtype_list_order[i])
    filted_atomtype_list_order.append('Others')

    print('filted_atomtype_list_order: {}, \n filted_bondtype_list_order: {}'.format(filted_atomtype_list_order, filted_bondtype_list_order))

    # mol to graph
    i = 0
    mol_sizes = []
    x_all = []
    y_all = []

    print('Transfer mol to matrices')
    if sdf_file:
        mols = []
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=False)
        for mol in suppl:
            c = mol.GetConformers()[0]
            new_mol = Chem.RemoveHs(mol, sanitize=False)
            for i in range(new_mol.GetNumAtoms()):
                new_mol.GetConformers()[0].SetAtomPosition(
                    i, (c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z))
            mols.append(new_mol)
    for smile, label in zip(smiles, labels):
        try:
            ##### CONFORMATION FOR MOLECULE DIST MATRIX #####
            if not sdf_file:
                mol = MolFromSmiles(smile)
                try:
                    # 3d
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, maxAttempts=5000)
                    AllChem.UFFOptimizeMolecule(mol)
                    mol = Chem.RemoveHs(mol)
                except:
                    # 2d
                    AllChem.Compute2DCoords(mol)
                #####
            else:
                mol = mols.pop(0)

            if dummyNode:
                (afm, adj, bft, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt, mat_positions) = molToGraph(mol, filted_bondtype_list_order, filted_atomtype_list_order, formal_charge_one_hot=formal_charge_one_hot).dump_as_matrices_Att_dummyNode()
            else:
                (afm, adj, bft, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt, mat_positions) = molToGraph(mol, filted_bondtype_list_order, filted_atomtype_list_order, formal_charge_one_hot=formal_charge_one_hot).dump_as_matrices_Att()

            x_all.append([afm, adj, bft, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt, mat_positions])
            y_all.append([label])
            mol_sizes.append(adj.shape[0])
            # feature matrices of mols, include Adj Matrix, Atom Feature, Bond Feature.
        except AttributeError:
            print('the smile: {} has an error'.format(smile))
        except RuntimeError:
            print('the smile: {} has an error'.format(smile))
        except ValueError as e:
            print('the smile: {}, can not convert to graph structure'.format(smile))
            print(e)
        except:
            print('the smile: {} has an error'.format(smile))
        else:
            pass

    print('Done.')
    return (x_all, y_all, target, mol_sizes)


class MolDatum():
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """
    def __init__(self, x, label, target, index):
        self.adj = x[1]
        self.afm = x[0]
        self.bft = x[2]
        self.orderAtt = x[3]
        self.aromAtt = x[4]
        self.conjAtt = x[5]
        self.ringAtt = x[6]
        self.distances = x[7]
        self.label = label
        self.target = target
        self.index = index

        
def construct_dataset(x_all, y_all, target):
    output = []
    for i in range(len(x_all)):
        output.append(MolDatum(x_all[i], y_all[i], target, i))
    return(output)


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of MolDatum
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        adj, afm, bft, orderAtt, aromAtt, conjAtt, ringAtt, distances  = self.data_list[key].adj, self.data_list[key].afm, self.data_list[key].bft, self.data_list[key].orderAtt, self.data_list[key].aromAtt, self.data_list[key].conjAtt, self.data_list[key].ringAtt, self.data_list[key].distances
        label = self.data_list[key].label
        return (adj, afm, bft, orderAtt, aromAtt, conjAtt, ringAtt, distances, label)

    
def mol_collate_func_reg(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    adj_list = []
    afm_list =[]
    label_list = []
    size_list = []
    bft_list = []
    distances_list = []
    orderAtt_list, aromAtt_list, conjAtt_list, ringAtt_list = [], [], [], []
    for datum in batch:
        if type(datum[8][0]) == np.ndarray:
            label_list.append(datum[8][0])
        else:
            label_list.append(datum[8])
        size_list.append(datum[0].shape[0])
    max_size = np.max(size_list)
    btf_len = datum[2].shape[0]
    # padding
    for datum in batch:
        filled_adj = np.zeros((max_size, max_size), dtype=np.float32)
        filled_adj[0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[0]
        
        filled_afm = np.zeros((max_size, datum[1].shape[1]), dtype=np.float32)
        filled_afm[0:datum[0].shape[0], :] = datum[1]
        
        filled_bft = np.zeros((btf_len, max_size, max_size), dtype=np.float32)
        filled_bft[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[2]

        filled_orderAtt = np.zeros((5, max_size, max_size), dtype=np.float32)
        filled_orderAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[3]

        filled_aromAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_aromAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[4]

        filled_conjAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_conjAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[5]

        filled_ringAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_ringAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[6]
        
        filled_distances = np.zeros((max_size, max_size), dtype=np.float32)
        filled_distances[0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[7]

        adj_list.append(filled_adj)
        afm_list.append(filled_afm)
        bft_list.append(filled_bft)
        orderAtt_list.append(filled_orderAtt)
        aromAtt_list.append(filled_aromAtt)
        conjAtt_list.append(filled_conjAtt)
        ringAtt_list.append(filled_ringAtt)
        distances_list.append(filled_distances)


    if use_cuda:
        return ([torch.from_numpy(np.array(adj_list)).cuda(), torch.from_numpy(np.array(afm_list)).cuda(),
                 torch.from_numpy(np.array(bft_list)).cuda(), torch.from_numpy(np.array(orderAtt_list)).cuda(),
                 torch.from_numpy(np.array(aromAtt_list)).cuda(), torch.from_numpy(np.array(conjAtt_list)).cuda(),
                 torch.from_numpy(np.array(ringAtt_list)).cuda(),
                 torch.from_numpy(np.array(distances_list)).cuda(),
                 torch.from_numpy(np.array(label_list)).cuda()])
    else:
        return ([torch.from_numpy(np.array(adj_list)), torch.from_numpy(np.array(afm_list)),
                 torch.from_numpy(np.array(bft_list)), torch.from_numpy(np.array(orderAtt_list)),
                 torch.from_numpy(np.array(aromAtt_list)), torch.from_numpy(np.array(conjAtt_list)),
                 torch.from_numpy(np.array(ringAtt_list)),
                 torch.from_numpy(np.array(distances_list)),
                 torch.from_numpy(np.array(label_list))])
    
    
def mol_collate_func_class(batch):
    adj_list = []
    afm_list =[]
    label_list = []
    size_list = []
    bft_list = []
    distances_list = []
    orderAtt_list, aromAtt_list, conjAtt_list, ringAtt_list = [], [], [], []

    for datum in batch:
        label_list.append(datum[8])
        size_list.append(datum[0].shape[0])
    max_size = np.max(size_list)
    btf_len = datum[2].shape[0]
    # padding
    for datum in batch:
        filled_adj = np.zeros((max_size, max_size), dtype=np.float32)
        filled_adj[0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[0]
        filled_afm = np.zeros((max_size, datum[1].shape[1]), dtype=np.float32)
        filled_afm[0:datum[0].shape[0], :] = datum[1]
        filled_bft = np.zeros((btf_len, max_size, max_size), dtype=np.float32)
        filled_bft[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[2]

        filled_orderAtt = np.zeros((5, max_size, max_size), dtype=np.float32)
        filled_orderAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[3]

        filled_aromAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_aromAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[4]

        filled_conjAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_conjAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[5]

        filled_ringAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_ringAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[6]
        
        filled_distances = np.zeros((max_size, max_size), dtype=np.float32)
        filled_distances[0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[7]

        adj_list.append(filled_adj)
        afm_list.append(filled_afm)
        bft_list.append(filled_bft)
        orderAtt_list.append(filled_orderAtt)
        aromAtt_list.append(filled_aromAtt)
        conjAtt_list.append(filled_conjAtt)
        ringAtt_list.append(filled_ringAtt)
        distances_list.append(filled_distances)

    if use_cuda:
        return ([torch.from_numpy(np.array(adj_list)).cuda(), torch.from_numpy(np.array(afm_list)).cuda(),
                 torch.from_numpy(np.array(bft_list)).cuda(), torch.from_numpy(np.array(orderAtt_list)).cuda(),
                 torch.from_numpy(np.array(aromAtt_list)).cuda(), torch.from_numpy(np.array(conjAtt_list)).cuda(),
                 torch.from_numpy(np.array(ringAtt_list)).cuda(),
                 torch.from_numpy(np.array(distances_list)).cuda(),
                 FloatTensor(label_list)])
    else:
        return ([torch.from_numpy(np.array(adj_list)), torch.from_numpy(np.array(afm_list)),
             torch.from_numpy(np.array(bft_list)),torch.from_numpy(np.array(orderAtt_list)),
                 torch.from_numpy(np.array(aromAtt_list)), torch.from_numpy(np.array(conjAtt_list)),
                 torch.from_numpy(np.array(ringAtt_list)),
                 torch.from_numpy(np.array(distances_list)),
                 FloatTensor(label_list)])

    
def construct_loader(x, y, target, batch_size, shuffle=True):
    data_set = construct_dataset(x, y, target)
    data_set = MolDataset(data_set)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=batch_size,
                                               collate_fn=mol_collate_func_class,
                                               shuffle=shuffle)
    return loader


def construct_loader_reg(x, y, target, batch_size, shuffle=True):
    data_set = construct_dataset(x, y, target)
    data_set = MolDataset(data_set)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=batch_size,
                                               collate_fn=mol_collate_func_reg,
                                               shuffle=shuffle)
    return loader
