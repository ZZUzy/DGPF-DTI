import traceback

import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import random
import io
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
import esm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaModel
import torch
from utils import *

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

def seq_feature(seq):
    residue_feature = []
    for residue in seq:
        # replace some rare residue with 'X'
        if residue not in pro_res_table:
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)

    pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))
    for i in range(len(seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]

    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    return seq_feature
def sequence_to_graph(target_key, target_sequence, distance_dir):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)
    # print('***',(os.path.abspath(os.path.join(distance_dir, target_key + '.npy'))))
    contact_map_file = os.path.join(distance_dir, target_key + '.npy')
    distance_map = np.load(contact_map_file)
    # the neighbor residue should have a edge
    # add self loop
    for i in range(target_size):
        distance_map[i, i] = 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1
    # print(distance_map)
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold
    # print(len(index_row))
    # print(len(index_col))
    # print(len(index_row_))
    # print(len(index_col_))
    # print(distance_map.shape)
    # print((len(index_row) * 1.0) / (distance_map.shape[0] * distance_map.shape[1]))
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])  # dege
        target_edge_distance.append(distance_map[i, j])  # edge weight
    target_feature = seq_feature(target_sequence)
    # residue_distance = np.array(target_edge_distance)  # consistent with edge
    # print('target_feature', target_feature.shape)
    # print(target_edge_index)
    # print('target_edge_index', np.array(target_edge_index).shape)
    # print('residue_distance', residue_distance.shape)
    # return target_size, target_feature, residue_edge_index, residue_distance
    return target_size, target_feature, target_edge_index, target_edge_distance

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X
def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_hot_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    mol_size = mol.GetNumAtoms()

    mol_features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        mol_features.append(feature / sum(feature))

    edges = []
    bond_type_np = np.zeros((mol_size, mol_size))
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        bond_type_np[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetBondTypeAsDouble()
        bond_type_np[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond.GetBondTypeAsDouble()
        # bond_type.append(bond.GetBondTypeAsDouble())
    g = nx.Graph(edges).to_directed()
    # print('@@@@@@@@@@@@@@@@@')
    # print(np.array(edges).shape,'edges')
    # print(np.array(g).shape,'g')

    mol_adj = np.zeros((mol_size, mol_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    # print(np.array(mol_adj).shape,'mol_adj')
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))

    bond_edge_index = []
    bond_type = []
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        bond_edge_index.append([i, j])
        bond_type.append(bond_type_np[i, j])
    # print(bond_edge_index)
    # print('smile_to_graph')
    # print('mol_features',np.array(mol_features).shape)
    # print('bond_edge_index',np.array(bond_edge_index).shape)
    # print('bond_type',np.array(bond_type).shape)
    return mol_size, mol_features, bond_edge_index, bond_type
    # return mol_size, mol_features, bond_edge_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "D:\\DeepChemChemBERTa-77M-MTR"
model_chem = RobertaModel.from_pretrained(model_name, num_labels=2, add_pooling_layer=True)  # 600->384
tokenizer = AutoTokenizer.from_pretrained(model_name)  # token=593 vocab_size=591, model_max_len=512
model_chem = model_chem.to(device)
def get_embeddings(smile):
    # embedding_df = pd.DataFrame(columns=['SMILES'] + [f'chemberta2_feature_{i}' for i in range(1, 385)])

        # truncate to the maximum length accepted by the model if no max_length is provided
    encodings = tokenizer(smile, return_tensors='pt', padding="max_length", max_length=290,
                                truncation=True)  # input_ids=1,290, attention-mask=1,290
        # encodings = encodings.to(device)
    encodings = {key: tensor.to(device) for key, tensor in encodings.items()}
    with torch.no_grad():
        output = model_chem(**encodings)  # last_hidden_state=1,290,384 pooler_output=1,384
        smiles_embeddings = output.last_hidden_state[0, 0, :]
            # smiles_embeddings = smiles_embeddings.squeeze(0)
        smiles_embeddings = smiles_embeddings.cpu()
        smiles_embeddings = np.array(smiles_embeddings, dtype=np.float64)

            # Ensure you move the tensor back to cpu for numpy conversion
            # dic = {**{'SMILES': row['SMILES']}, **dict(zip([f'chemberta2_feature_{i}' for i in range(1, 385)], smiles_embeddings.cpu().numpy().tolist()))}
            # embedding_df.loc[len(embedding_df)] = pd.Series(dic)

    return smiles_embeddings  # smiles_embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()  # esm 33->1280 alphabet=33
batch_converter = alphabet.get_batch_converter()  # Convert sequences to vectors
esm_model = esm_model.eval().to(device)
def Get_Protein_Feature(sequence):
    sequence = sequence[0:1000]  # Truncate to 1022 residues if longer

    # Prepare data for model
    data_part = [("Seq_Target", sequence)]

    # Convert to tokens
    _, _, batch_tokens = batch_converter(data_part)  # Convert to tokens
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        # Get token embeddings from the model
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)

    # Extract the representation from the 33rd layer
    token_representations = results["representations"][33]

    # Compute the average embedding (excluding padding tokens)
    emb_rep = token_representations[0, 1:len(sequence) + 1].mean(0)  # Skip padding token at position 0
    emb_rep = emb_rep.cpu().numpy()  # Convert to numpy array

    return emb_rep

def extract_features_from_input(smile, sequence, smile_max=100, protein_max=1000):
    """
    输入SMILES字符串和蛋白质序列，返回全部所需特征。
    """
    # 1. 计算 FCFP 指纹
    fcfp_feature = get_embeddings(smile)

    # 2. 计算 ESM 特征
    esm_feature = Get_Protein_Feature(sequence)

    # 3. SMILES 图结构
    mol_size, mol_features, mol_edge_index, mol_edges_weights = smile_to_graph(smile)
    smile_graph = {
        'size': mol_size,
        'features': mol_features,
        'edge_index': mol_edge_index,
        'edge_weight': mol_edges_weights
    }

    # 4. SMILES 序列编码
    smile_int = torch.from_numpy(label_smiles(smile, CHARISOSMISET, smile_max))
    smile_len = min(len(smile), smile_max)

    # 5. 蛋白质序列编码
    protein_int = torch.from_numpy(label_sequence(sequence, CHARPROTSET, protein_max))
    protein_len = min(len(sequence), protein_max)

    # 6. 蛋白质图结构
    # 创建一个伪ID用于graph创建（sequence_to_graph需要key）
    fake_key = 'temp_prot'
    pro_distance_dir = './pre_process/dummy_dataset/distance_map'
    protein_graph = sequence_to_graph(fake_key, sequence, pro_distance_dir)
    target_size, target_features, target_edge_index, target_edge_weight = protein_graph
    target_graph = {
        'size': target_size,
        'features': target_features,
        'edge_index': target_edge_index,
        'edge_weight': target_edge_weight
    }

    return {
        "smile_graph": smile_graph,
        "smile_sequence": smile_int,
        "fcfp_feature": fcfp_feature,
        "esm_feature": esm_feature,
        "target_graph": target_graph,
        "target_sequence": protein_int,
        "smile_len": smile_len,
        "protein_len": protein_len
    }
