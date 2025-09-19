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
sys.path.append('/')


# Convert the sets to DataFrame and save as CSV (without column names)
def save_to_csv(entries, filename):
    # The entries contain [Drug_SMILES, Protein_Seq, Interaction_Value]
    df = pd.DataFrame(entries)
    df.to_csv(filename, index=False, header=False)

# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

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

# nomarlize the residue feature
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

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

# one ont encoding
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


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


# mol smile to mol graph edge index
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

# target sequence to target graph
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


# data write to csv file
def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('drug_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def create_fold_setting_cold(df, fold_seed, frac, entities):
    """create cold-split where given one or multiple columns, it first splits based on
    entities in the columns and then maps all associated data points to the partition

    Args:
            df (pd.DataFrame): dataset dataframe
            fold_seed (int): the random seed
            frac (list): a list of train/valid/test fractions
            entities (Union[str, List[str]]): either a single "cold" entity or a list of
                    "cold" entities on which the split is done

    Returns:
            dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e]
        .drop_duplicates()
        .sample(frac=test_frac, replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a "
            "less stringent splitting strategy."
        )

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e]
        .drop_duplicates()
        .sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac "
            "or a less stringent splitting strategy."
        )

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_DTA_dataset(dataset='davis'):
    # load dataset
    dataset_dir = os.path.join('data', dataset)
    # drug smiles
    ligands = json.load(open(os.path.join(dataset_dir, 'ligands_can.txt')), object_pairs_hook=OrderedDict)
    # protein sequences
    proteins = json.load(open(os.path.join(dataset_dir, 'proteins.txt')), object_pairs_hook=OrderedDict)
    # affinity
    affinity = pickle.load(open(os.path.join(dataset_dir, 'Y'), 'rb'), encoding='latin1')
    # dataset divide
    train_fold_origin = json.load(open(os.path.join(dataset_dir, 'folds', 'train_fold_setting1.txt')))
    test_set = json.load(open(os.path.join(dataset_dir, 'folds', 'test_fold_setting1.txt')))
    train_set = [tt for t in train_fold_origin for tt in t]

    # load protein feature and predicted distance map
    process_dir = os.path.join('./', 'pre_process')
    pro_distance_dir = os.path.join(process_dir, dataset, 'distance_map')  # numpy .npy file

    # dataset process
    drugs = []  # rdkit entity
    prots = []  # sequences
    prot_keys = []  # protein id (or name)
    drug_smiles = []  # smiles
    # smiles
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    # seqs
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)

    # consist with deepDTA
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    # dataset content load
    opts = ['train', 'test']
    for opt in opts:
        if opt == 'train':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[train_set], cols[train_set]
            train_set_entries = []
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                train_set_entries.append(ls)

            csv_file = dataset + '_' + opt + '.csv'
            data_to_csv(csv_file, train_set_entries)

        elif opt == 'test':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[test_set], cols[test_set]
            test_set_entries = []
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                test_set_entries.append(ls)

            csv_file = dataset + '_' + opt + '.csv'
            data_to_csv(csv_file, test_set_entries)

    print('dataset:', dataset)
    # print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))
    print('train entries:', len(train_set))
    print('test entries:', len(test_set))

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for i in tqdm(range(len(prot_keys))):
        key = prot_keys[i]
        g_t = sequence_to_graph(key, proteins[key], pro_distance_dir)
        target_graph[key] = g_t

    # create smile graph
    smile_graph = {}
    for i in tqdm(range(len(drugs))):
        smile = drugs[i]
        print(smile)
        g_d = smile_to_graph(smile)
        smile_graph[smile] = g_d

    # for test
    # for smile in drugs:
    #     g_d = smile_to_graph(smile)
    #     smile_graph[smile] = g_d
    # count += 1
    # print(count, len(drugs), 'drugs')
    # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    # 'data/davis_fold_0_train.csv' or data/kiba_fold_0__train.csv'
    # train dataset construct
    train_csv = dataset + '_train' + '.csv'
    df_train_set = pd.read_csv(train_csv)
    train_drugs, train_prot_keys, train_Y = list(df_train_set['drug_smiles']), list(
        df_train_set['target_key']), list(df_train_set['affinity'])
    train_drugs, train_prot_keys, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_Y)
    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs,
                               target_key=train_prot_keys, y=train_Y, smile_graph=smile_graph,
                               target_graph=target_graph)

    # test dataset construct
    test_csv = dataset + '_test.csv'
    df_test_fold = pd.read_csv(test_csv)
    test_drugs, test_prots_keys, test_Y = list(df_test_fold['drug_smiles']), list(
        df_test_fold['target_key']), list(df_test_fold['affinity'])
    test_drugs, test_prots_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prots_keys), np.asarray(
        test_Y)
    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'test', xd=test_drugs,
                              target_key=test_prots_keys, y=test_Y, smile_graph=smile_graph,
                              target_graph=target_graph)
    return train_dataset, test_dataset

# ChemBerta Drug
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "D:\\DZY\\DeepChemChemBERTa-77M-MTR"
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

# Initialize ESM model and alphabet
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

def create_CPI_dataset(dataset='davis', ratio_n=1, save_dir='D:\\DZY\\WGNN-DTA-main\\WGNN-DTA-main\\data\\dataset_folds\\Davis_3407\\'):
    # load dataset
    data_file = os.path.join('D:\\DZY\\WGNN-DTA-main\\WGNN-DTA-main\\data', dataset, 'original', 'data.txt')
    proteins_file = os.path.join('D:\\DZY\\WGNN-DTA-main\\WGNN-DTA-main\\data', dataset, 'proteins.txt')

    if not os.path.exists(proteins_file):
        proteins = {}
        seq_list = []
        count = 1
        for line in open(data_file, 'r').readlines():
            if line.strip() == '':
                continue
            # print(line)
            arrs = line.split(' ')
            entry = [arrs[0], arrs[1], arrs[2]]
            if entry[1] not in seq_list:
                # print('involve.')
                key_str = 'prot' + str(count)
                assert key_str not in proteins.keys()
                proteins[key_str] = entry[1]
                seq_list.append(entry[1])
                count += 1
            # print(entry)
        save_obj(proteins, os.path.join(proteins_file))

    proteins = load_obj(os.path.join(proteins_file))
    proteins_rev = {v: k for k, v in proteins.items()}
    # proteins_rev_rev = {k: v for k, v in proteins.items()}

    all_entries = []
    all_p_entries = []
    all_n_entries = []
    drug_smiles = []
    pro_seqs = []
    pro_keys = []

    for line in open(data_file, 'r').readlines():
        if line.strip() == '':
            continue
        arrs = line.split(' ')
        entry = [arrs[0], proteins_rev[arrs[1]], float(arrs[2])]
        drug_smiles.append(arrs[0])
        pro_seqs.append(arrs[1])
        pro_keys.append(proteins_rev[arrs[1]])
        all_entries.append(entry)
        if float(arrs[2]) == 1:
            all_p_entries.append(entry)
        if float(arrs[2]) == 0:
            all_n_entries.append(entry)

    drug_smiles = list(set(drug_smiles))
    pro_seqs = list(set(pro_seqs))
    pro_keys = list(set(pro_keys))

    print('drug number:', len(drug_smiles))
    print('protein number:', len(pro_seqs), len(pro_keys))
    print('number of entries:', len(all_entries))
    print('number of positive entries:', len(all_p_entries))
    print('number of negative entries:', len(all_n_entries))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 检查是否已经存在划分好的数据集
    train_file = os.path.join(save_dir, 'train_set.csv')
    dev_file = os.path.join(save_dir, 'dev_set.csv')
    test_file = "D:\\DZY\\WGNN-DTA-main\\WGNN-DTA-main\\test_set.csv"
    # test_file = os.path.join(save_dir, 'test_set.csv')

    if os.path.exists(train_file) and os.path.exists(dev_file) and os.path.exists(test_file):
        print("已经存在划分好的数据集，正在加载...")
        # 直接加载已存在的划分好的数据集
        train_set = pd.read_csv(train_file, header=None)
        dev_set = pd.read_csv(dev_file, header=None)
        test_set = pd.read_csv(test_file, header=None)

    else:
        # np.random.seed(3407)
        # # shuffle
        # random.shuffle(all_entries)
        # random.shuffle(all_p_entries)
        # random.shuffle(all_n_entries)
        #
        # # to used all data
        # used_entries = all_entries
        #
        # # split training, validation and test sets
        # used_entries = np.array(used_entries)
        # ratio = 0.8
        # n = int(ratio * len(used_entries))
        # train_set, dataset_ = used_entries[:n], used_entries[n:]
        # ratio = 0.5
        # n = int(ratio * len(dataset_))
        # dev_set, test_set = dataset_[:n], dataset_[n:]
        #
        # # Save datasets as CSV files (no column headers)
        # save_to_csv(train_set, train_file)
        # save_to_csv(dev_set, dev_file)
        # save_to_csv(test_set, test_file)
        # print("Warm up !!! Training set, validation set, and test set have been saved")

        # 冷启动数据集的划分
        print("Creating cold-start datasets...")
        # Create cold-start train/valid/test sets
        cold_train_set = create_fold_setting_cold(
            df=pd.DataFrame(all_entries, columns=["drug", "protein", "interaction"]),
            fold_seed=3407,
            frac=[0.8, 0.1, 0.1],
            entities=["drug"]  # 仅按药物进行冷启动划分
        )
        cold_train_set["train"].to_csv(os.path.join(save_dir, 'cold_drug_train_set.csv'), index=False)
        cold_train_set["valid"].to_csv(os.path.join(save_dir, 'cold_drug_dev_set.csv'), index=False)
        cold_train_set["test"].to_csv(os.path.join(save_dir, 'cold_drug_test_set.csv'), index=False)

        # Create target cold-start dataset (e.g., target cold-start split)
        cold_target_set = create_fold_setting_cold(
            df=pd.DataFrame(all_entries, columns=["drug", "protein", "interaction"]),
            fold_seed=3407,
            frac=[0.8, 0.1, 0.1],
            entities=["protein"]  # 仅按靶标进行冷启动划分
        )
        cold_target_set["train"].to_csv(os.path.join(save_dir, 'cold_target_train_set.csv'), index=False)
        cold_target_set["valid"].to_csv(os.path.join(save_dir, 'cold_target_dev_set.csv'), index=False)
        cold_target_set["test"].to_csv(os.path.join(save_dir, 'cold_target_test_set.csv'), index=False)

        # Create Unseen cold-start dataset (e.g., neither drug nor protein has been seen)
        cold_unseen_set = create_fold_setting_cold(
            df=pd.DataFrame(all_entries, columns=["drug", "protein", "interaction"]),
            fold_seed=3407,
            frac=[0.8, 0.1, 0.1],
            entities=["drug", "protein"]  # 既按药物也按靶标进行冷启动划分
        )
        cold_unseen_set["train"].to_csv(os.path.join(save_dir, 'unseen_train_set.csv'), index=False)
        cold_unseen_set["valid"].to_csv(os.path.join(save_dir, 'unseen_dev_set.csv'), index=False)
        cold_unseen_set["test"].to_csv(os.path.join(save_dir, 'unseen_test_set.csv'), index=False)

        print("Cold-start datasets have been saved.")

    # drug to fcfp
    fcfp_feature = {}
    for i in tqdm(range(len(drug_smiles))):
        smile = drug_smiles[i]
        fcfp_embeding = get_embeddings(smile)
        fcfp_feature[smile] = fcfp_embeding

    # protein to esm
    esm_feature = {}
    for i in tqdm(range(len(pro_keys))):
        key = pro_keys[i]
        seq = proteins[key]
        esm_embeding = Get_Protein_Feature(seq)
        esm_feature[key] = esm_embeding

    process_dir = os.path.join('./', 'pre_process')
    pro_distance_dir = os.path.join(process_dir, dataset, 'distance_map')  # numpy .npy file
    # create target graph
    target_graph = {}
    target_lenth = {}
    for i in tqdm(range(len(pro_keys))):
        key = pro_keys[i]
        seq = proteins[key]
        g_t = sequence_to_graph(key, seq, pro_distance_dir)
        target_graph[key] = g_t
        target_lenth[key] = len(seq)

# create target sequence
    target_sequence = {}
    protein_max = 1000
    for i in tqdm(range(len(pro_keys))):
        key = pro_keys[i]
        proteinstr = proteins[key]
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, protein_max))
        target_sequence[key] = proteinint

    # create smile graph
    smile_graph = {}
    smile_lenth = {}
    for i in tqdm(range(len(drug_smiles))):
        smile = drug_smiles[i]
        g_d = smile_to_graph(smile)
        smile_graph[smile] = g_d
        smile_lenth[smile] = len(smile)

    smile_sequence = {}
    smile_max = 100
    for i in tqdm(range(len(drug_smiles))):
        smile = drug_smiles[i]
        smileint = torch.from_numpy(label_smiles(smile, CHARISOSMISET, smile_max))
        smile_sequence[smile] = smileint

    # 'data/davis_fold_0_train.csv' or data/kiba_fold_0__train.csv'
    # train dataset construct
    train_drugs, train_prot_keys, train_Y = np.asarray(train_set)[:, 0], np.asarray(train_set)[:, 1], np.asarray(
        train_set)[:, 2]
    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs, target_key=train_prot_keys,
                               y=train_Y.astype(float), smile_graph=smile_graph, target_graph=target_graph,target_seq=target_sequence,
                               smile_sequence=smile_sequence, esm=esm_feature, fcfp=fcfp_feature, smile_len=smile_lenth, target_len=target_lenth)
    # valid dataset construct
    dev_drugs, dev_prot_keys, dev_Y = np.asarray(dev_set)[:, 0], np.asarray(dev_set)[:, 1], np.asarray(dev_set)[:, 2]
    dev_dataset = DTADataset(root='data', dataset=dataset + '_' + 'dev', xd=dev_drugs,
                             target_key=dev_prot_keys, y=dev_Y.astype(float), smile_graph=smile_graph,target_graph=target_graph, target_seq=target_sequence,
                             smile_sequence=smile_sequence, esm=esm_feature, fcfp=fcfp_feature, smile_len=smile_lenth, target_len=target_lenth)
    # test dataset construct
    test_drugs, test_prots_keys, test_Y = np.asarray(test_set)[:, 0], np.asarray(test_set)[:, 1], np.asarray(
        test_set)[:, 2]
    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'test', xd=test_drugs,
                              target_key=test_prots_keys, y=test_Y.astype(float), smile_graph=smile_graph,
                              target_graph=target_graph, target_seq=target_sequence,
                              smile_sequence=smile_sequence, esm=esm_feature, fcfp=fcfp_feature, smile_len=smile_lenth, target_len=target_lenth)
    # temp_y = test_Y.astype(float) # for test
    # print(type(temp_y))
    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    create_CPI_dataset('human', 1)
