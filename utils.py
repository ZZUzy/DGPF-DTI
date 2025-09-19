import os
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
from tqdm import tqdm

def process_tensor(tensor):
    heads_mean = tensor.mean(dim=1).squeeze(0)

    drug_mean = heads_mean.mean(dim=1, keepdim=True).squeeze(1)

    drug_mean_shape = drug_mean.shape[0]+1
    print(drug_mean_shape)
    target_mean = heads_mean.mean(dim=0, keepdim=True).squeeze(0)
    target_mean_shape = target_mean.shape[0]+1
    print(target_mean_shape)
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    print('***************************************************')
    drug_normalized = normalize(drug_mean[1:drug_mean_shape])
    print(drug_normalized.shape)
    target_normalized = normalize(target_mean[1:target_mean_shape])
    print(target_normalized.shape)

    return drug_normalized, target_normalized

# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='D:\\...', dataset='davis',
                 xd=None, y=None, transform=None,pre_transform=None, smile_graph=None, target_key=None, target_graph=None, esm=None, fcfp=None, target_seq=None, smile_sequence=None, smile_len=None, target_len=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.drug = xd
        self.target = target_key
        self.y = y
        self.esm = esm
        self.fcfp = fcfp
        self.target_seq = target_seq
        self.smile_graph = smile_graph
        self.target_graph = target_graph
        self.smile_sequence = smile_sequence
        self.smile_len = smile_len
        self.target_len = target_len
        self.process(xd, target_key, y, smile_graph, target_graph, target_seq, smile_sequence, esm, fcfp, smile_len, target_len)

    @property
    def raw_file_names(self):
        pass

    # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_graph, target_seq, smile_sequence, esm, fcfp, smile_len, target_len):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'The three lists must have the same length!'
        data_list_mol = []
        data_list_pro = []
        data_list_target_seq = []
        data_list_smile_sequence = []
        data_list_esm = []
        data_list_fcfp = []
        data_list_smile_len = []
        data_list_target_len = []
        data_len = len(xd)
        print('loading tensors ...')
        for i in tqdm(range(data_len)):
            smiles = xd[i]
            tar_key = target_key[i]
            labels = y[i]
            seq = target_seq[tar_key]
            smile = smile_sequence[smiles]
            esm_feature = esm[tar_key]
            fcfp_feature = fcfp[smiles]
            smile_L = smile_len[smiles]
            seq_L = target_len[tar_key]

            # convert SMILES to molecular representation using rdkit
            mol_size, mol_features, mol_edge_index, mol_edges_weights = smile_graph[smiles]
            target_size, target_features, target_edge_index, target_edge_weight = target_graph[tar_key]

            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData_mol = DATA.Data(x=torch.Tensor(mol_features),
                                    edge_index=torch.LongTensor(mol_edge_index).transpose(1, 0),
                                    edge_weight=torch.FloatTensor(mol_edges_weights),
                                    y=torch.FloatTensor([labels]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([mol_size]))

            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    edge_weight=torch.FloatTensor(target_edge_weight),
                                    y=torch.FloatTensor([labels]))
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)
            data_list_target_seq.append(seq)
            data_list_smile_sequence.append(smile)
            data_list_esm.append(esm_feature)
            data_list_fcfp.append(fcfp_feature)
            data_list_target_len.append(seq_L)
            data_list_smile_len.append(smile_L)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        self.data_list_target_seq = data_list_target_seq
        self.data_list_esm = data_list_esm
        self.data_list_fcfp = data_list_fcfp
        self.data_list_smile_sequence = data_list_smile_sequence
        self.data_list_smile_len = data_list_smile_len
        self.data_list_target_len = data_list_target_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return GNNData_mol, GNNData_pro
        return self.data_mol[idx], self.data_pro[idx], self.data_list_esm[idx], self.data_list_fcfp[idx], self.data_list_target_seq[idx], self.data_list_smile_sequence[idx], self.data_list_smile_len[idx], self.data_list_target_len[idx]


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, loss_fn, TRAIN_BATCH_SIZE=512):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        drug_scope = data[2]
        protein_scope = data[3]
        esm = [torch.tensor(data).to(device) for data in data[4]]
        fcfp = [torch.tensor(data).float().to(device) for data in data[5]]
        seq = [data.clone().detach().to(device) for data in data[6]]
        smile = [data.clone().detach().to(device) for data in data[7]]
        smile_L = [torch.tensor(data).to(device) for data in data[8]]
        seq_L = [torch.tensor(data).to(device) for data in data[9]]
        optimizer.zero_grad()
        output, _, _ = model(data_mol, data_pro, drug_scope, protein_scope, esm, fcfp, seq, smile, smile_L, seq_L)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

# predict
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    weight_atom_total = torch.Tensor()
    weight_drug_total = torch.Tensor()
    T = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            drug_scope = data[2]
            protein_scope = data[3]
            esm = [torch.tensor(data).to(device) for data in data[4]]
            fcfp = [torch.tensor(data).float().to(device) for data in data[5]]
            seq = [data.clone().detach().to(device) for data in data[6]]
            smile = [data.clone().detach().to(device) for data in data[7]]
            smile_L = [torch.tensor(data).to(device) for data in data[8]]
            seq_L = [torch.tensor(data).to(device) for data in data[9]]
            output, weight_drug, weight_pro = model(data_mol, data_pro, drug_scope, protein_scope, esm, fcfp, seq, smile, smile_L, seq_L)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
            weight_atom_total = torch.cat((weight_atom_total, weight_drug.cpu()), 0)
            weight_drug_total = torch.cat((weight_drug_total, weight_pro.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), weight_atom_total.numpy().flatten(), weight_drug_total.numpy()#, T_New.numpy()


# prepare the protein and drug pairs
def collate(data_list):
    drug_scope = []
    protein_scope = []
    esm = []
    fcfp = []
    seq = []
    smile_seq = []
    smile_L = []
    seq_L = []
    for data in data_list:
        # 假设你要提取的内容是 `x` 或其他数据字段，而不是直接用索引
        drug_scope.append(tuple(data[0].x.shape))  # 或根据需要访问不同的属性
        protein_scope.append(tuple(data[1].x.shape))
        esm.append(data[2])
        fcfp.append(data[3])
        seq.append(data[4])
        smile_seq.append(data[5])
        smile_L.append(data[6])
        seq_L.append(data[7])
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])

    return batchA, batchB, drug_scope, protein_scope, esm, fcfp, seq, smile_seq, smile_L, seq_L
