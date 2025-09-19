import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

class LinearTransform(nn.Module):
    def __init__(self):
        super(LinearTransform, self).__init__()
        self.linear1 = nn.Linear(384, 512)  # for seed 12 for better score it was on 384>64>128
        self.linear2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Reshape the input tensor to [batch_size, 1024]
        x = x.view(x.size(0), -1)
        # Apply the first linear layer
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        # Apply the second linear layer
        x = torch.relu(self.linear2(x))

        return x
class LinearTransform_esm(nn.Module):
    def __init__(self):
        super(LinearTransform_esm, self).__init__()
        self.linear1 = nn.Linear(1280, 512)
        self.linear2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Reshape the input tensor to [batch_size, 1024]
        x = x.view(x.size(0), -1)

        # Apply the first linear layer
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)

        # Apply the second linear layer
        x = torch.relu(self.linear2(x))

        return x


# GAT based model
class CPI_GAT(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, output_dim=128,
                 dropout=0.2):
        super(CPI_GAT, self).__init__()

        print('CPI_GAT loading ...')
        self.n_output = n_output
        self.mol_conv1 = GATConv(num_features_mol, num_features_mol, heads=2, dropout=dropout)
        self.mol_conv2 = GATConv(num_features_mol * 2, num_features_mol * 2, heads=2, dropout=dropout)
        self.mol_conv3 = GATConv(num_features_mol * 4, num_features_mol * 4, dropout=dropout)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 8, 1024)  # num_features_mol * 4
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GATConv(num_features_pro, num_features_pro * 2, heads=2, dropout=dropout)
        self.pro_conv3 = GATConv(num_features_pro * 4, num_features_pro * 4, dropout=dropout)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 8, 1024)  # num_features_pro * 4
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.protein_esm = LinearTransform_esm()
        self.drug_chem = LinearTransform()

        self.seq_encoder = Seq_Encoder()
        self.smile_encoder = Smile_Encoder()
        self.embeding_dim = 256
        self.is_bidirectional = True
        self.bilstm_layers = 2
        self.lstm_dim = 128
        self.hidden_dim = 256
        self.n_heads = 8
        # smile
        self.smiles_input_fc = nn.Linear(self.embeding_dim, self.lstm_dim)
        self.smiles_lstm = nn.LSTM(self.lstm_dim, self.lstm_dim, self.bilstm_layers, batch_first=True,
                                   bidirectional=self.is_bidirectional, dropout=0.2)
        self.ln1 = torch.nn.LayerNorm(self.lstm_dim * 2)
        self.enhance1 = GroupingUpdateModule(groups=20)
        # protein
        self.protein_input_fc = nn.Linear(self.embeding_dim, self.lstm_dim)
        self.protein_lstm = nn.LSTM(self.lstm_dim, self.lstm_dim, self.bilstm_layers, batch_first=True,
                                    bidirectional=self.is_bidirectional, dropout=0.2)
        self.ln2 = torch.nn.LayerNorm(self.lstm_dim * 2)
        self.enhance2 = GroupingUpdateModule(groups=200)

        self.smiletopro = Attention(256, 256)
        self.protosmile = Attention(256, 256)

        self.xtoxt = Attention(128, 128)
        self.xttox = Attention(128, 128)

        self.llmfc1 = nn.Linear(256, 512)
        self.llmfc2 = nn.Linear(512, 512)
        self.llmout = nn.Linear(512, 128)
        #
        self.graphfc1 = nn.Linear(256, 512)
        self.graphfc2 = nn.Linear(512, 512)
        self.graphout = nn.Linear(512, 128)
        #
        self.squfc1 = nn.Linear(512, 1024)
        self.squfc2 = nn.Linear(1024, 512)
        self.squout = nn.Linear(512, 128)
        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.sigmoid = nn.Sigmoid()
        #
        self.llmsqufc1 = nn.Linear(256, 512)
        self.llmsqufc2 = nn.Linear(512, 512)
        self.llmsquout = nn.Linear(512, 128)
        #
        self.llmgrafc1 = nn.Linear(256, 512)
        self.llmgrafc2 = nn.Linear(512, 512)
        self.llmgraout = nn.Linear(512, 128)

    def forward(self, data_mol, data_pro, drug_scope, protein_scope, esm, fcfp, seq, smile):
        # get graph input
        mol_x, mol_edge_index, mol_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch

        # llm drug
        fcfp_tensor = torch.stack(fcfp, dim=0)
        fcfp = self.drug_chem(fcfp_tensor)

        # llm pro
        esm_tensor = torch.stack(esm, dim=0)
        esm = self.protein_esm(esm_tensor)

        # drug graph
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv3(x, mol_edge_index)

        mol_vecs = []
        start_idx = 0
        for (slice, _) in drug_scope:
            end_idx = start_idx + slice  # 计算每个切片的结束位置
            mol_vecs.append(x[start_idx:end_idx, :])  # 切片并添加到列表中
            start_idx = end_idx  # 更新起始索引

        max_node = max(d.size(0) for d in mol_vecs)
        adaptive_avgpool = nn.AdaptiveAvgPool1d(max_node)
        adaptive_maxpool = nn.AdaptiveMaxPool1d(max_node)
        avgpooled_graphs = torch.stack(
            [adaptive_avgpool(g.unsqueeze(0).transpose(1, 2)).squeeze(0).transpose(0, 1) for g in mol_vecs], dim=0)
        maxpooled_graphs = torch.stack(
            [adaptive_maxpool(g.unsqueeze(0).transpose(1, 2)).squeeze(0).transpose(0, 1) for g in mol_vecs], dim=0)

        x = torch.cat((avgpooled_graphs,maxpooled_graphs),2) # 312
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # res graph
        xt = self.pro_conv1(target_x, target_edge_index, target_weight)
        xt = self.relu(xt)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        res_vecs = []
        start_idx = 0
        for (slice, _) in protein_scope:
            end_idx = start_idx + slice  # 计算每个切片的结束位置
            res_vecs.append(xt[start_idx:end_idx, :])  # 切片并添加到列表中
            start_idx = end_idx  # 更新起始索引

        max_node = max(d.size(0) for d in res_vecs)
        adaptive_mxapool = nn.AdaptiveMaxPool1d(max_node)
        adaptive_avgpool = nn.AdaptiveAvgPool1d(max_node)
        maxpooled_graphs = torch.stack(
            [adaptive_mxapool(g.unsqueeze(0).transpose(1, 2)).squeeze(0).transpose(0, 1) for g in res_vecs], dim=0)
        avgpooled_graphs = torch.stack(
            [adaptive_avgpool(g.unsqueeze(0).transpose(1, 2)).squeeze(0).transpose(0, 1) for g in res_vecs], dim=0)
        xt = torch.cat((maxpooled_graphs,avgpooled_graphs),2)  # 132
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # graph Cross
        adaptive_mxapool = nn.AdaptiveMaxPool1d(1)
        xt_sum = adaptive_mxapool(xt.permute(0, 2, 1)).permute(0, 2, 1)

        adaptive_mxapool = nn.AdaptiveMaxPool1d(1)
        x_sum = adaptive_mxapool(x.permute(0, 2, 1)).permute(0, 2, 1)

        _, weight_smtop, _ = self.xtoxt(xt_sum, x)
        x = x * weight_smtop

        _, weight_ptosm, _ = self.xttox(x_sum, xt)
        xt = xt * weight_ptosm

        x = adaptive_mxapool(x.permute(0, 2, 1)).squeeze(2)
        xt = adaptive_mxapool(xt.permute(0, 2, 1)).squeeze(2)

        # squence
        seq = torch.stack(seq, dim=0) # 128,1000
        seqs = self.seq_encoder(seq) # 128,1000,256
        seqs = self.protein_input_fc(seqs)  # B * tar_len * lstm_dim 128,1000,128
        seqs = self.enhance2(seqs)

        # smile
        smile = torch.stack(smile, dim=0)
        smiles = self.smile_encoder(smile)
        smiles = self.smiles_input_fc(smiles)  # B * seq len * lstm_dim
        smiles = self.enhance1(smiles)

        # drugs and proteins BiLSTM
        smiles, _ = self.smiles_lstm(smiles)  # B * seq len * lstm_dim*2
        smiles = self.ln1(smiles)
        seqs, _ = self.protein_lstm(seqs)  # B * tar_len * lstm_dim *2
        seqs = self.ln2(seqs)

        # Cross smile and squence
        adaptive_mxapool = nn.AdaptiveMaxPool1d(1)
        seqs_sum = adaptive_mxapool(seqs.permute(0, 2, 1)).permute(0, 2, 1)

        adaptive_mxapool = nn.AdaptiveMaxPool1d(1)
        smiles_sum = adaptive_mxapool(smiles.permute(0, 2, 1)).permute(0, 2, 1)

        _, weight_smtop, _ = self.smiletopro(seqs_sum, smiles)
        smiles = smiles * weight_smtop

        _, weight_ptosm, _ = self.protosmile(smiles_sum, seqs)
        seqs = seqs * weight_ptosm

        smile_outputs = adaptive_mxapool(smiles.permute(0, 2, 1)).squeeze(2)
        seq_outputs = adaptive_mxapool(seqs.permute(0, 2, 1)).squeeze(2)

        # graph concat
        gra = torch.cat((x, xt), 1)
        gra = self.graphfc1(gra)
        gra = self.relu(gra)
        gra = self.dropout(gra)
        gra = self.graphfc2(gra)
        gra = self.relu(gra)
        gra = self.dropout(gra)
        gra = self.graphout(gra)

        # llm concat
        llm = torch.cat((esm, fcfp), 1)
        llm = self.llmfc1(llm)
        llm = self.relu(llm)
        llm = self.dropout(llm)
        llm = self.llmfc2(llm)
        llm = self.relu(llm)
        llm = self.dropout(llm)
        llm = self.llmout(llm)

         # smile and squence concat
        squ = torch.cat((smile_outputs, seq_outputs), 1)
        squ = self.squfc1(squ)
        squ = self.relu(squ)
        squ = self.dropout(squ)
        squ = self.squfc2(squ)
        squ = self.relu(squ)
        squ = self.dropout(squ)
        squ = self.squout(squ)

        # fusion
        llmsqu = torch.cat((llm, squ),1)
        llmsqu = self.llmsqufc1(llmsqu)
        llmsqu = self.relu(llmsqu)
        llmsqu = self.dropout(llmsqu)
        llmsqu = self.llmsqufc2(llmsqu)
        llmsqu = self.relu(llmsqu)
        llmsqu = self.dropout(llmsqu)
        llmsqu = self.llmsquout(llmsqu)

        #
        llmgra = torch.cat((llm, gra),1)
        llmgra = self.llmgrafc1(llmgra)
        llmgra = self.relu(llmgra)
        llmgra = self.dropout(llmgra)
        llmgra = self.llmgrafc2(llmgra)
        llmgra = self.relu(llmgra)
        llmgra = self.dropout(llmgra)
        llmgra = self.llmgraout(llmgra)

        # total concat
        xc = torch.cat((llmgra, llmsqu), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))

        return out

class Seq_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_emb = ConvEmbedding(seq_vocab_size, embedding_size, 'seq')

    def forward(self, seq_input):
        output_emb = self.seq_emb(seq_input)
        return output_emb

class Smile_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.smile_emb = ConvEmbedding(smile_vocab_size, embedding_size, 'smile')

    def forward(self, smile_input):
        output_emb = self.smile_emb(smile_input)
        return output_emb

class ConvEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, type):
        super().__init__()
        if type == 'seq':
            self.embed = nn.Embedding(vocab_size, embedding_size)
        elif type == 'smile':
            self.embed = nn.Embedding(vocab_size, embedding_size)

    def forward(self, inputs):
        output_emb = self.embed(inputs)
        return output_emb

embedding_size = 256
seq_vocab_size = 25
smile_vocab_size = 64

from torch.nn.parameter import Parameter

class GroupingUpdateModule(nn.Module):
    def __init__(self, groups=32):
        super(GroupingUpdateModule, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1))
        self.bias = Parameter(torch.ones(1, groups, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):  # (b, l, )
        b, c, h = x.size()
        x = x.view(b * self.groups, -1, h)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h)
        x = x * self.sig(t)
        x = x.view(b, c, h)
        return x

class Attention(nn.Module):
    def __init__(self, weight_dim, feature_dim):
        super().__init__()
        self.w = nn.Parameter(torch.rand(feature_dim, weight_dim))
        self.b = nn.Parameter(torch.zeros(weight_dim))
        self.W_attention = nn.Linear(feature_dim, weight_dim)

    def forward(self, sum_input, weight_output):
        h = torch.relu(self.W_attention(sum_input))
        hs = torch.relu(self.W_attention(weight_output))
        weight_ini = torch.matmul(h, hs.permute(0, 2, 1))
        weight = torch.sigmoid(torch.matmul(h, hs.permute(0, 2, 1))).permute(0, 2, 1)
        h_output = weight * hs
        return h_output, weight, weight_ini
