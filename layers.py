import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class AtomBlock(torch.nn.Module):
    # gcn + bi gru + attention
    def __init__(self, in_dim, h_dim1, h_dim2, num_layer, bidirectional):
        super(AtomBlock, self).__init__()
        self.embedding = nn.Linear(in_dim, 256)
        self.gcn = SAGEConv(256, h_dim1)
        self.gru = nn.GRU(h_dim1, h_dim2, num_layer, bidirectional=bidirectional)
        self.att = nn.Linear(h_dim2 * 2, h_dim2 * 2)
        self.v = nn.Linear(h_dim2 * 2, 1)

    def forward(self, input_node, input_edge):
        out = self.embedding(input_node)
        out = self.gcn(out, input_edge)
        out = torch.unsqueeze(out, 1)
        out, _ = self.gru(out)
        shapes = out.shape
        out = out.view(shapes[0] * shapes[1], -1)
        att = self.v(torch.tanh(self.att(out)))
        att_score = F.softmax(att, dim=1)
        scored_out = out * att_score
        return scored_out


class ResidueBlock(torch.nn.Module):
    # gcn + bi gru + attention
    def __init__(self, in_dim, h_dim1, h_dim2, num_layer, bidirectional):
        super(ResidueBlock, self).__init__()
        self.gcn = SAGEConv(in_dim, h_dim1)
        self.gru = nn.GRU(h_dim1, h_dim2, num_layer, bidirectional=bidirectional)
        self.att = nn.Linear(h_dim2 * 2, h_dim2 * 2)
        self.v = nn.Linear(h_dim2 * 2, 1)

    def forward(self, input_node, input_edge):
        out = self.gcn(input_node, input_edge)
        out = torch.unsqueeze(out, 1)
        out, _ = self.gru(out)
        shapes = out.shape
        out = out.view(shapes[0] * shapes[1], -1)
        att = self.v(torch.tanh(self.att(out)))
        att_score = F.softmax(att, dim=1)
        scored_out = out * att_score
        return scored_out


if __name__ == '__main__':
    import pickle
    from configs import DefaultConfig
    configs = DefaultConfig()

    train_path = r'./data/train352-r5.5-a2.3.pkl'

    # data_context
    with open(train_path, 'rb') as f:
        train_list = pickle.load(f)
        test_protein = train_list[3]

    # print(test_protein.keys())
    p_a_node = torch.tensor(test_protein['atom_graph_node'], dtype=torch.float).cuda()
    p_a_edge = torch.tensor(test_protein['atom_graph_edge'], dtype=torch.long).cuda()

    p_r_node = torch.tensor(test_protein['residue_graph_node'], dtype=torch.float).cuda()
    p_r_edge = torch.tensor(test_protein['residue_graph_edge'], dtype=torch.long).cuda()

    m = ResidueBlock(configs.residue_feature_dim, configs.gcn_hidden_dim, configs.gru_hidden_dim, configs.gru_layers, configs.bidirectional).cuda()
    o = m(p_r_node, p_r_edge)
    print(o.shape)

    m = AtomBlock(configs.atom_feature_dim, configs.gcn_hidden_dim, configs.gru_hidden_dim, configs.gru_layers, configs.bidirectional).cuda()
    o = m(p_a_node, p_a_edge)
    print(o.shape)