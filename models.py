import torch
import torch.nn.functional as F
from torch import nn
from layers import AtomBlock, ResidueBlock
from configs import DefaultConfig
configs = DefaultConfig()


def atom2residue(atom_mat, residue_mat, a2r_map):
    if len(atom_mat) != len(a2r_map):
        return None
    else:
        new_atom_mat = torch.zeros((residue_mat.shape[0], atom_mat.shape[-1]))
        for a_id, a in enumerate(atom_mat):
            r_id = a2r_map[a_id]
            new_atom_mat[r_id] += a
        return new_atom_mat


class HSSPPI(torch.nn.Module):
    def __init__(self, a_in_dim, r_in_dim, h_dim1, h_dim2, num_layer, bi):
        super(HSSPPI, self).__init__()
        mlp_dim = configs.mlp_dim
        dropout_rate = configs.dropout_rate
        self.atom_block1 = AtomBlock(a_in_dim, h_dim1, h_dim2, num_layer, bi)
        self.residue_block1 = ResidueBlock(r_in_dim, h_dim1, h_dim2, num_layer, bi)

        if bi:
            self.atom_block2 = AtomBlock(h_dim2 * 2, h_dim1, h_dim2, num_layer, bi)
            self.residue_block2 = ResidueBlock(h_dim2 * 2, h_dim1, h_dim2, num_layer, bi)
        else:
            self.atom_block2 = AtomBlock(h_dim2, h_dim1, h_dim2, num_layer, bi)
            self.residue_block2 = ResidueBlock(h_dim2, h_dim1, h_dim2, num_layer, bi)

        if bi:
            h_dim2 = h_dim2 * 2

        self.linear1 = nn.Sequential(
            nn.Linear(h_dim2, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, a_input_node, a_input_edge, r_input_node, r_input_edge, a2r_map):
        skip = 0
        # block 1
        a_out1 = self.atom_block1(a_input_node, a_input_edge)
        r_out1 = self.residue_block1(r_input_node, r_input_edge)

        a_out_map1 = atom2residue(a_out1.cpu(), r_out1.cpu(), a2r_map).cuda()
        ar_out1 = r_out1 + a_out_map1
        skip += ar_out1
        # print('skip1', skip.shape, 'ar_out1', ar_out1.shape)

        # block 2
        a_out2 = self.atom_block2(a_out1, a_input_edge)
        r_out2 = self.residue_block2(ar_out1, r_input_edge)

        a_out_map2 = atom2residue(a_out2.cpu(), r_out2.cpu(), a2r_map).cuda()
        ar_out2 = r_out2 + a_out_map2
        skip += ar_out2
        # print('skip2', skip.shape, 'ar_out2', ar_out2.shape)

        # out
        out = self.linear1(skip)
        out = self.linear2(out)
        return out


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

    ar_map = torch.tensor(test_protein['a2r_map']).cuda()

    m = HSSPPI(configs.atom_feature_dim, configs.residue_feature_dim, configs.gcn_hidden_dim, configs.gru_hidden_dim,
               configs.gru_layers, configs.bidirectional).cuda()

    o = m(p_a_node, p_a_edge, p_r_node, p_r_edge, ar_map)
    print(o.shape)