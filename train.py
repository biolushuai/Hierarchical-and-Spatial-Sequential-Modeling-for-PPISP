import os
import torch
import numpy as np
import pickle
from configs import DefaultConfig
from losses import WeightedCrossEntropy
from models import HSSPPI

configs = DefaultConfig()


def train(model, train_proteins, val_proteins, model_save_path, num=1):
    epochs = configs.epochs
    lr = configs.learning_rate
    weight_decay = configs.weight_decay
    neg_wt = configs.neg_wt

    print(model_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay, nesterov=True)
    loss_fn = WeightedCrossEntropy(neg_wt=neg_wt, device=device)

    model.train()
    train_losses = []
    val_losses = []
    best_loss = 999
    count = 0
    for e in range(epochs):
        print("Runing {} epoch".format(e + 1))
        e_loss = 0.
        for train_p in train_proteins:
            # print(train_p['complex_code'])
            if torch.cuda.is_available():
                train_p_a_node = torch.FloatTensor(train_p['atom_graph_node']).cuda()
                train_p_a_edge = torch.LongTensor(train_p['atom_graph_edge']).cuda()
                train_p_r_node = torch.FloatTensor(train_p['residue_graph_node']).cuda()
                train_p_r_edge = torch.LongTensor(train_p['residue_graph_edge']).cuda()
                train_p_label = torch.LongTensor(train_p['label']).cuda()
                train_p_ar_map = torch.tensor(train_p['a2r_map']).cuda()
            # else:
            #     train_p_a_node = torch.FloatTensor(train_p['atom_graph_node'])
            #     train_p_a_edge = torch.FloatTensor(train_p['atom_graph_edge'])
            #     train_p_r_node = torch.FloatTensor(train_p['residue_graph_node'])
            #     train_p_r_edge = torch.FloatTensor(train_p['residue_graph_edge'])
            #     train_p_label = torch.LongTensor(train_p['label'])

            optimizer.zero_grad()
            # batch_pred = model(train_p_r_node, train_p_r_edge)
            # batch_pred = model(train_p_a_node, train_p_a_edge, train_p_r_node, train_p_ar_map)
            batch_pred = model(train_p_a_node, train_p_a_edge, train_p_r_node, train_p_r_edge, train_p_ar_map)
            batch_loss = loss_fn.computer_loss(batch_pred, train_p_label)

            b_loss = batch_loss.item()
            e_loss += b_loss
            batch_loss.backward()
            optimizer.step()

        e_loss /= len(train_proteins)
        train_losses.append(e_loss)
        with open(os.path.join(model_save_path, 'train_losses{}.txt'.format(num)), 'a+') as f:
            f.write(str(e_loss) + '\n')

        e_loss = 0.
        for val_p in val_proteins:
            if torch.cuda.is_available():
                val_p_a_node = torch.FloatTensor(val_p['atom_graph_node']).cuda()
                val_p_a_edge = torch.LongTensor(val_p['atom_graph_edge']).cuda()
                val_p_r_node = torch.FloatTensor(val_p['residue_graph_node']).cuda()
                val_p_r_edge = torch.LongTensor(val_p['residue_graph_edge']).cuda()
                val_p_label = torch.LongTensor(val_p['label']).cuda()
                val_p_ar_map = torch.tensor(val_p['a2r_map']).cuda()
            # else:
            #     val_p_a_node = torch.FloatTensor(val_p['atom_graph_node'])
            #     val_p_a_edge = torch.FloatTensor(val_p['atom_graph_edge'])
            #     val_p_r_node = torch.FloatTensor(val_p['residue_graph_node'])
            #     val_p_r_edge = torch.FloatTensor(val_p['residue_graph_edge'])
            #     val_p_label = torch.LongTensor(val_p['label'])

            optimizer.zero_grad()
            # batch_pred = model(val_p_r_node, val_p_r_edge)
            # batch_pred = model(val_p_a_node, val_p_a_edge, val_p_r_node, val_p_ar_map)
            batch_pred = model(val_p_a_node, val_p_a_edge, val_p_r_node, val_p_r_edge, val_p_ar_map)
            batch_loss = loss_fn.computer_loss(batch_pred, val_p_label)

            b_loss = batch_loss.item()
            e_loss += b_loss

        e_loss /= len(val_proteins)
        val_losses.append(e_loss)
        with open(os.path.join(model_save_path, 'val_losses{}.txt'.format(num)), 'a+') as f:
            f.write(str(e_loss) + '\n')

        if best_loss > val_losses[-1]:
            count = 0
            torch.save(model.state_dict(), os.path.join(os.path.join(model_save_path, "model{}.tar".format(num))))
            best_loss = val_losses[-1]
            print("UPDATE\tEpoch {}: train loss {}\tval loss {}".format(e + 1, train_losses[-1], val_losses[-1]))
        else:
            count += 1
            if configs.early_stop and count >= configs.early_stop:
                return None


if __name__ == '__main__':
    seeds = [649737, 395408, 252356, 343053, 743746]

    a_dist = [2.3]
    r_dist = [5.5]

    for a_d in a_dist:
        for r_d in r_dist:
            train_path = os.path.join(r'./data', 'train352-r' + str(r_d) + '-a' + str(a_d) + '.pkl')

            models_save_path = os.path.join(r'./models_saved', 'test')

            # data_context
            with open(train_path, 'rb') as f:
                train_list = pickle.load(f)
                train_list = train_list[:3]

            samples_num = len(train_list)
            split_num = int(configs.split_rate * samples_num)
            data_index = train_list
            np.random.shuffle(data_index)
            train_data = data_index[:split_num]
            val_data = data_index[split_num:]

            train_model = HSSPPI(configs.atom_feature_dim, configs.residue_feature_dim, configs.gcn_hidden_dim,
                                 configs.gru_hidden_dim, configs.gru_layers, configs.bidirectional).cuda()

            current_experiment = 1

            for seed_id, seed in enumerate(seeds):
                print('experiment:', current_experiment)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                torch.backends.cudnn.deterministic = True

                train(train_model, train_data, val_data, models_save_path, seed_id + 1)
                current_experiment += 1