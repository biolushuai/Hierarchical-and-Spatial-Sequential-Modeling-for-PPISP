class DefaultConfig(object):
    train_dataset_path = r'./data/train352-r5.5-a2.3.pkl'
    test_dataset_path = r'./data/test70-r5.5-a2.3.pkl'
    save_path = r'./models_saved'

    epochs = 100
    learning_rate = 0.001
    weight_decay = 5e-4
    dropout_rate = 0.2
    neg_wt = 0.1
    split_rate = 0.8

    # block
    atom_feature_dim = 37
    residue_feature_dim = 1024

    # gcn
    gcn_hidden_dim = 512

    # gru
    gru_hidden_dim = 128
    gru_layers = 1
    bidirectional = True

    # mlp
    mlp_dim = 128

    early_stop = False

    seeds = [649737, 395408, 252356, 343053, 743746]