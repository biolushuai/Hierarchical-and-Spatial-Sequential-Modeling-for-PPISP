import os
import csv
import torch
import pickle
from metrics import *
from configs import DefaultConfig
from models import HSSPPI

configs = DefaultConfig()


def test(model, test_proteins):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    all_labels = []
    all_preds = []

    all_auc_roc = []
    all_auc_pr = []

    all_acc = []
    all_mcc = []
    all_f_score = []
    all_precision = []
    all_recall = []
    all_sc = []
    all_sp = []

    for test_p in test_proteins:
        # print(test_g['protein_id'])
        if torch.cuda.is_available():
            test_p_a_node = torch.FloatTensor(test_p['atom_graph_node']).cuda()
            test_p_a_edge = torch.LongTensor(test_p['atom_graph_edge']).cuda()
            test_p_r_node = torch.FloatTensor(test_p['residue_graph_node']).cuda()
            test_p_r_edge = torch.LongTensor(test_p['residue_graph_edge']).cuda()
            test_p_label = torch.LongTensor(test_p['label'])
            test_p_ar_map = torch.tensor(test_p['a2r_map']).cuda()

        # p_preds = model(test_p_r_node, test_p_r_edge)
        # p_preds = model(test_p_a_node, test_p_a_edge, test_p_r_node, test_p_ar_map)
        p_preds = model(test_p_a_node, test_p_a_edge, test_p_r_node, test_p_r_edge, test_p_ar_map)
        p_preds = p_preds.data.cpu().numpy()
        test_p_label = test_p_label.numpy()

        all_labels.append(test_p_label)
        all_preds.append(p_preds)

        g_auc_roc = compute_auc_roc(test_p_label, p_preds)
        g_auc_pr = compute_auc_pr(test_p_label, p_preds)
        all_auc_pr.append(g_auc_pr)
        all_auc_roc.append(g_auc_roc)

    # 最佳阈值下的模型性能
    y_test = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    # print(len(y_test), len(y_pred))

    w_auc_roc = compute_auc_roc(y_test, y_pred)
    w_auc_pr = compute_auc_pr(y_test, y_pred)
    # print(w_auc_roc, w_auc_pr)

    thresholds = np.arange(0.0, 1.0, 0.001)
    mcc = np.zeros(shape=(len(thresholds)))

    # 拟合模型
    for index, elem in enumerate(thresholds):
        y_pred_prob = (y_pred > elem).astype('int')
        mcc[index] = matthews_corrcoef(y_test, y_pred_prob)

    # 查找最佳阈值
    index = np.argmax(mcc)
    thresholdOpt = round(thresholds[index], ndigits=4)
    mccOpt = round(mcc[index], ndigits=4)
    print('Best Threshold: {} with MCC: {}'.format(thresholdOpt, mccOpt))

    y_pred[y_pred >= thresholdOpt] = 1
    y_pred[y_pred < thresholdOpt] = 0
    w_acc = compute_acc(y_test, y_pred)
    _, w_recall, w_precision, w_f1_score, w_sc, w_sp = compute_performance(y_test, y_pred)

    for (p_pred, p_label, protein) in zip(all_preds, all_labels, test_proteins):
        p_pred[p_pred >= thresholdOpt] = 1
        p_pred[p_pred < thresholdOpt] = 0

        g_acc = compute_acc(p_label, p_pred)
        g_mcc, g_recall, g_precision, g_f1_score, g_sc, g_sp = compute_performance(p_label, p_pred)

        all_acc.append(g_acc)
        all_recall.append(g_recall)
        all_precision.append(g_precision)
        all_mcc.append(g_mcc)
        all_f_score.append(g_f1_score)
        all_sc.append(g_sc)
        all_sp.append(g_sp)

    all_results = [np.mean(all_auc_roc), np.mean(all_auc_pr), np.mean(all_acc), np.mean(all_mcc), np.mean(all_f_score),
                   np.mean(all_precision), np.mean(all_recall), np.mean(all_sc), np.mean(all_sp), thresholdOpt]

    all_w_results = [w_auc_roc, w_auc_pr, w_acc, mccOpt, w_f1_score, w_precision, w_recall, w_sc, w_sp, thresholdOpt]

    # return all_auc_roc, all_auc_pr, all_results, all_w_results
    return all_results, all_w_results


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # a_dist = [1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
    # r_dist = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

    a_dist = [2.3]
    r_dist = [5.5]

    for a_d in a_dist:
        for r_d in r_dist:
            models_saved_path = r'./models_saved'
            # models_saved_path = os.path.join(r'./models_saved', 'test')
            models = os.listdir(models_saved_path)
            test_path = os.path.join(r'./data', 'test70-r' + str(r_d) + '-a' + str(a_d) + '.pkl')


            with open(test_path, 'rb') as f:
                test_data = pickle.load(f)

            test_model = HSSPPI(configs.atom_feature_dim, configs.residue_feature_dim, configs.gcn_hidden_dim,
                                configs.gru_hidden_dim, configs.gru_layers, configs.bidirectional).cuda()

            with open(os.path.join(models_saved_path, 'average_results.csv'), 'w', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['auc_roc', 'auc_pr', 'acc', 'mcc_max', 'f1', 'p', 'r', 'sc', 'sp', 't_max'])

            with open(os.path.join(models_saved_path, 'whole_results.csv'), 'w', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['auc_roc', 'auc_pr', 'acc', 'mcc_max', 'f1', 'p', 'r', 'sc', 'sp', 't_max'])
            num = 1
            for m in models:
                if '.tar' in m:
                    test_model_path = os.path.join(models_saved_path, m)
                    print(test_model_path)
                    test_model_sd = torch.load(test_model_path)
                    test_model.load_state_dict(test_model_sd)
                    if torch.cuda.is_available():
                        model = test_model.cuda()

                    experiment_results, whole_results = test(test_model, test_data)

                    with open(os.path.join(models_saved_path, 'average_results.csv'), 'a+', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(experiment_results)

                    with open(os.path.join(models_saved_path, 'whole_results.csv'), 'a+', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(whole_results)
                    num += 1