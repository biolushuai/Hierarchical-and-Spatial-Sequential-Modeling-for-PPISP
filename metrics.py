import numpy as np
from sklearn.metrics import *


def compute_acc(labels, preds):
    acc = accuracy_score(labels, preds)
    return acc


def compute_auc_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    auc_roc = auc(fpr, tpr)
    return auc_roc


def compute_auc_pr(labels, preds):
    p, r, _ = precision_recall_curve(labels, preds)
    auc_pr = auc(r, p)
    return auc_pr


def compute_performance(labels, preds):
    confusion = confusion_matrix(labels, preds)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    mcc = matthews_corrcoef(labels, preds)
    recall = recall_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    f1_score = (2 * recall * precision) / (recall + precision)
    sensitivity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)
    return mcc, recall, precision, f1_score, sensitivity, specificity
