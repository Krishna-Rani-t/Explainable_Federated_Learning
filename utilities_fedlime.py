
import math
from sklearn.metrics import roc_auc_score, confusion_matrix


def all_metrics(y_true, y_pre):
    conf = confusion_matrix(y_true, y_pre.round())
    TN = conf[0][0]
    FP = conf[0][1]
    FN = conf[1][0]
    TP = conf[1][1]
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    BalanceACC = (sensitivity + specificity) / 2
    G_mean = math.sqrt(sensitivity * specificity)
    FN_rate = FN / (FN + TP)
    FP_rate = FP / (FP + TN)
    Precision = TP / (TP + FP)
    f1_sc = 2 * (sensitivity * Precision) / (sensitivity + Precision)
    acc = (TP + TN) / (TP + TN + FN + FP)
    auc = roc_auc_score(y_true, y_pre)
    
    return sensitivity, specificity, BalanceACC, G_mean, FN_rate, FP_rate, Precision, f1_sc, acc, auc
