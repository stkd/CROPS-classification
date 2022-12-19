import numpy as np
import torch

from sklearn.metrics import confusion_matrix


def wp_score(y_pred, y_true):
    eps=1e-20
    f1_dict = {}
    precision_list = []
    TP_list = []
    FN_list = []
    
    y_pred = torch.argmax(y_pred, dim=1)

    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    confusion = confusion_matrix(y_true, y_pred)

    for i in range(len(confusion)):
        TP = confusion[i, i]
        FP = sum(confusion[:, i]) - TP
        FN = sum(confusion[i, :]) - TP

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        result_f1 = 2 * precision  * recall / (precision + recall + eps)

        TP_list.append(TP)
        FN_list.append(FN)
        f1_dict[i] = result_f1
        precision_list.append(precision)

    total_image = y_pred.shape[0]
    weighted = 0.
    for i in range(len(confusion)):
        weighted += precision_list[i] * (TP_list[i] + FN_list[i])

    WP = weighted / total_image

    return f1_dict, WP