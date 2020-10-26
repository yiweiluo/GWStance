from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, classification_report
import pandas as pd

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "micro_f1": micro_f1,
        "macro_f1":macro_f1,
        "acc_and_macro_f1": (acc + macro_f1) / 2,
    }

def cm(preds, labels):
    return confusion_matrix(preds, labels)#, labels=['agree','neutral','disagree'])


def classif_report(preds, labels, target_names):
    return classification_report(labels, preds, labels=[0,1,2], target_names=target_names, output_dict=True)


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'climate' or task_name == 'climate-weight':
        return {'acc': simple_accuracy(preds, labels),
                'acc_and_f1': acc_and_f1(preds, labels),
                'cm': cm(preds, labels),
                'per_class': classif_report(preds, labels, target_names=["disagree", "neutral", "agree"])}
    else:
        raise KeyError(task_name)
