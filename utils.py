import torch
import os
import random
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings


def load_data(data_path, tokenizer=None, load_part=None):
    data = []
    num = 0
    with open(data_path) as f:
        for line in f.readlines():
            num += 1
            if load_part != None and num > load_part:
                break
            obj = json.loads(line)
            data.append(obj)
    
    texts = [D['text'] for D in data]
    # final_data = []
    if load_part != 0 and tokenizer != None:
        max_seq_len = max(tokenizer(texts, add_special_tokens=True, return_length=True)['length'])
        # for i, length in enumerate(tokenizer(texts, add_special_tokens=True, return_length=True)['length']):
            # if length > 512:
                # print(data[i]['id'], length)
            # else:
                # final_data.append(data[i])
        print('Max length of text in the corpus:', max_seq_len)
    # return final_data
    return data


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_id_label_map(labels):
    id2label = dict(enumerate(labels))
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(labels)
    return id2label, label2id, num_labels


# {'reasons': [[0, 2], ...]} -> ['none', 'reason_right', 'none', ...]
# turn just related edges to all edges of the argument graph included none-related edges
def generate_relation_labels(ARs, num_acs):
    relation_labels = []
    for i in range(num_acs):
        for j in range(num_acs):
            if i == j:
                continue
            is_existing = False
            for AR_type in ARs.keys():
                if [i, j] in ARs[AR_type]:
                    relation_labels.append(AR_type)
                    is_existing = True
                    break
            if not is_existing:
                relation_labels.append('none')
    return relation_labels


def process_data(data, label2id_ac, label2id_ar):
    new_data = []
    for D in data:
        relation_labels = generate_relation_labels(D['ARs'], len(D['ACs']))
        new_data.append({
            'ACs': [label2id_ac[ac] for ac in D['ACs']],
            'ARs': [label2id_ar[rel] for rel in relation_labels],
            'ACs_span': D['ACs_span'],
            'text': D['text']
        })
    return new_data


def print_performance(pred_comp, gold_comp, pred_rel, gold_rel, test_type='val', print_report=True):

    scores_comp = get_metric_scores(pred_comp, gold_comp)
    scores_rel_bool = get_metric_scores(type2bool(get_pred_rel(pred_rel, 'bool')), type2bool(gold_rel))
    scores_rel_type = get_metric_scores(*filter_non_rel(get_pred_rel(pred_rel, 'type'), gold_rel))
    f1_score_avg = (scores_comp['f1_score'] + scores_rel_bool['f1_score'] + scores_rel_type['f1_score']) / 3
    print(f'{test_type}_f1_score_comp', scores_comp['f1_score'])
    print(f'{test_type}_f1_score_rel_bool', scores_rel_bool['f1_score'])
    print(f'{test_type}_f1_score_rel_type', scores_rel_type['f1_score'])
    print(f'{test_type}_f1_score_avg', f1_score_avg)
    if print_report:
        print('-----------------------------------------------------------------------------')
        print(f'{test_type} components classification report:\n', scores_comp['report'])
        print(f'{test_type} relations identification report:\n', scores_rel_bool['report'])
        print(f'{test_type} relations classification report:\n', scores_rel_type['report'])
        print('-----------------------------------------------------------------------------')
    return scores_comp, scores_rel_bool, scores_rel_type, f1_score_avg


def get_metric_scores(pred, gold):
    pred = [p.cpu() for p in pred]
    gold = [g.cpu() for g in gold]
    scores = {}
    with warnings.catch_warnings(record=True) as w:
        scores['accuracy_score'] = accuracy_score(gold, pred)
        scores['f1_score'] = f1_score(gold, pred, average='macro')
        scores['precision_score'] = precision_score(gold, pred, average='macro')
        scores['recall_score'] = recall_score(gold, pred, average='macro')
        scores['report'] = classification_report(gold, pred, digits=3)
    return scores


# argmax_threshold = 0.05 # CDCP
argmax_threshold = 0.1 # PE
def threshold_argmax(cls_out, keepdim=False):
    cls_out[0] = 0
    # if cls_out[2] > argmax_threshold:
        # cls_out[1] = 0
    # for i in range(1, len(cls_out)):
        # if cls_out[i] > argmax_threshold:
            # cls_out[0] = 0
            # break
    return torch.argmax(cls_out, keepdim=keepdim)


def get_pred_rel(pred_rel, form):
    if form == 'bool':
        argmax = torch.argmax
    elif form == 'type':
        argmax = threshold_argmax
    new_pred_rel = []
    for p in pred_rel:
        new_pred_rel.append(argmax(p, keepdim=False))
    return new_pred_rel


def type2bool(type_tensor_list):
    bool_tensor_list = []
    type_of_tensor = type_tensor_list[0]
    for tt in type_tensor_list:
        if tt == 0.:
            bool_tensor_list.append(torch.tensor(0.).type_as(type_of_tensor))
        else:
            bool_tensor_list.append(torch.tensor(1.).type_as(type_of_tensor))
    return bool_tensor_list


def filter_non_rel(pred_rel, gold_rel):
    tmp_pred, tmp_gold = [], []
    for p, g in zip(pred_rel, gold_rel):
        if g != 0:
            tmp_pred.append(p)
            tmp_gold.append(g)
    return tmp_pred, tmp_gold