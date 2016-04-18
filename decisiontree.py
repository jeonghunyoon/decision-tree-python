from math import log

'''
(data sample)
dataset = [feat_vector1, feat_vector2,...]
feat_vector1 = [1, 1, 'c1']
feat_vector2 = [1, 0, 'c2']
'''


def calc_shannon_entropy(datasets):
    label_counts = dict()
    for feat_vector in datasets:
        current_label = feat_vector[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1
    shannon_entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / len(datasets)
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def split_by_feat_val(datasets, idx, val):
    split_datasets = []
    for feat_vector in datasets:
        if feat_vector[idx] is val:
            split_datasets.append(feat_vector)
    return split_datasets
