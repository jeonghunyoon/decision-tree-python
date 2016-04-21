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
            reduced_feat_vec = feat_vector[:idx]
            reduced_feat_vec.extend(feat_vector[idx + 1:])
            split_datasets.append(reduced_feat_vec)
    return split_datasets


def choose_best_feat(datasets):
    num_of_feat = len(datasets[0]) - 1
    base_entropy = calc_shannon_entropy(datasets)
    base_info_gain = 0.0
    best_feat = -1
    for i in range(num_of_feat):
        feat_values = set([feat_vec[i] for feat_vec in datasets])
        new_entropy = 0.0
        for value in feat_values:
            sub_datasets = split_by_feat_val(datasets, i, value)
            prob = float(len(sub_datasets)) / len(datasets)
            new_entropy += prob * calc_shannon_entropy(sub_datasets)
        info_gain = base_entropy - new_entropy
        if (info_gain > base_info_gain):
            base_info_gain = info_gain
            best_feat = i
    return best_feat
