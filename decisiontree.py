from math import log
import operator

'''
(data sample)
dataset = [feat_vector1, feat_vector2,...]
feat_vector1 = [1, 1, 'c1']
feat_vector2 = [1, 0, 'c2']
You can try,
datasets = [[1,1,'Fish'],[1,1,'Fish'],[1,0,'no Fish'],[0,1,'no Fish'],[0,1,'no Fish']]
labels = ['no surfacing', 'flippers']
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


def vote_majority(class_lists):
    class_counts = dict()
    for class_name in class_lists:
        dict[class_name] = class_counts.get(class_name, 0) + 1
    sorted_class_counts = sorted(class_counts.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_counts[0][0]


def create_tree(datasets, labels):
    class_list = [feat_vec[-1] for feat_vec in datasets]
    if len(set(class_list)) is 1:
        return class_list[0]
    if len(datasets[0]) is 1:
        return vote_majority(class_list)
    idx_choose = choose_best_feat(datasets)
    feat_choose = labels[idx_choose]
    del (labels[idx_choose])
    tree = {feat_choose: {}}
    feat_val_list = set([feat_vec[idx_choose] for feat_vec in datasets])
    for feat_val in feat_val_list:
        sub_labels = labels[:]
        tree[feat_choose][feat_val] = create_tree(split_by_feat_val(datasets, idx_choose, feat_val), sub_labels)
    return tree