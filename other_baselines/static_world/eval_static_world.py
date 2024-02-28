import numpy as np
import os
import copy
from tqdm import trange

# Setups =================================================================================================
test_idx_dir = "../../data/cam4docc/test_ids/"
test_results_dir = "../../data/cam4docc/results/"
gt_dir = "../../data/cam4docc/segmentation/"

objects_max_label = 9

test_seqs = os.listdir(test_idx_dir)
test_segmentations = os.listdir(test_results_dir)
dimension = [512, 512, 40]
future_ious = [0, 0, 0, 0]
# ========================================================================================================

def cm_to_ious(cm):
    mean_ious = []
    cls_num = len(cm)
    for i in range(cls_num):
        tp = cm[i, i]
        p = cm[:, i].sum()
        g = cm[i, :].sum()
        union = p + g - tp
        mean_ious.append(tp / union)
    return mean_ious

def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    iou_per_pred = (bin_count[-1]/(bin_count[-1]+bin_count[1]+bin_count[2]))
    return bin_count[:max_label ** 2].reshape(max_label, max_label),iou_per_pred

for i in trange(len(test_seqs)):
    segmentation_file = test_results_dir + test_seqs[i]
    if not os.path.exists(segmentation_file):
        continue
    segmentation = np.load(segmentation_file,allow_pickle=True)['arr_0']
    test_seqs_idxs = np.load(os.path.join(test_idx_dir, test_seqs[i]))["arr_0"]
    gt_segmentation_file = os.path.join(gt_dir, test_seqs[i])
    gt_segmentation_seqs = np.load(gt_segmentation_file,allow_pickle=True)['arr_0']

    # hard coding for input:3 output:4
    for t in range(3,7):
        # static world using present predictions
        segmentation_t = segmentation[0]
        pred_segmentation = np.zeros((dimension[0], dimension[1], dimension[2]))
        for j in range(segmentation_t.shape[0]):
            cur_ind = segmentation_t[j, :3].astype(int)
            cur_label = segmentation_t[j, -1]
            pred_segmentation[cur_ind[0],cur_ind[1],cur_ind[2]] = cur_label

        gt_segmentation = np.zeros((dimension[0], dimension[1], dimension[2]))
        gt_segmentation_raw = gt_segmentation_seqs[t]
        for k in range(gt_segmentation_raw.shape[0]):
            cur_ind = gt_segmentation_raw[k, :3].astype(int)
            cur_label = gt_segmentation_raw[k, -1]
            gt_segmentation[cur_ind[0],cur_ind[1],cur_ind[2]] = cur_label
        hist_cur, iou_per_pred = fast_hist(pred_segmentation.astype(int), gt_segmentation.astype(int), max_label=objects_max_label)
        if t <= 3:
            future_ious[0] = future_ious[0] + hist_cur
        if t <= 4:
            future_ious[1] = future_ious[1] + hist_cur
        if t <= 5:
            future_ious[2] = future_ious[2] + hist_cur
        if t <= 6:
            future_ious[3] = future_ious[3] + hist_cur

for t in range(len(future_ious)):
    print("iou for step "+str(t), cm_to_ious(future_ious[t]))