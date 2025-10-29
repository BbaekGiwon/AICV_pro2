from tqdm import tqdm
import numpy as np
from tabulate import tabulate

smooth = 1e-7
def resultConf(test_label, predict_model, weight):
    recall_list = []
    specificity_list = []
    precision_list = []
    acc_list = []
    dice_list = []
    
    for i in tqdm(range(test_label.shape[0])):
            
        gt = test_label[i] # ground truth binary mask
        pr = predict_model[i] > weight# binary prediction
        
        gt = gt.astype(bool)
        pr = pr.astype(bool)
        
        if np.sum(gt)== 0:
            continue
        seg1_n = pr == 0
        seg1_t = pr == 1
        
        gt1_n = gt == 0
        gt1_t = gt == 1
        
        tp = np.sum(seg1_t&gt1_t)
        fp = np.sum(seg1_t&gt1_n)
        tn = np.sum(seg1_n&gt1_n)    
        fn = np.sum(seg1_n&gt1_t)

        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)
        dice = (2 * tp) / (2*tp + fp + fn)
        
        recall_list.append(recall)
        specificity_list.append(specificity)
        precision_list.append(precision)
        acc_list.append(acc)
        dice_list.append(dice)

    dice_list = np.array(dice_list)
    recall_list = np.array(recall_list)
    specificity_list = np.array(specificity_list)
    precision_list = np.array(precision_list)
    
    dice_list = dice_list[~np.isnan(dice_list)]
    recall_list = recall_list[~np.isnan(recall_list)]
    specificity_list = specificity_list[~np.isnan(specificity_list)]
    precision_list = precision_list[~np.isnan(precision_list)]
    
    return recall_list, specificity_list, precision_list, acc_list, dice_list


def evaluation(label_arrays, pred_arrays, n_classes, weight=0.5):

    recall_list = []
    specificity_list = []
    precision_list = []
    acc_list = []
    dice_list = []
    for c in range(n_classes):
        lab = (label_arrays[...,c] > 0).astype(int)
        prd = (pred_arrays[...,c] > 0).astype(int)
        
        recall, specificity, precision, acc, dice = resultConf(lab, prd, weight)
        recall_list.extend(recall)
        specificity_list.extend(specificity)
        precision_list.extend(precision)
        acc_list.extend(acc)
        dice_list.extend(dice)
        
    row_data1 = []
    row_item1 = []

    row_item1.append('{0}'.format(0.5))
    row_item1.append('{0:.2f}'.format(np.mean(recall_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(specificity_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(precision_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(acc_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(dice_list)*100))
    row_data1.append(row_item1)

    head1 = ["Weight", "Sensitivity", "Specificity", "Precision", "Accuracy", "DSC"]
    print(tabulate(row_data1, headers=head1, tablefmt="grid"))
    print('')