import torch

# SR : Segmentation Result
# GT : Ground Truth
smooth = 1e-6

def get_accuracy(SR_total,GT_total,threshold=0.5, num_classes=3):
    per_class = torch.zeros((len(SR_total),num_classes)).to(SR_total.device)
    for i in range(len(SR_total)):
        for class_id in range(num_classes):
            SR = SR_total[i, class_id, ...]
            GT = GT_total[i, class_id, ...]

            SR = SR > threshold
            GT = GT.bool()
            
            corr = torch.sum(SR==GT)
            tensor_size = SR.size(0)*SR.size(1)
            acc = float(corr)/float(tensor_size)
            per_class[i,class_id] = acc
    return per_class.mean()

def get_sensitivity(SR_total,GT_total,threshold=0.5, num_classes=3):
    per_class = torch.zeros((len(SR_total),num_classes)).to(SR_total.device)
    for i in range(len(SR_total)):
        for class_id in range(num_classes):
            SR = SR_total[i, class_id, ...]
            GT = GT_total[i, class_id, ...]
            # Sensitivity == Recall
            SR = SR > threshold
            GT = GT.bool()
            # TP : True Positive
            # FN : False Negative
            # TP = ((SR==1)+(GT==1))==2
            # FN = ((SR==0)+(GT==1))==2
            TP = (SR & GT)
            FN = (~SR & GT)
            SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + smooth)     
            per_class[i,class_id] = SE
    return per_class.mean()

def get_specificity(SR_total,GT_total,threshold=0.5, num_classes=3):
    per_class = torch.zeros((len(SR_total),num_classes)).to(SR_total.device)
    for i in range(len(SR_total)):
        for class_id in range(num_classes):
            SR = SR_total[i, class_id, ...]
            GT = GT_total[i, class_id, ...]
            
            SR = SR > threshold
            GT = GT.bool()

            # TN : True Negative
            # FP : False Positive
            # TN = ((SR==0)+(GT==0))==2
            # FP = ((SR==1)+(GT==0))==2
            TN = (~SR & ~GT)
            FP = (SR & ~GT)
            SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + smooth)
            
            per_class[i,class_id] = SP
    return per_class.mean()


def get_precision(SR_total,GT_total,threshold=0.5, num_classes=3):
    per_class = torch.zeros((len(SR_total),num_classes)).to(SR_total.device)
    for i in range(len(SR_total)):
        for class_id in range(num_classes):
            SR = SR_total[i, class_id, ...]
            GT = GT_total[i, class_id, ...]

            SR = SR > threshold
            GT = GT.bool()

            # TP : True Positive
            # FP : False Positive
            # TP = ((SR==1)+(GT==1))==2
            # FP = ((SR==1)+(GT==0))==2
            TP = (SR & GT)
            FP = (SR & ~GT)
            PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + smooth)
            per_class[i,class_id] = PC
    return per_class.mean()

def get_F1(SR_total,GT_total,threshold=0.5, num_classes=3):
    
    # Sensitivity == Recall
    SE = get_sensitivity(SR_total,GT_total,threshold=threshold,num_classes=num_classes)
    PC = get_precision(SR_total,GT_total,threshold=threshold,num_classes=num_classes)

    F1 = 2*SE*PC/(SE+PC + smooth)

    return F1

def get_JS(SR_total,GT_total,threshold=0.5, num_classes=3):
    per_class = torch.zeros((len(SR_total),num_classes)).to(SR_total.device)
    for i in range(len(SR_total)):
        for class_id in range(num_classes):
            SR = SR_total[i, class_id, ...]
            GT = GT_total[i, class_id, ...]
            # JS : Jaccard similarity
            SR = SR > threshold
            GT = GT.bool()

            Inter = torch.sum((SR & GT))
            Union = torch.sum((SR | GT))

            JS = float(Inter)/(float(Union) + smooth)
            per_class[i,class_id] = JS
    return per_class.mean()

def get_DC(SR_total,GT_total,threshold=0.5, num_classes=3):
    per_class = torch.zeros((len(SR_total),num_classes)).to(SR_total.device)
    for i in range(len(SR_total)):
        for class_id in range(num_classes):
            SR = SR_total[i, class_id, ...]
            GT = GT_total[i, class_id, ...]
            # DC : Dice Coefficient
            SR = SR > threshold
            GT = GT.bool()

            Inter = torch.sum((SR & GT))
            DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + smooth)
            per_class[i,class_id] = DC
    return per_class.mean()



