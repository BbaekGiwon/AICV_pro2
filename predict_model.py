import os
import sys, cv2
sys.path.append('./pathSegmentation')
sys.path.append('./utils')

from glob import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
from scipy.ndimage import center_of_mass
import concurrent.futures
import time
from pathSeg.ml.hovernet import HoVerNet, post_process_batch_hovernet
import segmentation_models_pytorch as smp

import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from param import img_wid, img_hei
from param import class_list, data_dir, save_dir, save_dir_list
from param import model_path1, model_path2
from param import batch_size
from evaluation import evaluation

print(f"GPUs used:\t{torch.cuda.device_count()}")
# device = torch.device("cuda:3")
device = torch.device("cuda:0")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device:\t\t{device}")

for sd in save_dir_list:
    os.makedirs(sd, exist_ok=True)

class CustomDataset(Dataset):
    def __init__(self, image_list, label_list):
        self.img_path = image_list
        self.lab_path = label_list
        self.tf = ToTensor()

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        path = self.img_path[idx]
        image=self.tf(Image.open(self.img_path[idx]))
        label=self.tf(np.load(self.lab_path[idx]))
        return image, label, path


def model_load():
    # model 1: cell segmentation 
    model1 = HoVerNet(n_classes=None)
    model1.load_state_dict(torch.load(model_path1, map_location=device))
    model1.to(device)
    model1.eval()

    # model 2: class region segmentation 
    model2 = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(class_list),
    ).to(device)
    model2.load_state_dict(torch.load(model_path2, map_location=device))
    model2.to(device)
    model2.eval()

    return model1, model2

# one-hot to original
def post_process_batch_unet(outputs, n_classes=4):
    predict = F.softmax(outputs,dim=1)

    predict = predict.cpu().detach().numpy() > 0.5

    predicted_arr_data = []
    for i in range(len(predict)):
        predict_temp = np.zeros((img_wid, img_hei))
        for c in range(n_classes):
            predict_temp = np.where(predict[i,c] == 1, c, predict_temp)
        predicted_arr_data.append(predict_temp)
    return predicted_arr_data

def predict_model(dataloader, model1, model2):
    image_arrays = []
    label_arrays = []
    pred_arrays1 = []
    pred_arrays2 = []
    with torch.no_grad():
        for data in tqdm(dataloader):
                
            # send the data to the GPU
            images = data[0].float().to(device)
            labels = data[1].float().to(device)
            path = data[2]

            outputs1 = model1(images)
            outputs2 = model2(images).to(device)

            predict1 = post_process_batch_hovernet(outputs1, n_classes=None)
            predict2 = post_process_batch_unet(outputs2, len(class_list))

            # file_name = os.path.basename(path[0]).split('.')[0]
            # save_path1 = f'{save_dir}/predict_results_1/{file_name}.npy'
            # save_path2 = f'{save_dir}/predict_results_2/{file_name}.npy'
            
            # np.save(save_path1, predict1)
            # np.save(save_path2, predict2)
            

            image_arrays.extend(images.cpu().detach().numpy())
            label_arrays.extend(labels.cpu().detach().numpy())
            pred_arrays1.extend(predict1)
            pred_arrays2.extend(predict2)
    image_arrays = np.transpose(np.array(image_arrays), (0,2,3,1))
    label_arrays = np.transpose(np.array(label_arrays), (0,2,3,1))
    pred_arrays1 = np.array(pred_arrays1)
    pred_arrays2 = np.array(pred_arrays2)
    return image_arrays, label_arrays, pred_arrays1, pred_arrays2


class PostProcessing():
    def __init__(self, image_list, pred_arrays1, pred_arrays2):
        self.image_list = image_list
        self.cell__seg = pred_arrays1
        self.class_seg = pred_arrays2

    def process_single_image(self, index):
        image_path = self.image_list[index]
        file_name = os.path.basename(image_path).split('.png')[0]
        prd = self.cell__seg[index]
        lab = self.class_seg[index]

        cell_total = np.zeros((1024, 1024, len(class_list)))
        # for j in tqdm(range(1, np.max(prd)), leave=False):
        for j in range(1, int(np.max(prd))):
            prd_cell = (prd == j).astype(int)
            center = center_of_mass(prd_cell)  # center = (y,x)

            if np.isnan(center[0]) or np.isnan(center[1]):
                continue
            y = int(center[0])
            x = int(center[1])
            cell_class = int(lab[y, x])
            cell_total[..., cell_class] += prd_cell * j

        save_path = f'{save_dir}/seg_result_img/{file_name}.npy'
        np.save(save_path, cell_total)
        return cell_total


if __name__ == '__main__':
    start_time = time.time()
    print("Start Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-fn", "--filename", type=str, default='', help="선택적 파라미터 (기본값: '')")
    
    args = parser.parse_args()
    if args.filename == '':
        label_list = glob(f"{save_dir}/preprocessing/*.npy")
        image_list = [f"{data_dir}/image/{os.path.basename(path).split('.')[0]}.png" for path in label_list]
        print(f"{len(image_list)}, {len(label_list)}     파일 이름이 제공되지 않았습니다.")
    else:
        print(f"파일 이름: {args.filename}")
        label_list = [f"{save_dir}/preprocessing/{args.filename}.npy"]
        image_list = [f"{data_dir}/image/{args.filename}.png"]
        print(f"{len(image_list)}, {len(label_list)}     파일 이름: {args.filename}")
    
    ####################################################################################################
    dataset = CustomDataset(image_list, label_list)
    print('Done - CustomDataset')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print('Done - DataLoader')

    model1, model2 = model_load()
    print('Done - model_load')
    image_arrays, label_arrays, pred_arrays1, pred_arrays2 = predict_model(dataloader, model1, model2)
    print('Done - predict_model')
    
    print(image_arrays.shape, np.min(image_arrays), np.max(image_arrays))
    print(label_arrays.shape, np.min(label_arrays), np.max(label_arrays))
    print(pred_arrays1.shape, np.min(pred_arrays1), np.max(pred_arrays1))
    print(pred_arrays2.shape, np.min(pred_arrays2), np.max(pred_arrays2))

    pp = PostProcessing(image_list, pred_arrays1, pred_arrays2)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pred_arrays = list(tqdm(executor.map(pp.process_single_image, range(len(image_list))), total=len(image_list)))
    pred_arrays = np.array(pred_arrays)
    print(pred_arrays.shape, np.min(pred_arrays), np.max(pred_arrays))

    evaluation(label_arrays, pred_arrays, len(class_list))

    for i in range(len(image_list)):
        image_path = image_list[i]
        file_name = os.path.basename(image_path).split('.')[0]
        cv2.imwrite(f"{save_dir}/seg_result_img/image/{file_name}.png", cv2.cvtColor(image_arrays[i], cv2.COLOR_BGR2RGB) * 255.0)
        
        cv2.imwrite(f"{save_dir}/seg_result_img/label/{file_name}_0.png", label_arrays[i,:,:,0])
        cv2.imwrite(f"{save_dir}/seg_result_img/label/{file_name}_1.png", label_arrays[i,:,:,1])
        cv2.imwrite(f"{save_dir}/seg_result_img/label/{file_name}_2.png", label_arrays[i,:,:,2])
        cv2.imwrite(f"{save_dir}/seg_result_img/label/{file_name}_3.png", label_arrays[i,:,:,3])

        cv2.imwrite(f"{save_dir}/seg_result_img/prediction/{file_name}_0.png", pred_arrays[i,:,:,0])
        cv2.imwrite(f"{save_dir}/seg_result_img/prediction/{file_name}_1.png", pred_arrays[i,:,:,1])
        cv2.imwrite(f"{save_dir}/seg_result_img/prediction/{file_name}_2.png", pred_arrays[i,:,:,2])
        cv2.imwrite(f"{save_dir}/seg_result_img/prediction/{file_name}_3.png", pred_arrays[i,:,:,3])
    end_time = time.time()
    print("\nEnd Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")