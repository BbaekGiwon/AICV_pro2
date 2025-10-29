import os, cv2, json
from glob import glob
import numpy as np
from tqdm import tqdm
import argparse
import time
from param import img_wid, img_hei
from param import class_list, data_dir, save_dir
import os

def json2numpy(json_path):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        json_object = json.load(f)
    objects_list = {
        "Stroma": [],
        "Immune": [],
        "Normal": [],
        "Tumor": [],
    }
    label_overay = np.zeros((img_wid, img_hei, len(class_list)), dtype=np.float32)
    for polygon_object in json_object["content"]["file"]["object"]:
            objects_list[polygon_object["label"]].append(
                np.array(polygon_object["coordinate"]).astype(np.int32)
            )
    
    cnt = 1
    for n, key in enumerate(objects_list):
        for obj in objects_list[key]:
            mask_image = np.zeros((img_wid, img_hei), dtype=np.uint8)
            cv2.fillPoly(mask_image, [obj], color=255)
            label_overay[...,n] = np.where(mask_image==255, cnt, label_overay[...,n])
            cnt += 1
    return label_overay

def create_forlder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
if __name__ == '__main__':
    start_time = time.time()
    print("Start Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-fn", "--filename", type=str, default='', help="선택적 파라미터 (기본값: '')")
    
    args = parser.parse_args()
    if args.filename == '':
        json_list = glob(f"{data_dir}/label/*.json")
        print(f"{len(json_list)}     파일 이름이 제공되지 않았습니다.")
    else:
        print(f"파일 이름: {args.filename}")
        json_list = [f"{data_dir}/label/{args.filename}.json"]
        print(f"{len(json_list)}     파일 이름: {args.filename}")

    ####################################################################################################
    create_forlder(f'{save_dir}/preprocessing/')
    for i in tqdm(range(len(json_list))):
        json_path = json_list[i]
        file_name = os.path.basename(json_path).split('.')[0]

        label_overay = json2numpy(json_path)
        np.save(f'{save_dir}/preprocessing/{file_name}.npy', label_overay)

    end_time = time.time()
    print("\nEnd Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")