img_wid = 1024
img_hei = 1024

class_list = {
    0:['NT_stroma'],
    1:['NT_immune'],
    2:['NT_epithelial'],
    3:['Tumor'],
}

data_dir = "../data/origin"
# data_dir = "../../../제출-유효성/4. 평가용 데이터셋 + 목록/test"
save_dir = "../data"

model_path1 = "../model/best_seg_BR_cell.pt"
model_path2 = "../model/best_seg_BR_class.pt"

batch_size = 1

save_dir_list = [
    f'{save_dir}/preprocessing',
    f'{save_dir}/seg_result_img/image',
    f'{save_dir}/seg_result_img/label',
    f'{save_dir}/seg_result_img/prediction',
]
