img_wid = 1024
img_hei = 1024

class_list = {
    0:['NT_stroma'],
    1:['NT_immune'],
    2:['NT_epithelial'],
    3:['Tumor'],
}

data_dir = "../data/origin"
save_dir = "../data"

model_path1 = "../model/best_seg_BR_cell.pt"
model_path2 = "../model/best_seg_BR_class.pt"

batch_size = 2

save_dir_list = [
    f'{save_dir}/preprocessing',
    f'{save_dir}/seg_result_img/image',
    f'{save_dir}/seg_result_img/label',
    f'{save_dir}/seg_result_img/prediction',
]
