# PixelLink

This code is based on PixlLink and PSENet, the performance is not satisfactory

## Requirements
* Python 3.6
* PyTorch v0.4.1+
* opencv-python 3.4


## Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ic15.py --arch vgg16 --batch_size 24
```

## Testing
```
CUDA_VISIBLE_DEVICES=0 python test_ic15.py --resume [path of model]
```

## Eval script for ICDAR 2015
```
cd eval
sh eval_ic15.sh
```


## Performance
| Dataset | Pretrained | Precision (%) | Recall (%) | F-measure (%) | FPS (1080Ti) | Input |
| - | - | - | - | - | - | - |
| ICDAR2015 | No | 81.2 | 75 | 78 | 5 | 1280*768 |

## TODO
- [ ] Find the bug of the performance issue.
- [ ] Accomplish the code with better config file and more datasets


## Trainning model Graph 2 GPU
#--------------------------------#
!CUDA_VISIBLE_DEVICES=0,1 python train_ic15.py --pretrain /models/checkpoints/checkpoint_390.pth  --arch vgg16 --batch_size 1 --checkpoint /models
## Config deploy Nginx
Folder: /etc/nginx/sites-enabled/ai_product

## convert pdf to images 
find . -maxdepth 1 -type f -name '*.pdf' -exec pdftoppm -jpeg {} {} \;



## Train model 
cmd : CUDA_VISIBLE_DEVICES=0,1 python train_ic15.py --n_epoch 1111 --lr 0.01 --pretrain models/checkpoints/checkpoint_840.pth  --arch vgg16 --batch_size 1 --checkpoint models/checkpoints/
# --pretrain đường dẫn đến file model 
# --checkpoint đường dẫn lưu model 
# -- arch backbone 

# Dataset 
home/gem/phucph/OCR/backend/apps/text_detector/PixelLink/dataset/ 

file : icdar2015_loader.py  file xử lý là load ảnh trainning 
    dòng 219-225 : 
        img_paths, gt_paths =  load_data_gt(data_dirs, gt_dirs)
        diff_img_paths, diff_gt_paths = load_data_gt(data_diff_dirs, gt_dirs)

        self.img_paths.extend(img_paths)
        self.gt_paths.extend(gt_paths)
        self.diff_img_paths.extend(diff_img_paths)
        self.diff_gt_paths.extend(diff_gt_paths)

    Tạo 2 danh sách diff_img_paths, diff_gt_paths chứa dường dẫn đến những ảnh khó và ground truth tương ứng 
    Hàm : def get_img_gt(self, index):
        
        random_truth = torch.rand(1).item() < config.OHEM_ratio
        
        if random_truth:
          index = random.randint(0, len(self.diff_img_paths)-1)
          img_path = self.diff_img_paths[index]
          gt_path = self.diff_gt_paths[index]
        
          return img_path, gt_path 

        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        return img_path, gt_path
    Hàm láy ra ảnh khó theo tỉ lệ random < config.ohem raio ( 0.3 tương ứng với 30%)


file : icdar2015_testloader.py file xử lý là load ảnh test

file : transfrom.py và images_process.py chứa các hàm xử lý dữ liệu và tăng cường dữ liệu như resize padding, xoay, ... 
    file transfrom.py 
    Dòng 112-121 lựa chọn các hàm tăng cường dữ liệu 
    class build_transfrom(object):
    def __init__(self):
        self.augment = Compose([
            RandomResize(),
            RandomRatioScale(),
            
            # RandomRotate90(),
            RandomRotate(),
            RandomCrop()
        ])
# train_ic15.py

dòng 359 - 389 : tải model pretrain 
dòng 77-78 : Lựa chon bacbone cho model 

## test 
cmd : CUDA_VISIBLE_DEVICES=0,1 python test_ic15.py --resume /home/gem/phucph/OCR/backend/apps/text_detector/PixelLink/models/checkpoints/model_test_2images/checkpoint_847.pth 
# --rusume đường dẫn đến model 

# test_ic15.py 

dòng 85- 115 : load model pretrain 

