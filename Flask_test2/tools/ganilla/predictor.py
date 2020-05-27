import os
from .data import CreateDataLoader
from .models import create_model
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

from PIL import Image

def create_dir(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)

class options:
    def __init__(self, img_dir):
        self.model = "cycle_gan"
        self.gpu_ids = []
        self.isTrain = 0
        self.checkpoints_dir = "./saved_models"
        self.name = "test_cyclegan"
        self.continue_train = 0
        self.epoch = 'latest'
        self.verbose = 0
        self.resize_or_crop = "resize_and_crop"
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netG = 'resnet_fpn'
        self.norm = 'instance'
        self.no_dropout = False
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.fpn_weights = [1.0, 1.0, 1.0, 1.0]
        self.loadSize = 512
        self.fineSize = 512
        self.img_dir = img_dir
        self.result_dir = "./results"
        create_dir(self.result_dir)

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
def get_transforms(o):
    transform_list = []
    osize = [o.loadSize, o.loadSize]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    transform_list.append(transforms.RandomCrop(o.fineSize))
    transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    transf = transforms.Compose(transform_list)
    return transf

def load_dataset(img_dir):
    data_path = img_dir
    train_dataset = ImageFolderWithPaths( root = data_path, transform= get_transforms( options("./imgs/") ))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=13,
        num_workers=10,
        shuffle=False
    )
    return train_loader
    
class Predictor():
    def __init__(self, img_dir ) :
        self.opt = options(img_dir)
        model = create_model(self.opt)
        model.setup(self.opt)
        self.genA = model.netG_A
        self.genA.eval()
        del model
        self.device = torch.device("cuda:0")
        self.genA.to(self.device)
    def predict(self):
        for batch_idx, s in enumerate(load_dataset(self.opt.img_dir)):
            imgs, _, paths = s
            with torch.no_grad():
                preds = (( ((self.genA(imgs.to(self.device)).permute(0,2,3,1).cpu().numpy() * 0.5) + 0.5) * 255).astype(np.uint8) )
            for idx, img_name in enumerate(list(paths)):
                img_name = os.path.join(self.opt.result_dir, img_name.split("/")[3])
                Image.fromarray(preds[idx]).save(img_name)
            

