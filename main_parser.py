import argparse
import os
from gray_scale_converter.base_gray_scale import base_gray_scale_model

from IPython.display import display
from PIL import Image
import torchvision.transforms as transforms

from utils.util import save_image, tensor2im

import pickle
import time
import tqdm

import numpy as np
import torch

from compressedGan.models import create_model

def add_training_mode_args(parser):
    parser.add_argument("--lr", required=False, help = "learning_rate", default= 0.001, type = float)
    parser.add_argument("--train_dir", required=True, help = "Directory to find training data")
    parser.add_argument("--img_extension", required=False, help = "Extension for the images", default = "png", type = str)
    parser.add_argument("--batch_size", required=False, help = "batch_size to be used", default = 32, type = int)
    parser.add_argument("--epochs", required=False, help = "Number of epochs", default= 30, type=int)
    parser.add_argument("--model_save_dir", required=False, help = "Directory to save the model", default = "./saved_models")

def add_testing_mode_args(parser):
    parser.add_argument("--lr", required=False, help = "learning_rate", default= 0.01, type = float)
    parser.add_argument("--epochs", required=False, help = "Number of epochs", default= 30, type=int)
    parser.add_argument("--batch_size", required=False, help = "batch_size to be used", default = 32, type = int)
    parser.add_argument("--model_save_dir", required=False, help = "Directory to save the model", default = "./saved_models")
    parser.add_argument("--test_dir", required=True, help = "Directory to find training data")
    parser.add_argument("--img_extension", required=False, help = "Extension for the images", default = "png", type = str)
    parser.add_argument("--test_save_dir", required=False, help = "Directory to save the results after inference", default = "./results")
    parser.add_argument("--model_path", required= True, help = "Full Path to the model to be used")

def add_train_cycle_gan_args(parser):
    parser.add_argument("--lr", required=False, help = "learning_rate", default= 0.001, type = float)
    parser.add_argument("--train_dir", required=True, help = "Directory to find training data")
    parser.add_argument("--batch_size", required=False, help = "batch_size to be used", default = 1, type = int)
    parser.add_argument("--epochs", required=False, help = "Number of epochs", default= 80, type=int)
    parser.add_argument("--epoch_decay", required=False, help = "Number of epochs for decaying after epochs", default= 20, type=int)
    parser.add_argument("--model_save_dir", required=False, help = "Directory to save the model", default = "./saved_models")

def add_test_cycle_gan_args(parser):
    parser.add_argument("--epoch_use", required=False, help = "Epoch number to be used for testing", default = "latest")
    parser.add_argument("--test_dir", required=True, help = "Directory to find testing data")
    parser.add_argument("--model_path", required=False, help = "Directory to find the model", default = "./saved_models")
    parser.add_argument("--test_model_name", required=False, help = "Name of model", default = "test_cyclegan")
    parser.add_argument("--result_dir", required=False, help = "where to store the predictions", default = "./result" )

def add_train_cycle_gan_compressed_args(parser):
    parser.add_argument("--lr", required=False, help = "learning_rate", default= 0.001, type = float)
    parser.add_argument("--train_dir", required=True, help = "Directory to find training data")
    parser.add_argument("--batch_size", required=False, help = "batch_size to be used", default = 1, type = int)
    parser.add_argument("--epochs", required=False, help = "Number of epochs", default= 80, type=int)
    parser.add_argument("--epoch_decay", required=False, help = "Number of epochs for decaying after epochs", default= 20, type=int)
    parser.add_argument("--model_save_dir", required=False, help = "Directory to save the model", default = "./saved_models")
    parser.add_argument("--stat_A", required=False, help="Statistical information for ground-truth A images for computing FID to guide the achitecture optimization",
                        default= None)
    parser.add_argument("--stat_B", required=False, help="Statistical information for ground-truth B images for computing FID to guide the achitecture optimization",
                        default= None)
    parser.add_argument("--ground_truth_A", required=False,
                        help="Ground truth A images to be provided to produce stat_A. The sub-folder should be train",
                        default=None)
    parser.add_argument("--ground_truth_B", required=False,
                        help="Ground truth B images to be provided to produce stat_B. The sub-folder should be val",
                        default=None)

def add_test_cycle_gan_compressed_args(parser):
    parser.add_argument("--epoch_use", required=False, help = "Epoch number to be used for testing", default = "latest")
    parser.add_argument("--test_dir", required=True, help = "Directory to find testing data")
    parser.add_argument("--model_path", required=False, help = "Directory to find the model", default = "./saved_models")
    parser.add_argument("--test_model_name", required=False, help = "Name of model", default = "test_cyclegan")
    parser.add_argument("--result_dir", required=False, help = "where to store the predictions", default = "./result" )







def main():
    ### Initialize Parser and add some base options
    parser = argparse.ArgumentParser(description="Use Different Models to be trained or run Inference using pretrained Models")
    parser.add_argument( "--model_name", required=False, help = " 'lab_gray' or 'base_gray' or 'cycle-gan' ", default= "base_gray", type = str)
    parser.add_argument("--train_or_test", required=True, help = "'train' or 'test' ", default="train", type = str)
    if (parser.parse_known_args()[0].model_name == "base_gray" or parser.parse_known_args()[0].model_name == "lab_gray"): # If using the base_gray scale or lab model 
        if (parser.parse_known_args()[0].train_or_test == "train"): ## If training
            add_training_mode_args(parser)
            myModel = base_gray_scale_model( lr = parser.parse_args().lr , batch_size = parser.parse_args().batch_size, epochs = parser.parse_args().epochs, 
                                            model_save_dir = parser.parse_args().model_save_dir,  train_dir = parser.parse_args().train_dir, extension = parser.parse_args().img_extension) 
            myModel.train()
        if (parser.parse_known_args()[0].train_or_test == "test"): ## If testng
            add_testing_mode_args(parser)
            myModel = base_gray_scale_model( lr = parser.parse_args().lr , batch_size = parser.parse_args().batch_size, epochs = parser.parse_args().epochs, 
                                            model_save_dir = parser.parse_args().model_save_dir, extension = parser.parse_args().img_extension, test_dir= parser.parse_args().test_dir,
                                            test_save_dir= parser.parse_args().test_save_dir, model_path= parser.parse_args().model_path)
            myModel.model.summary()
            myModel.predict()
            
            
    elif (parser.parse_known_args()[0].model_name == "cycle-gan"):
        if (parser.parse_known_args()[0].train_or_test == "train"):
            add_train_cycle_gan_args(parser)
            os.system("python ganilla/train.py --dataroot {train_dir} --lr {lr}  --epoch latest --batch_size {batch_size} --niter {epochs} --niter_decay {epoch_decay} --checkpoints_dir {model_save_dir}  \
            --loadSize 512 --fineSize 512 --display_winsize 512 --save_epoch_freq 20   --name test_cyclegan --model cycle_gan --netG resnet_fpn" .format(train_dir = parser.parse_args().train_dir, lr = parser.parse_args().lr, batch_size = parser.parse_args().batch_size, model_save_dir = parser.parse_args().model_save_dir, epochs = parser.parse_args().epochs,
                epoch_decay = parser.parse_args().epoch_decay) )
            
            
        if (parser.parse_known_args()[0].train_or_test == "test"):
            add_test_cycle_gan_args(parser)
            os.system("python ganilla/test.py --epoch {latest} --results_dir {result_dir} --dataroot {data_path} \
                         --checkpoints_dir {model_path} --loadSize 512 --fineSize 512 --display_winsize 512 --name {test_model_name} \
                              --model cycle_gan --netG resnet_fpn" .format(latest = parser.parse_args().epoch_use, data_path = parser.parse_args().test_dir,
                               model_path = parser.parse_args().model_path, test_model_name = parser.parse_args().test_model_name, result_dir = parser.parse_args().result_dir))
    elif (parser.parse_known_args()[0].model_name == "cycle-gan-compressed"):
        if (parser.parse_known_args()[0].train_or_test == "train"):
            add_train_cycle_gan_compressed_args(parser)
            stat_A = parser.parse_args().stat_A
            stat_A = stat_A.rstrip('/')
            stat_B = parser.parse_args().stat_B
            stat_B = stat_B.rstrip('/')
            ground_truth_A = parser.parse_args().ground_truth_A
            ground_truth_B = parser.parse_args().ground_truth_B

            # Get the statistical information of the Ground-truth A and B images if not provided. To do that, generated testing images from the state-of-the-art should be provided.
            if stat_A is None and ground_truth_A is None:
                raise AssertionError("No ground truth images A or statistics of the ground truth images A provided")
            if stat_B is None and ground_truth_B is None:
                raise AssertionError("No ground truth images B or statistics of the ground truth images B provided")

            if stat_A is None:
                if not os.path.exists(stat_A+"/train"):
                    raise AssertionError("The folder of ground truth images A is not correct. No train sub-folder.")
                os.system("python compressedGan/get_real_stat.py --dataroot {ground_truth_A} --load_size 512 --crop_size 512 --dataset_mode aligned --phase train --output_path {model_save_dir}/real_stat_A_path_aligned/cycle-gan-compressed-A.npz".format(
                    ground_truth_A=parser.parse_args().ground_truth_A, model_save_dir=parser.parse_args().model_save_dir
                ))
                stat_A = "{model_save_dir}/real_stat_A_path_aligned/cycle-gan-compressed-A.npz".format(
                    model_save_dir=parser.parse_args().model_save_dir
                )
            if stat_B is None:
                if not os.path.exists(stat_B + "/val"):
                    raise AssertionError("The folder of ground truth images B is not correct. No val sub-folder.")
                os.system("python compressedGan/get_real_stat.py --dataroot {ground_truth_B} --load_size 512 --crop_size 512 --dataset_mode aligned --phase val --output_path {model_save_dir}/real_stat_B_path_aligned/cycle-gan-compressed-B.npz".format(
                    ground_truth_B=parser.parse_args().ground_truth_B,model_save_dir=parser.parse_args().model_save_dir
                ))
                stat_B = "{model_save_dir}/real_stat_B_path_aligned/cycle-gan-compressed-B.npz".format(
                    model_save_dir=parser.parse_args().model_save_dir
                )
            #train the mobile (decomposed model)
            os.system("python compressedGan/train.py --dataroot {train_dir} --display_winsize 512 --crop_size 512 --load_size 512 --model cycle_gan --lr {lr} --log_dir {model_save_dir}/mobile --real_stat_A_path {stat_A} --real_stat_B_path {stat_B}".format(
                train_dir=parser.parse_args().train_dir, lr=parser.parse_args().lr,
                batch_size=parser.parse_args().batch_size, model_save_dir=parser.parse_args().model_save_dir,
                epochs=parser.parse_args().epochs,
                epoch_decay=parser.parse_args().epoch_decay, stat_A=stat_A,stat_B=stat_B
            ))
            #distill the knowledge from the teacher network to the student network
            os.system("python compressedGan/distill.py --dataroot {train_dir} --display_winsize 512 --crop_size 512 --load_size 512 --dataset_mode unaligned --lr {lr} --nepochs {epochs} --nepochs_decay {epoch_decay}  --distiller resnet --log_dir {model_save_dir}/distill --gan_mode lsgan --student_ngf 32 --ndf 64 --restore_teacher_G_path {model_save_dir}/mobile/checkpoints/latest_net_G_A.pth --restore_pretrained_G_path {model_save_dir}/mobile/checkpoints/latest_net_G_A.pth --restore_D_path {model_save_dir}/mobile/checkpoints/latest_net_D_A.pth --real_stat_path {stat_B} --lambda_recon 10 --lambda_distill 0.01 --save_epoch_freq 20".format(
                train_dir=parser.parse_args().train_dir, lr=parser.parse_args().lr,
                batch_size=parser.parse_args().batch_size, model_save_dir=parser.parse_args().model_save_dir,
                epochs=parser.parse_args().epochs,
                epoch_decay=parser.parse_args().epoch_decay, stat_B=stat_B
            ))
            #train the "once-and-for-all" network for architectural searching
            os.system("python compressedGan/train_supernet.py --dataroot {train_dir} --display_winsize 512 --crop_size 512 --load_size 512 --lr {lr} --nepochs {epochs} --nepochs_decay {epoch_decay} --dataset_mode unaligned --supernet resnet --log_dir {model_save_dir}/supernet --gan_mode lsgan --student_ngf 32 --ndf 64 --restore_teacher_G_path {model_save_dir}/mobile/checkpoints/latest_net_G_A.pth --restore_student_G_path {model_save_dir}/distill/checkpoints/latest_net_G.pth --restore_D_path {model_save_dir}/distill/checkpoints/latest_net_D.pth --real_stat_path {stat_B} --lambda_recon 10 --lambda_distill 0.01 --save_epoch_freq 20 --config_set channels-32".format(
                train_dir=parser.parse_args().train_dir, lr=parser.parse_args().lr,
                batch_size=parser.parse_args().batch_size, model_save_dir=parser.parse_args().model_save_dir,
                epochs=parser.parse_args().epochs*2,
                epoch_decay=parser.parse_args().epoch_decay*2, stat_B=stat_B
            ))
            #search for the best architecture and save the result
            os.system("python compressedGan/search.py --dataroot {train_dir} --display_winsize 512 --crop_size 512 --load_size 512 --lr {lr} --dataset_mode single --restore_G_path {model_save_dir}/supernet/checkpoints/latest_net_G.pth --output_path {model_save_dir}/supernet/result.pkl --ngf 32 --batch_size {batch_size} --config_set channels-32 --real_stat_path {stat_B}".format(
                train_dir=parser.parse_args().train_dir, lr=parser.parse_args().lr,
                batch_size=parser.parse_args().batch_size, model_save_dir=parser.parse_args().model_save_dir,
                epochs=parser.parse_args().epochs * 2,
                epoch_decay=parser.parse_args().epoch_decay * 2, stat_B=stat_B
            ))

        if (parser.parse_known_args()[0].train_or_test == "test"):
            add_test_cycle_gan_compressed_args(parser)
            #open the searched architecture configuration and create the compressed model based on the information
            with open('{model_path}/supernet/result.pkl'.format(
                    model_path=parser.parse_args().model_path
            ), 'rb') as f:
                opt = pickle.load(f)
            model_ours = create_model(opt, verbose=False)
            model_ours.setup(opt, verbose=False)
            #create the normalizer to normalize the input
            transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform = transforms.Compose(transform_list)
            imgs_dir = parser.parse_args().test_dir
            files = os.listdir(imgs_dir)
            #load the images and produce the output
            for file in files:
                if not file.endswith('.jpg'):
                    continue
                base = file.split('.')[0]
                path = os.path.join(imgs_dir, file)
                A_img = Image.open(path).convert('RGB')
                input = transform(A_img).to('cuda:0')
                input = input.reshape([1, 3, 256, 256])
                output_ours = model_ours.netG(input).cpu()
                B_ours = tensor2im(output_ours)
                save_image(B_ours, '{test_dir}/compressed/{base}.png'.format(
                    test_dir = parser.parse_args().test_dir,
                    base = base
                ), create_dir=True)



main()



