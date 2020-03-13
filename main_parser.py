import argparse
import os
from gray_scale_converter.base_gray_scale import base_gray_scale_model

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

def add_test_cycle_gan_args(parser):
    parser.add_argument("--lr", required=False, help = "learning_rate", default= 0.001, type = float)
    parser.add_argument("--train_dir", required=True, help = "Directory to find training data")
    parser.add_argument("--batch_size", required=False, help = "batch_size to be used", default = 32, type = int)
    parser.add_argument("--epochs", required=False, help = "Number of epochs", default= 80, type=int)
    parser.add_argument("--epoch_decay", required=False, help = "Number of epochs for decaying after epochs", default= 20, type=int)
    parser.add_argument("--model_save_dir", required=False, help = "Directory to save the model", default = "./saved_models")






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
            add_test_cycle_gan_args(parser)
            os.system("python ganilla/train.py --dataroot {train_dir} --lr {lr} --batch_size {batch_size} --niter {epochs} --niter_decay {epoch_decay} --checkpoints_dir {model_save_dir}  \
            --loadSize 512 --fineSize 512 --display_winsize 512 --name test_cyclegan --model cycle_gan --netG resnet_fpn" .format(train_dir = parser.parse_args().train_dir, lr = parser.parse_args().lr, batch_size = parser.parse_args().batch_size, model_save_dir = parser.parse_args().model_save_dir, epochs = parser.parse_args().epochs,
             epoch_decay = parser.parse_args().epoch_decay) )
            
            """
        if (parser.parse_known_args()[0].train_or_test == "test"):
            #Test Param
            """

                                            


    
    
    
    
   

main()



