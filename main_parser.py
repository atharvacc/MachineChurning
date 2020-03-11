import argparse

def add_training_mode_args(parser):
    parser.add_argument("--lr", required=False, help = "learning_rate", default= 0.01, type = int)
    parser.add_argument("--train_dir", required=True, help = "Directory to find training data")
    parser.add_argument("--img_extension", required=False, help = "Extension for the images", default = "png", type = str)
    parser.add_argument("--batch_size", required=False, help = "batch_size to be used", default = 32, type = int)
    parser.add_argument("--epochs", required=False, help = "Number of epochs", default= 30, type=int)
    parser.add_argument("--model_save_dir", required=False, help = "Directory to save the model", default = "./saved_models")

def add_testing_mode_args(parser):
    parser.add_argument("--test_dir", required=True, help = "Directory to find training data")
    parser.add_argument("--img_extension", required=False, help = "Extension for the images", default = "png", type = str)
    parser.add_argument("--test_save_dir", required=False, help = "Directory to save the results after inference", default = "./results")
    parser.add_argument("--model_path", required= True, help = "Full Path to the model to be used")


def main():
    parser = argparse.ArgumentParser(description="Use Different Models to be trained or run Inference using pretrained Models")
    parser.add_argument( "--model_name", required=True, help = " 'lab_gray' or 'base_gray' or 'cycle-gan' ", default= "base_gray", type = str)
    parser.add_argument("--train_or_test", required=True, help = "'train' or 'test' ", default="train", type = str)
    if (parser.parse_known_args()[0].model_name == "base_gray" or parser.parse_known_args()[0].model_name == "lab_gray"):
        if (parser.parse_known_args()[0].train_or_test == "train"):
            add_training_mode_args(parser)
        if (parser.parse_known_args()[0].train_or_test == "test"):
            add_testing_mode_args(parser)

    
    
    
    
   

main()



