# AI BASED COLOR MAPPING

## This code provides three kind of color mappers. 
- Gray Scale Converter using RGB Color scale
- Gray Scale Converter using LAB Color scale
- Modified CYCLE-GAN for content preservation

### Gray Scale Converter using RGB Color scale
 This model uses a ConvNet Architecture to map from gray scale images to color images. 

### Gray Scale Converter using LAB Color scale
This model uses a ConvNet Architecture to map from gray scale images to color images.

### Modified CYCLE-GAN for content preservation
This model uses a modified cycle gan based on the Ganilla paper[https://arxiv.org/abs/2002.05638]

#### Training and Testing
We use our "main parser" to train and test this model as follows using the following parameters. For the two gray
scale converters we have the following params that needs to be set
##### Training Gray Scale Converters
- --model_name: model to be used (lab_gray/base_gray)
- --train_or_test: training or testing (train/test)
- --lr: Learning rate to be used 
- --train_dir: Directory to find the training data. We just need the images since the input and output is generated within the scripts
- --img_extension: Extension of all of the images. Ex: "jpg"
- --batch_size: The batch size to be used
- --epochs: Number of Epochs to train for
- --model_save_dir: Directory to save the final trained model in 

##### Testing Gray Scale Converters
- --test_dir: Directory where the test data can be found
- --img_extension: Extension of the images
- --test_save_dir: Directory where the results should be saved
- --model_path: Full path to model. The model should be saved as a h5 file

* FOR CYCLE GAN WE NEED A VERY SPECIFIC FOLDER STRUCTURE. We have 4 folders, trainA, trainB, testA, and testB. The images should be stored within them.
REFER TO MUSE DATASET THAT CAN BE DOWNLOADED AS A REFERENCE *
###### Training Modified Cycle-GAN model
- --lr: Learning rate to be used 
- --train_dir: Directory to find the training data. We just need the images since the input and output is generated within the scripts
- --batch_size: The batch size to be used
- --epochs: Number of Epochs to train for at the base learning rate
- --epochs_decay: Number of epochs to lienar decay the learning rate after the number of epochs specified in the previous argument finishes running
- --model_save_dir: Directory to save the final trained model in 

###### Testing Modified Cycle-GAN model
- --epoch_use: The epoch number within the saved models that is used for training
- --test_dir: Directory where the test data can be found. 
- --model_path: Directory where the model is saved
- --test_model_name: Name given to model. default name is test_cyclegan
- --result_dir: Directory where the results are stored.





