# AI BASED COLOR MAPPING

## This code provides three kind of color mappers. 
- Gray Scale Converter using RGB Color scale
- Gray Scale Converter using LAB Color scale
- A modified CYCLE-GAN for content preservation

### Gray Scale Converter using RGB Color scale
 This model uses a ConvNet Architecture to map from gray scale images to color images. 

### Gray Scale Converter using LAB Color scale
This model uses a ConvNet Architecture to map from gray scale images to color images.

###

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

###### Training


