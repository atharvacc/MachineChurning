# AI BASED COLOR MAPPER FOR Microscopy


## We provide 2 models and a parser to train the model
- Cycle-Gan with saliency loss
- Gray-scale converter

## Installing the libraries
```bash
pip install -r requirements.txt
#Run the command below if you get GPU Driver issues
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## Training the model and inference
- pretrained models can be found here 
https://drive.google.com/drive/folders/1FdxMdDVCZHN-V0x_tOvS4wemccTl6XEg?usp=sharing
- Copy the pretrained model folder to MachineChurning/App/. THE DIRECTORY STRUCTURE IS IMPORTANT FOR THE CODE TO BUILD.
- Copy your files to the MachineChurning Directory with the following tree structure

```bash
MUSE/
├── testA
├── testB
├── trainA
└── trainB
```

```python
python main_parser.py --train_or_test train <args> # For training
python main_parser.py --train_or_test test <args> # For testing
```
The documentation for main_parser can be found under docs/parser_documentation.md

## Loading the interface
- Navigate to App
- Run the commands below
```bash
bash run_flask.sh #Run on port 8888
bash run_flask.sh local #Run locally
``` 
<img src='results/int.png' width=384 height=384>

<img src='results/int_res.png' width=384 height=384>

## A few results
### Original MUSE
<img src='results/muse.png' width=384 height=384>

### Predicted H&E
<img src='results/muse_pred.png' width=384 height=384>
