# We provide 2 models and a parser to train the model
- Cycle-Gan with saliency loss
- Gray-scale converter

# Installing the libraries
```bash
pip install -r requirements.txt
#Run the command below if you get GPU Driver issues
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
# Training the model and inference

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

# Loading the interface
- Navigate to App
```bash
bash run_flask.sh #Run on port 8888
bash run_flask.sh #Run locally
```

