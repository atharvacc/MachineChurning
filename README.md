# MachineChurning



### TODO

#### Parsing


- Add parser for preprocessing images for high resolution images
- Testing gray Scale and lab_gray to make sure its working (AC - IP)
- Testing Setting for cycle-gan, and refine training. (AC - IP)
- Host data for FIBI, and provide script to donwload it 
- Fix image being corrupted bug for wget

#### Modelling
- Construct full_image from stacks (AC - IP)
- Add metric for style and content evaluation.
- Label pixels for H&E And MUSE and FIBI images.
- Plot Loss curve for all 3 models to understand it better
- Communicate results with professor for evaluation.
- Modify with architecture to use U-Net to see if that makes a difference.
- Port PyTorch code to Tensorflow. 

#### Interface
- Build graphical interface 
- Set up API for model inference
- Test graphical interface locally.
- Look at builiding pipelines to store data for training purposes later on.


### Done

#### Parsing
- Add Parsing support for basic gray scale converting (Done- AC)
- Add parsing support for lab gray scale converting(Done- AC)
- Add parsing support for cycle-gan and run it (Done-AC)
- Host data on bucket and add script to download it (Done for MUSE- AC)

#### Modelling

#### Interface
