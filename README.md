# MachineChurning



### TODO

#### Parsing


- Add parser for preprocessing images for high resolution images
- Testing Setting for cycle-gan, and refine training. (AC - IP)
- Host data for FIBI, and provide script to donwload it 
- Set up ACL to download files
#### Modelling

- Add metric for style and content evaluation.
- Label pixels for H&E And MUSE and FIBI images.
- Plot Loss curve for all 3 models to understand it better
- Communicate results with professor for evaluation.
- Modify with architecture to use U-Net to see if that makes a difference.
- Add sliding window for blending 
- Mess around w load size to predict bigger input + Understand how it works

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
- Testing gray scale and lab gray (Done- AC)
#### Modelling
- Construct full_image from stacks (AC - Done)

#### Interface
