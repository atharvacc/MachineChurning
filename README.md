# MachineChurning



### TODO



#### Modelling

- Add metric for style and content evaluation.
- Label pixels for H&E And MUSE and FIBI images.
- Plot Loss curve for all 3 models to understand it better
- Communicate results with professor for evaluation.
- Implement segmentation saliency map to improve the accuracy of the model
- Implement the compression architecture to speed up the model

#### Interface
- Parellize blending function 
- Build graphical interface 
- Set up API for model inference
- Test graphical interface locally.



### Done

#### Parsing
- add Parsing support for basic gray scale converting (Done- AC)
- Add parsing support for lab gray scale converting(Done- AC)
- Add parsing support for cycle-gan and run it (Done-AC)
- Host data on bucket and add script to download it (Done for MUSE- AC)
- Testing gray scale and lab gray (Done- AC)
#### Modelling
- Construct full_image from stacks (Done-AC)
- Testing Setting for cycle-gan, and refine training. (Done - AC)
- Add sliding window for blending(Done-AC) 
#### Interface
- Look at builiding pipelines to store data for training purposes later on. (Done -AC: Flask app supports this)
- Mess around w load size to predict bigger input + Understand how it works (Done - AC)
- Improve model evaluation time by using custom script instead of running Ganilla. (Done - AC)
