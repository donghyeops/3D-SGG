# 3D Scene Graph Generation
- Paper: *"Modelling Dynamics of Indoor Environments with 3D Scene Graphs", ICROS*
- 3D Scene Graph includes Objects, Attributes and Spatial Relationships in 3D virsual environments (Ai2THOR).
    <p></p>
    <p align="left" vlign="center">
      <img src="./imgs/sgg.png" height="350">
    </p>

## Structure
- **ARNet_ai2thor** : Deep Neural Networks for 3D Scene Graph Generation
  - AttNet: Attributes Recognition Network
  - RelNet: Spatial Relationship Recognition Network
  - TransNet: 3D Localization Network
    <p align="left" vlign="center">
      <img src="./imgs/sr.png" height="250">
    </p>
- **VeQA** : Question Answering System (this need reasoning server with knowrob ontology)  
    <p align="left" vlign="center">
      <img src="./imgs/vqa.png" height="250">
    </p>

## Dependency
```
pytorch==0.4.1
ai2thor==0.0.44
pillow
graphviz
matplotlib
opencv-python
colormap
easydev
easydict
```

## How to run
__1. Prepare yolo v3 model__
   1) get yolo model from [IQA].
   2) clone [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) code on this repo.
   3) convert model to pytorch version using PyTorch-YOLOv3 codes.


__2. Run 3D-SGG system__
   1) install dependency
   ```
   pip install -r requirements.txt
   ```
   2) collect raw data using GUI controller.
   ```
   python run_thor.py
   # log dataset (use pre-defined action history)
   ```
   3) preprocessing raw data.
   ```
   python data_preprocessing.py
   ```
   4) train each networks in ARNet_ai2thor.
   ```
   cd ARNet_ai2thor
   python ai2thor_make_roidb.py
   python train_AttNet.py
   python train_RelNet.py
   python train_TransferNet.py
   ```
   5) run total system on ai2thor.
   ```
   python run_thor.py
   # check dnn checkboxes
   # check visualization checkbox
   # controll agent using button or keyboard
   ```


__Exra. Run VeQA system__
   1) generate QA dataset.
   ```
   python generate_question.py
   ```
   2) train semantic parser model.
   ```
   python semantic_parser.py
   ```
   3) test QA system. (this is need to knowledge reasoning engine)
   ```
   # run knowledge reasoning engine with knowrob.
   python run_qa.py
   ```
