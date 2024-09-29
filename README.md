# DSSW
This is the implementation for our paper entitled: "Dynamic Sub-Sequence Warping: A Representation-Based Similarity Measure for Long Time Series"
![image text](https://github.com/zhourongleiden/DSSW/blob/main/DSSW.png) <br>
Figure: The proposed DSSW method consists of a time series segmentation module, a feature extractor module, and a similarity measure module.
## Contents
### File
* DSSW.py: the script for testing DSSW <br>
* args.py: parameter configurations <br>
* cloud_model: the proposed DSSW method
### Data
Please download the data for the experiment from [Google Drive](https://drive.google.com/file/d/1ijqmPkFZXC3HZhZ6zRxfz9Rc3qnt2WnG/view?usp=share_link), 
and unzip the file to create the "data" folder.
## Run the command
Please run `python DSSW.py --dataname dataset_name --noc value_of_noc --w1 value_of_w1 --w2 value_of_w2` to test DSSW on a specific dataset <br>
(Please set value of "dataset_name", "noc", "w1", and "w2" as those defined in the paper)
