# Metric3DV2-on-NYUV2
This repository contains a Python script built on top of [Metric3D](https://github.com/YvanYin/Metric3D).

Running this repository requires a correctly set-up and working Metric3D environment.
In order to run the scripts:   
1- Clone the repository [here](https://github.com/YvanYin/Metric3D) and create environment / install requirements as described there.  
2- Change the environment name in .sh file(If you did not create a conda environment, you might run the python script right away) .  
3- Put the files under the main project folder .     
4- If you created a conda environment, you can run the respective script using:  
```bash
bash nyu_test.sh
```
If you did not create a conda environment for this project, just run:
```bash
python nyu_test.py
```


## About the code
Once you run the script, it will try to download the dataset under /home/{your_username}/nyu_cache. All my scripts use the cache there, so if you already have it, please move the readily available dataset to there.  
  
It is further possible to change the dataset sampling by:  

```python
dataset = load_dataset("sayakpaul/nyu_depth_v2", split="validation[:654]", cache_dir=home_dir+"/nyu_cache")
dataset = dataset.select(range(0, 654, 6))  # Sample every 6th data in dataset
```


## Acknowledgment
This work is based on [Metric3D](https://github.com/YvanYin/Metric3D), developed by [Wei Yin](https://github.com/YvanYin).    
Dataset used can be found at [here](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2).
