# Jet classification on FPGA

This repository contains the necessary code to produce training data for b/q/g tagging of R=0.4 Seeded Cone Jets at L1. It also contains a WIP script to run a small DNN/CNN on the data.

To produce the neccessary files, you can either run an "instructional" Juypter notebook (runJetTagging.ipynb) over a small input root file, for larger files its recommended to rather run the python script makeh5.py (with the run script runh5.py).
The notebook also contains a minimal example for jet tagging.


With synthesize.py, you can also generate firmware for Graph Convolutional Networks and dense fully connected networks, for the models in https://github.com/sznajder/JetID-L1
You need a Vivado certificate to run (e.g logging into geonosis from lxplus (ssh -XY geonosis)).


Everything is running Tensorflow 2.5, and the hls4ml bartsia release

Set up the Conda environment:
```
conda env create -f environment.yml
conda activate hls4ml-l1jets
```

To generate a small trainng file and train a DNN, run the WIP notebook runJetTagging.ipynb.
To synthesize models, open the file synthesize.py and set synth=True if you wish to run the Vivado synthesis, and False if you want to read already generated projects, then do

```
python synthesize.py
```
