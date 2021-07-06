# Jet classification on FPGA

This code generates firmware for Graph Convolutional Networks and dense fully connected networks.
You need a Vivado certificate to run (e.g logging into geonosis from lxplus (ssh -XY geonosis)).

This is using Tensorflow 2.5, and the hls4ml bartsia release
Set up the Conda environment:
```
conda env create -f environment.yml
conda activate hls4ml-tf25-hack
```
Open the file synthesize.py and set synth=True if you wish to run the VIvado synthesis, and False if you want to read already generated projects, then do
```
python synthesize.py
```
