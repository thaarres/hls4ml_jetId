# Jet classification on FPGA

This repository contains the necessary code to translate the MLP, GCN, Interaction Network and GarNet in  https://github.com/sznajder/JetID-L1 into firmware.
You need a Vivado certificate to run (e.g logging into geonosis from lxplus (ssh -XY geonosis)).


Everything is running Tensorflow 2.5, to set up the Conda environment:
```
conda env create -f environment.yml
conda activate hls4ml-l1jets
```

This code requires a special version of hls4ml which includes GarNet and a few minor bug fixes not yet on master.
You can install it via
```
 pip install git+https://github.com/thaarres/hls4ml.git@jet_tag_paper_garnet_generic_quant
```
The script to be run is synthesize.py, to run the first time edit the model list and do
```
python synthesize.py -C -B #creates the projects and runs the synthesis
```
Other options:
```
python synthesize.py -C.    #creates the projects
python synthesize.py -C -B #creates the projects and runs the synthesis
python synthesize.py   # assumes you have already synthesized the models and prints out a latex table with latency/resources/accuracy
```
