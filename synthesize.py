# This code requires a special version of hls4ml which includes GarNet and a few minor bug fixes not yet on master.
# You can install it via
# pip install git+https://github.com/thaarres/hls4ml.git@jet_tag_paper_garnet_generic_quant
# The current projects can be inspected at https://thaarres.web.cern.ch/thaarres/l1_jet_tagging/l1_jet_tagging_hls4ml_dataset/

import sys, os
import hls4ml
import pickle
import tensorflow as tf
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QActivation, QDense, QConv1D, QConv2D, quantized_bits
from qkeras.autoqkeras.utils import print_qmodel_summary

from pathlib import Path
import pprint 

import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score

from garnet_v2 import GarNet

import matplotlib.pyplot as plt 

from joblib import Parallel, delayed
import time
import shutil

import argparse

def print_dict(d, indent=0):
  align=20
  for key, value in d.items():
    print('  ' * indent + str(key), end='')
    if isinstance(value, dict):
      print()
      print_dict(value, indent+1)
    else:
      print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))


def rename(model, layer, new_name):
    def _get_node_suffix(name):
        for old_name in old_nodes:
            if old_name.startswith(name):
                return old_name[len(name):]

    old_name = layer.name
    old_nodes = list(model._network_nodes)
    new_nodes = []

    for l in model.layers:
        if l.name == old_name:
            l._name = new_name
            # vars(l).__setitem__('_name', new)  # bypasses .__setattr__
            new_nodes.append(new_name + _get_node_suffix(old_name))
        else:
            new_nodes.append(l.name + _get_node_suffix(l.name))
    model._network_nodes = set(new_nodes)
                
def synthezise(mname,plotpath,ONAME,build=False):
  
  # Make output directories
  PLOTS = '{}/{}/'.format(plotpath,mname)
  if not os.path.isdir(PLOTS):
    os.mkdir(PLOTS)
    shutil.copyfile('{}/index.php'.format(plotpath), '{}/index.php'.format(PLOTS)) #Private: for rendering web
  if not os.path.isdir(ONAME):
    os.mkdir(ONAME)
  
  # Load model  
  model = tf.keras.models.load_model('JetTagModels/{}.h5'.format(mname),
                                     custom_objects={'QDense': QDense,
                                                     'QActivation': QActivation,
                                                     'QConv1D' : QConv1D,
                                                     'QConv2D' : QConv2D,
                                                     'quantized_bits': quantized_bits,
                                                     'GarNet': GarNet
                                                   })                                       
    
  if DEBUG:
    model.summary()
    print(model.get_config())
  
  # for i,layer in enumerate(model.layers):
  #   if layer.__class__.__name__ in ['InputLayer']:
  #     _name = "inpt_" + str(i)
  #     rename(model, layer, _name)
  #
  # model.save('TMP_{}.h5'.format(mname))
  # loaded = tf.keras.models.load_model('TMP_{}.h5'.format(mname),
  #                                    custom_objects={'QDense': QDense,
  #                                                    'QActivation': QActivation,
  #                                                    'QConv1D' : QConv1D,
  #                                                    'QConv2D' : QConv2D,
  #                                                    'quantized_bits': quantized_bits,
  #                                                    'GarNet': GarNet
  #                                                  })
      
    
  # Get softmax layer name
  for layer in model.layers:
    if layer.__class__.__name__ in ['Activation']:
      cfg = layer.get_config()
      if cfg['activation'].find('softmax')!=-1:
        softmax_name = layer.name
        print("{}: Tune hls4ml softmax implementation!".format(layer.name))

  
  # Make more QKeras compatible
  hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
  hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
  hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
  
  # make hls config
  config = hls4ml.utils.config_from_keras_model(model, granularity='name',default_reuse_factor=1) #, default_precision='ap_fixed<24,12>'
  config['Model']['Strategy'] = 'Latency'
  # config['Model']['ReuseFactor'] = 50
  config['LayerName'][softmax_name]['exp_table_t'] = 'ap_fixed<18,8>'
  config['LayerName'][softmax_name]['inv_table_t'] = 'ap_fixed<18,4>'
  config['LayerName'][softmax_name]['Strategy'] = 'Stable'
  # Handle large span of numerical values in input
  inputPrecision = 'ap_fixed<20,10,AP_RND,AP_SAT>'
  for layer in model.layers:
    if layer.__class__.__name__ in ['BatchNormalization']:
      config['LayerName'][layer.name]['Precision']['scale']   = inputPrecision
      config['LayerName'][layer.name]['Precision']['bias']    = inputPrecision
      config['LayerName'][layer.name]['Precision']['result']  = inputPrecision
    if layer.__class__.__name__ in ['InputLayer']:
      config['LayerName'][layer.name]['Precision']['scale']   = inputPrecision
      config['LayerName'][layer.name]['Precision']['bias']    = inputPrecision
      config['LayerName'][layer.name]['Precision']['result'] = inputPrecision
      
    if layer.__class__.__name__ in ['QConv1D']: # For interaction network
       if 'tmul' in layer.name and 'linear' not in layer.name:
         config['LayerName'][layer.name]['Precision']['weight'] = 'ap_uint<1>'
         config['LayerName'][layer.name]['Precision']['bias'] = 'ap_uint<1>'
    if 'gar_1' in layer.name:
      config['LayerName'][layer.name]['Precision']['aggregator_distance_weights'] = 'ap_fixed<16,6,AP_TRN,AP_SAT>'
          
  # Add tracing to all hls model layers before adding non-traceable layers            
  for layer in config['LayerName'].keys():
    config['LayerName'][layer]['Trace'] = True
  
  if 'InteractionNetwork' in mname or 'QGraphConv' in mname:
    config['SkipOptimizers'] = ['reshape_stream']
    if 'InteractionNetwork' in mname:
      # config['LayerName'][softmax_name]['Strategy'] = 'Stable'
      config['LayerName']['clone_permute_48'] = {}
      config['LayerName']['clone_permute_48']['Precision'] = inputPrecision
      config['LayerName']['concatenate_25'] = {}
      config['LayerName']['concatenate_25']['Precision'] = inputPrecision   
      
  #Special cases:      
  changeStrategy = False
  if changeStrategy: #Change strategy if layer is > 4,096. Doesn't work to set strategy per layer for io_stream type models
      for layer in model.layers:
        config['LayerName'][layer.name]['Strategy'] = 'Latency'
        w = layer.get_weights()[0]
        layersize = np.prod(w.shape)
        print("{}: {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
        if (layersize > 4096): # assuming that shape[0] is batch, i.e., 'None'
          print("Layer {} is too large ({}), changing strategy Latency --> Resource".format(layer.name,layersize))
          config['LayerName'][layer.name]['Strategy'] = 'Resource'

  print_dict(config)
  
  # Old hls4ml
  cfg = hls4ml.converters.create_config() 
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'

  if 'QGraphConv' in mname or 'InteractionNetwork' in mname:
    cfg['IOType']     = 'io_stream'
  else:
    cfg['IOType']     = 'io_parallel'
  cfg['HLSConfig']  = config
  cfg['KerasModel'] = model
  cfg['OutputDir']  = '{}/{}'.format(ONAME,mname)
  
  
  print("Convert to hls")
  print(cfg)
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  print("Compile")
  hls_model.compile()
  
  # Do plots
  hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='{}/hls4ml_in_plot_{}.png'.format(PLOTS,mname))
  tf.keras.utils.plot_model(model,to_file='{}/keras_in_plot_{}.png'.format(PLOTS,mname))
  
 

  
    
  if 'Garnet' in mname:
    
    nconst = int(mname.split("_")[3])

    X_test = np.load('/eos/home-t/thaarres/level1_jet_validation_samples/x_test_{}const_garnet.npy'.format(nconst))
    Y_test = np.load('/eos/home-t/thaarres/level1_jet_validation_samples/y_test_{}const_garnet.npy'.format(nconst), allow_pickle=True)
    X_test = X_test[:3000]
    Y_test = Y_test[:3000]

    vmax = nconst
    feat = 3
    V_test = np.ones((X_test.shape[0],1))*vmax

    y_hls  = np.array([])
    y_keras= np.array([])
    for i,j in zip(X_test,V_test.astype(np.float64)):
      i = np.expand_dims(i, axis=0)
      j = np.expand_dims(j, axis=0)
      x_hls = [i, j.astype(np.float64)]
      y_keras_ = model.predict(x_hls)
      y_hls_ = hls_model.predict(x_hls).reshape(y_keras_.shape)
      y_hls = np.concatenate([y_hls, y_hls_], axis=0) if y_hls.size else y_hls_
      y_keras = np.concatenate([y_keras, y_keras_], axis=0) if y_keras.size else y_keras_

    X_test = [X_test,V_test]

    hls4ml_pred, hls4ml_trace = hls_model.trace(X_test[:1000])

    print("hls4ml layer 'gar_1', first sample:")
    print(hls4ml_trace['gar_1'][0])

    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_test[:1000])
    print("Keras layer 'gar_1', first sample:")
    print(keras_trace['gar_1'][0])
    y_hls = hls_model.predict(X_test)


  elif 'GraphConv' in mname or mname.find('InteractionNetwork')!=-1:
    
    nconst = int(mname.split("_")[4])
    # Has shape (-1,8,3)
    X_test = np.ascontiguousarray(np.load('/eos/home-t/thaarres/level1_jet_validation_samples/x_test_{}const_QGraphConv.npy'.format(nconst)))
    Y_test = np.load('/eos/home-t/thaarres/level1_jet_validation_samples/y_test_{}const_QGraphConv.npy'.format(nconst), allow_pickle=True)
    X_test = X_test[:3000]
    Y_test = Y_test[:3000]
    
    y_keras = model.predict(X_test)
    y_hls = hls_model.predict(np.ascontiguousarray(X_test))
  else:
    X_test = np.reshape(X_test, (-1,24))
    y_keras = model.predict(X_test)
    y_hls = hls_model.predict(np.ascontiguousarray(X_test))

  accuracy_keras  = float(accuracy_score (np.argmax(Y_test,axis=1), np.argmax(y_keras,axis=1)))
  accuracy_hls4ml = float(accuracy_score (np.argmax(Y_test,axis=1), np.argmax(y_hls,axis=1)))

  accs = {}
  accs['cpu'] = accuracy_keras
  accs['fpga'] = accuracy_hls4ml

  with open('{}/{}/acc.txt'.format(ONAME,mname), "wb") as fp:
    pickle.dump(accs, fp)
  print('Keras:\n', accuracy_keras)
  print('hls4ml:\n', accuracy_hls4ml)

  if DEBUG:
    print('X_test[0]:\n', X_test[0])
    print('Y_test[0]:\n', Y_test[0])
    print('y_keras[0]:\n', y_keras[0])
    print('y_hls[0]:\n', y_hls[0])

    layer = list(hls_model.get_layers())[2]
    w = list(layer.get_weights())
    for w_ in w:
      print("Getting weights: ", w_)
      q = w_.quantizer
      print("Getting quantizer: ", q)
      if q:
        print("q(w_.data)) ",q(w_.data))
        print("q.quantizer_fn.scale) ",q.quantizer_fn.scale)

  # Plot the ROC curves
  colors  = ['#d73027','#fc8d59','#fee090','#e0f3f8','#91bfdb','#4575b4']
  labels = ['gluon', 'quark', 'W', 'Z', 'top']
  fpr = {}
  tpr = {}
  auc1 = {}
  fig = plt.figure()
  ax = fig.add_subplot()

  for i, label in enumerate(labels):
          fpr[label], tpr[label], threshold = roc_curve(Y_test[:,i], y_keras[:,i])
          auc1[label] = auc(fpr[label], tpr[label])
          ax.plot(tpr[label],fpr[label],label='%s, auc = %.1f%%'%(label,auc1[label]*100.),c=colors[i])
          fpr[label], tpr[label], threshold = roc_curve(Y_test[:,i], y_hls[:,i])
          auc1[label] = auc(fpr[label], tpr[label])
          ax.plot(tpr[label],fpr[label],label='%s HLS, auc = %.1f%%'%(label,auc1[label]*100.),linestyle='dotted',c=colors[i])
  ax.semilogy()
  ax.set_xlabel("sig. efficiency")
  ax.set_ylabel("bkg. mistag rate")
  ax.set_ylim(0.001,1)
  ax.set_xlim(0.,1.)
  plt.figtext(0.2, 0.83,r'{}'.format(mname))
  #ax.set_grid(True)
  ax.legend(loc='lower right')
  plt.savefig('{}/ROC_keras_{}.png'.format(PLOTS,mname))

  # if not 'GarNet' in mname: #TODO! Add profiling for multiple inputs
  wp, wph, ap, aph = hls4ml.model.profiling.numerical(model,hls_model,X_test)
  wp.savefig("{}/wp_{}.png".format(PLOTS,mname))
  wph.savefig("{}/wph_{}.png".format(PLOTS,mname))
  ap.savefig("{}/ap_{}.png".format(PLOTS,mname))
  aph.savefig("{}/aph_{}.png".format(PLOTS,mname))
  fig = hls4ml.model.profiling.compare(model,hls_model,X_test)
  fig.savefig("{}/compare_{}.png".format(PLOTS,mname))

  
  print("Running synthesis!")
  if build:
    hls_model.build(csim=False, synth=True, vsynth=True)

def getReports(indir):
  
    with open("{}/acc.txt".format(indir), "rb") as fp:
      acc = pickle.load(fp)
  
    data_ = {}
    if 'Garnet' in indir:
      data_['architecture']   = 'GarNet'
    elif 'GraphConv' in indir:
      data_['architecture']   = 'GCN'  
    elif 'InteractionNetwork' in indir:
      data_['architecture']   = 'IN'
    else:
      data_['architecture']   = 'MLP'

    data_['precision']   = str(indir.split('_')[-1].replace('bit','')).replace('/','')
    data_['acc_ratio']   = round(acc['fpga']/acc['cpu'],2)
    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
    
    if report_vsynth.is_file() and report_csynth.is_file():
        # Get the resources from the logic synthesis report 
        with report_vsynth.open() as report:
          lines = np.array(report.readlines())
          lut   = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
          ff    = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
          bram  = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
          dsp   = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
          lut_rel = round(float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5]),1)
          ff_rel  = round(float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5]),1)
          bram_rel= round(float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5]),1)
          dsp_rel = round(float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5]),1)
          
          
          data_['lut']     = "{} ({}\%)".format(lut  ,lut_rel )
          data_['ff']      = "{} ({}\%)".format(ff   ,ff_rel  )
          data_['bram']    = "{} ({}\%)".format(bram ,bram_rel)
          data_['dsp']     = "{} ({}\%)".format(dsp  ,dsp_rel )
        
          
          
        
        with report_csynth.open() as report:
          lines = np.array(report.readlines())
          lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
          data_['latency_clks'] = round(int(lat_line.split('|')[2]))
          data_['latency_ns']   = round(int(lat_line.split('|')[2])*5.0)
          data_['latency_ii']   = round(int(lat_line.split('|')[6]))
    
    return data_


# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-C", "--create", help="Create projects", action="store_true")
parser.add_argument("-B", "--build", help="Build projects", action="store_true")
parser.add_argument("--plotdir", help="Output path for plots", default="/eos/home-t/thaarres/www/l1_jet_tagging/l1_jet_tagging_hls4ml_dataset/")
parser.add_argument("-o", "--outdir", help="Output path for projects", default="/mnt/data/thaarres/HLS_PROJECTS")
parser.add_argument("-D", "--debug", help="High verbose", action="store_true")
args = parser.parse_args()
  
   
if __name__ == "__main__":
  

  # List of models to synthesize
  models = [
            # "model_QMLP_nconst_8_nbits_6",
            # "model_QMLP_nconst_8_nbits_8",
            # "model_QGraphConv_nconst_8_nbits_4",
            # "model_QGraphConv_nconst_8_nbits_6",
            # "model_QGraphConv_nconst_8_nbits_8",
            # "model_Garnet_nconst_8_nbits_8",
            # "model_Garnet_nconst_16_nbits_8",
            # "model_Garnet_nconst_32_nbits_8",
            # "model_Garnet_nconst_8_nbits_4",
            # "model_Garnet_nconst_8_nbits_6",
            # "model_Garnet_nconst_8_nbits_8",
            "model_QInteractionNetwork_Conv1D_nconst_8_nbits_4",
            "model_QInteractionNetwork_Conv1D_nconst_8_nbits_6",
            "model_QInteractionNetwork_Conv1D_nconst_8_nbits_8"            
          ]
  
  PLOTS = args.plotdir
  ONAME = args.outdir
  DEBUG = args.debug
  
  # Generate projects and produce firmware  
  if args.create:  
    start = time.time()
    Parallel(n_jobs=4, backend='multiprocessing')(delayed(synthezise)(modelname,PLOTS,ONAME,build=args.build) for modelname in models)
    end = time.time()
    print('Ended after {:.4f} s'.format(end-start))
      
    
  # Only read projects
  else:
    
    import pandas
    dataMap = {'architecture':[],'precision':[], 'acc_ratio':[], 'dsp':[], 'lut':[], 'ff':[],'bram':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
    
    for mname in models:
      print("Reading hls project {}/{}/".format(ONAME,mname))
      
      datai = getReports('{}/{}/'.format(ONAME,mname))
      for key in datai.keys():
         dataMap[key].append(datai[key])
    
    dataPandas = pandas.DataFrame(dataMap)
    print(dataPandas)
    print(dataPandas.to_latex(columns=['architecture','precision','acc_ratio','latency_ns','latency_clks','latency_ii','dsp','lut','ff','bram'],header=['Architecture','Precision ( \# bits )', 'Accuracy Ratio (FPGA/CPU)','Latency [ns]','Latency [clock cycles]','II [clock cycles]','DSP','LUT','FF','BRAM'],index=False,escape=False))
