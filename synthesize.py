import hls4ml
import tensorflow as tf
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QActivation
from qkeras import QDense, QConv1D, QConv2D, quantized_bits
from qkeras.autoqkeras.utils import print_qmodel_summary

from pathlib import Path
import pprint 

import numpy as np

def synthezise(model):
  print("Synthesizing {}".format(mname))
   
  hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
  hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
  hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

  config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  # config['Model']['ReuseFactor'] = 1
  # config['Model']['Strategy'] = 'Latency'
  # config['LayerName'][softmax_name]['Strategy'] = 'Stable' #TODO! Check that this works and accuracy isn't screwed up
  config['LayerName'][softmax_name]['exp_table_t'] = 'ap_fixed<18,8>'
  config['LayerName'][softmax_name]['inv_table_t'] = 'ap_fixed<18,4>'

  cfg = hls4ml.converters.create_vivado_config()
  if mname.find('QGraphConv')!=-1 or mname.find('model_QInteractionNetwork_nconst_8_nbits_8')!=-1:
    print("USING IO STREAM!")
    cfg['IOType']     = 'io_stream' # Must set this if using CNNs! #TODO! Check that this works and accuracy isn't screwed up
  cfg['HLSConfig']  = config
  cfg['KerasModel'] = model
  cfg['OutputDir']  = '{}'.format(mname)
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'

  hls_model = hls4ml.converters.keras_to_hls(cfg)
  hls_model.compile()
  hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="mname_plot.pdf")
  hls_model.build(csim=False, synth=True, vsynth=True)

def getReports(indir):
    data_ = {}
    
    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
    
    if report_vsynth.is_file() and report_csynth.is_file():
        print('Found valid vsynth and synth in {}! Fetching numbers'.format(indir))
        
        # Get the resources from the logic synthesis report 
        with report_vsynth.open() as report:
          lines = np.array(report.readlines())
          data_['lut']     = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
          data_['ff']      = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
          data_['bram']    = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
          data_['dsp']     = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
          data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
          data_['ff_rel']  = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
          data_['bram_rel']= float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
          data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])
        
        with report_csynth.open() as report:
          lines = np.array(report.readlines())
          lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
          data_['latency_clks'] = int(lat_line.split('|')[2])
          data_['latency_ns']  = float(lat_line.split('|')[2])*5.0
          data_['latency_ii']   = int(lat_line.split('|')[6])
    
    return data_
   
if __name__ == "__main__": 
  
  models = [
            # "model_QGraphConv_nconst_8_nbits_8_hacked",
            #"model_QInteractionNetwork_nconst_8_nbits_8", #TODO! Not working, get hls4ml code from Javier?
            "model_QGraphConv_nconst_8_nbits_4",
            "model_QGraphConv_nconst_8_nbits_6",
            "model_QGraphConv_nconst_8_nbits_8",
            "model_QMLP_nconst_8_nbits_4",
            "model_QMLP_nconst_8_nbits_6",
            "model_QMLP_nconst_8_nbits_8"
          ]

  synth = True  # Synthesize the models
          
  for mname in models:
 
    model = tf.keras.models.load_model('JetTagModels/{}.h5'.format(mname),
                                       custom_objects={'QDense': QDense,
                                                       'QActivation': QActivation,
                                                       'QConv1D' : QConv1D,
                                                       'QConv2D' : QConv2D,
                                                       'quantized_bits': quantized_bits
                                                     })
    
    model.summary()
    print_qmodel_summary(model)
  
    softmax_name = 'activation3'
    if mname.find('QGraphConv')!=-1:
      softmax_name = 'softmax'
  
    if synth:  
      synthezise(model)
    
    # Only read latency and resource consumption     
    else:
      print("Reading hls project {}".format(mname))  
      
      data = getReports('/eos/home-t/thaarres/l1_jet_tagging_v2/{}/'.format(mname))

      print("\n Resource usage and latency: {}".format(mname))
      pprint.pprint(data)

