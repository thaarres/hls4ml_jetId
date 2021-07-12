import os 
import math
import uproot
import numpy as np
import h5py
import sys 
import tensorflow_io as tfio
import tensorflow as tf

#from skhep.math import LorentzVector

def getParticle(branches, label, iEvt):
    features = ["_pt", "_eta", "_phi", "_mass", "_pdgId", "_vz"]
    cands =  np.array([])
    outNames = []
    for i in range(30):
        this_cand =  np.array([])
        for f in features:
            f_name = "%s_dau%i%s" %(label, i, f)
            outNames.append(f_name)
            my_column = np.array(branches[f_name][iEvt])
            my_column = np.reshape(my_column, (my_column.shape[0],1,1))
            this_cand = np.concatenate((this_cand, my_column), axis=2) if this_cand.size else my_column
        cands = np.concatenate((cands, this_cand), axis=1) if cands.size else this_cand
    return cands, outNames

def findGenJet(recoJet, genJets):
    idx = -99
    for i, genJet in enumerate(genJets):
        if genJet[0] == recoJet[4]:
            idx = i

    return idx

def getJet(branches, label, iEvt, gen=False):
    features = ["_pt", "_eta", "_phi", "_mass", "_genpt", "_gendr"]
    if gen == True: 
        features = ["_pt", "_eta", "_phi", "_mass", "_partonFlavour", "_hadronFlavour"]
    for i in range(len(features)):
        features[i] = "%s%s" %(label, features[i])
    jets = np.array([])
    for f in features:
        my_column = np.array(branches[f][iEvt])
        my_column = np.reshape(my_column, (my_column.shape[0],1))
        jets = np.concatenate((jets, my_column), axis=1) if jets.size else my_column
    if gen == False:
        features.append("{}_partonFlavour".format(label))
        features.append("{}_hadronFlavour".format(label))
        features.append("{}_truth".format(label))
        for i in range(0,3):
            my_column = np.zeros(len(branches[f][iEvt]))
            my_column = np.reshape(my_column, (my_column.shape[0],1))
            jets = np.concatenate((jets, my_column), axis=1) if jets.size else my_column
        
    return jets, features
    

if __name__ == "__main__":
  

  file_in = uproot.open("/eos/home-t/thaarres/L1_tuple/perfNano_TTbar_PU200_GenParts_More.root")
  tree = file_in.get("Events")
  branches = tree.arrays()

  nJets = branches["nL1PuppiSC4Jets"]
  first = int(sys.argv[1])
  last = min(int(sys.argv[2]), len(nJets))

      
  outFileName = "ttbar_%i_%i.h5" %(first, last)
      
  recoJets_all = np.array([])
  recoCands_all = np.array([])

  recoJetFeatures = []
  recoCandFeatures = []

  for i in range(first, last):

      recoJets, recoJetFeatures   = getJet(branches, "L1PuppiSC4Jets", i) #L1PuppiSC4Jets, L1PuppiSC4EmuJets
      recoCands, recoCandFeatures = getParticle(branches, "L1PuppiSC4Jets", i)
      genJets, genJetFeatures     = getJet(branches, "GenJets", i, True)
    
      if recoJets.shape[0] == 0 or genJets.shape[0] == 0: 
          continue

      for j in range(recoJets.shape[0]):
        
          match_idx = findGenJet(recoJets[j], genJets)
          if match_idx == -99:
              recoJets[j][6] = 0
              recoJets[j][7] = 0
              recoJets[j][8] = 0
          else:
              genMatch = genJets[match_idx,:]
            
              partonFlavour  = genJets[match_idx,4]
              hadronFlavour  = genJets[match_idx,5]
            
              recoJets[j][6] = partonFlavour
              recoJets[j][7] = hadronFlavour
        
              # Is b
              if math.fabs(partonFlavour) == 5 and math.fabs(hadronFlavour) == 5 :
                  recoJets[j][8] = int(1)
            
              # Is light
              elif 0<math.fabs(partonFlavour)<5 and math.fabs(hadronFlavour) == 0 :
                  recoJets[j][8] = int(2)
            
              # Is gluon
              elif math.fabs(partonFlavour) == 21 and math.fabs(hadronFlavour) == 0 :
                  recoJets[j][8] = int(3)        
              else:
                  recoJets[j][8] = int(0)
      recoJets_all  = np.concatenate((recoJets_all, recoJets),  axis = 0) if recoJets_all.size else recoJets
      recoCands_all = np.concatenate((recoCands_all, recoCands), axis = 0) if recoCands_all.size else recoCands

  print("ALL Reco jets", recoJets_all.shape)
  print("ALL Reco Cands", recoCands_all.shape)
  file_in.close()
  outFile = h5py.File(outFileName, "w")

  outFile.create_dataset('recoJets', data=recoJets_all, compression='gzip', dtype='f')
  outFile.create_dataset('recoCands', data=recoCands_all, compression='gzip', dtype='f')
  outFile.create_dataset('labels', data=recoJets_all[:,-1].T, compression='gzip', dtype='i')
  recoJetFeatures = [st.encode('utf8') for st in recoJetFeatures] 
  outFile.create_dataset('recoJetFeatureNames', data=recoJetFeatures, compression='gzip')
  recoCandFeatures = [st.encode('utf8') for st in recoCandFeatures]
  outFile.create_dataset('recoCandFeatureNames',  data=recoCandFeatures, compression='gzip')
 
  outFile.close()

  print('Number of b-quarks = {}'    .format(len(recoJets_all[(recoJets_all[:,8])==1])))
  print('Number of light quarks = {}'.format(len(recoJets_all[(recoJets_all[:,8])==2])))
  print('Number of gluons = {}'      .format(len(recoJets_all[(recoJets_all[:,8])==3])))
  print('Undefined = {}'             .format(len(recoJets_all[(recoJets_all[:,8])==0])))
        