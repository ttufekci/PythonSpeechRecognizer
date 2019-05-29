import os
import pickle
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from hmmlearn import hmm

def trainmodel(model,wavlist):
    X = np.array([])
    for wavfile in wavlist:
        path = os.path.join('trainingwav', wavfile)
        (rate, sig) = wav.read(path)
        mfcc_feat = mfcc(sig, rate, nfft=1024)

        if len(X) == 0:
            X = mfcc_feat
        else:
            X = np.append(X, mfcc_feat, axis=0)

    modelSayi = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=1000)
    modelSayi.fit(X)

    hmmPath = os.path.join('hmm',model)
    file = open(hmmPath,"wb")
    pickle.dump(modelSayi,file)
    file.close()

def main():
    trainmodel('bir', ['1_01.wav','1_02.wav','1_03.wav', '1_04.wav','1_05.wav'])
    trainmodel('iki',['2_01.wav','2_02.wav','2_03.wav', '2_04.wav','2_05.wav'])
    trainmodel('uc',['3_01.wav', '3_02.wav', '3_03.wav', '3_04.wav', '3_05.wav'])
    trainmodel('dort',['4_01.wav','4_02.wav','4_03.wav', '4_04.wav','4_05.wav'])
    trainmodel('bes',['5_01.wav','5_02.wav','5_03.wav','5_04.wav','5_05.wav'])
    trainmodel('alti',['6_01.wav', '6_02.wav', '6_03.wav', '6_04.wav', '6_05.wav'])
    trainmodel('yedi',['7_01.wav', '7_02.wav', '7_03.wav', '7_04.wav', '7_05.wav'])
    trainmodel('sekiz',['8_01.wav', '8_02.wav', '8_03.wav', '8_04.wav', '8_05.wav'])
    trainmodel('dokuz',['9_01.wav','9_02.wav','9_03.wav', '9_04.wav','9_05.wav'])
    trainmodel('on',['10_01.wav', '10_02.wav', '10_03.wav', '10_04.wav', '10_05.wav'])
if __name__ == '__main__':
    main()