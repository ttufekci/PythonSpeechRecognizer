import os
import pickle
import sys
from python_speech_features import mfcc

import scipy.io.wavfile as wav

def loadModels():
    global modelBir, modelIki, modelUc, modelDort, modelBes, modelAlti, modelYedi, modelSekiz, modelDokuz, modelOn
    hmmPath = os.path.join('hmm', 'bir')
    file = open(hmmPath, "rb")
    modelBir = pickle.load(file)
    file.close()
    hmmPath = os.path.join('hmm', 'iki')
    file = open(hmmPath, "rb")
    modelIki = pickle.load(file)
    file.close()
    hmmPath = os.path.join('hmm', 'uc')
    file = open(hmmPath, "rb")
    modelUc = pickle.load(file)
    file.close()
    hmmPath = os.path.join('hmm', 'dort')
    file = open(hmmPath, "rb")
    modelDort = pickle.load(file)
    file.close()
    hmmPath = os.path.join('hmm', 'bes')
    file = open(hmmPath, "rb")
    modelBes = pickle.load(file)
    file.close()
    hmmPath = os.path.join('hmm', 'alti')
    file = open(hmmPath, "rb")
    modelAlti = pickle.load(file)
    file.close()
    hmmPath = os.path.join('hmm', 'yedi')
    file = open(hmmPath, "rb")
    modelYedi = pickle.load(file)
    file.close()
    hmmPath = os.path.join('hmm', 'sekiz')
    file = open(hmmPath, "rb")
    modelSekiz = pickle.load(file)
    file.close()
    hmmPath = os.path.join('hmm', 'dokuz')
    file = open(hmmPath, "rb")
    modelDokuz = pickle.load(file)
    file.close()
    hmmPath = os.path.join('hmm', 'on')
    file = open(hmmPath, "rb")
    modelOn = pickle.load(file)
    file.close()

def testWav(file):
    (rate, sig) = wav.read(file)
    mfcc_feat_test = mfcc(sig, rate, nfft=1024)

    modelSkorBir = modelBir.score(mfcc_feat_test)
    modelSkorIki = modelIki.score(mfcc_feat_test)
    modelSkorUc = modelUc.score(mfcc_feat_test)
    modelSkorDort = modelDort.score(mfcc_feat_test)
    modelSkorBes = modelBes.score(mfcc_feat_test)
    modelSkorAlti = modelAlti.score(mfcc_feat_test)
    modelSkorYedi = modelYedi.score(mfcc_feat_test)
    modelSkorSekiz = modelSekiz.score(mfcc_feat_test)
    modelSkorDokuz = modelDokuz.score(mfcc_feat_test)
    modelSkorOn = modelOn.score(mfcc_feat_test)

    t = max(modelSkorBir, modelSkorIki, modelSkorUc, modelSkorDort, modelSkorBes, modelSkorAlti, modelSkorYedi,
            modelSkorSekiz, modelSkorDokuz, modelSkorOn)

    if (t == modelSkorBir):
        return 'bir'
    if (t == modelSkorIki):
        return 'iki'
    if (t == modelSkorUc):
        return 'uc'
    if (t == modelSkorDort):
        return 'dort'
    if (t == modelSkorBes):
        return 'bes'
    if (t == modelSkorAlti):
        return 'alti'
    if (t == modelSkorYedi):
        return 'yedi'
    if (t == modelSkorSekiz):
        return 'sekiz'
    if (t == modelSkorDokuz):
        return 'dokuz'
    if (t == modelSkorOn):
        return 'on'


if __name__ == '__main__':
    loadModels()
    fileName = sys.argv[1]
    testFile = os.path.join('testingwav', fileName)
    print(testWav(testFile))