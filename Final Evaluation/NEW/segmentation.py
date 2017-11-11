# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 00:02:56 2017

@author: Manesh
"""

import matlab.engine
import numpy as np
from audioFeatureExtraction import getFeatureForSegment
from collections import Counter
import pickle

class Segmentor:
    
    def __init__(self):
        self.eng = self.startMatlab()
    
    def startMatlab(self):
        return matlab.engine.start_matlab()
    
    def getSegments(self, fileName):
        segments = self.eng.getSegments(fileName)
        results = []
        for i in segments:
            segment = np.array(i).flatten()
            features = getFeatureForSegment(segment,44100)
            if(outlier_detector.predict(features)[0]==-1):
                results.append(-1)
                continue
            results.append(classifier.predict(features)[0])
        return results

def writeNormal():
    f = open("data.txt","wb")
    f.write("NORMAL???")
    f.close()
    
def writeProcessing():
    f = open("data.txt","wb")
    f.write("PROCESSING???")
    f.close()

def writeResult(bird, Accurecy):
    data = "RESULT?" + str(bird) + "?" + str(Accurecy) + "?"
    f = open("data.txt","wb")
    f.write(data)
    f.close()
    
def processPrediction(results):
    birdType = Counter(results).most_common(1)[0][0]
    Probability = Counter(results).most_common(1)[0][1]*100.0/len(results)
    writeResult(birdType,round(Probability, 2))
    print birdType," with ", Probability, "%"

segmentor = Segmentor()
classifier = pickle.load(open('techno_model.sav', 'rb'))
#outlier_detector = pickle.load(open('outlier_detector.sav', 'rb'))
outlier_detector = pickle.load(open('outlier_detector_IF.sav', 'rb'))
#pathToAudio = "G:\FYP Project\Final Evaluation\\NEW\\0\\seg.wav"
pathToAudio = "G:\FYP Project\Final Evaluation\\NEW\\3\\9.wav"
results = segmentor.getSegments(pathToAudio)
processPrediction(results)
   
   