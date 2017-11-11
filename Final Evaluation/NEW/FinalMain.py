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
import mysql.connector
from mysql.connector import errorcode
import time

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

class sql:

    def __init__(self):
        try:
            self.cnx = mysql.connector.connect(user='root',password='',database='auralz')
            print 'connection successful'
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)
            

    def query(self,sql):
        cursor = self.cnx.cursor()
        cursor.execute(sql)
        return cursor.fetchone()

    def rows(self):
        return cursor.rowcount
    
    def closedb(self):
        self.cnx.close()
    
    def getRoI(self):
        cursor = self.cnx.cursor()
        query = ("SELECT id, filePath FROM `roi` WHERE status = 0 ORDER BY timeStamp LIMIT 1")
        cursor.execute(query)
        try:
            (roiid, filePath) = cursor.fetchone()
            cursor.close()
        except:
            print "nothing to fetch"
            return None
        else:
            print "id = "+str(roiid)
            print "file path = "+filePath
            return roiid , filePath

    def updateRes(self,roiid,birdType,prob): 
        cursor = self.cnx.cursor()
        updateQuery = """UPDATE `roi` SET `result` = %s, `accuracy` = %s, `status` = '1' WHERE `roi`.`id` = %s"""
        cursor.execute(updateQuery, (int(birdType),prob,roiid))
        self.cnx.commit()
        cursor.close()
    
def processPrediction(results):
    birdType = Counter(results).most_common(1)[0][0]
    Probability = Counter(results).most_common(1)[0][1]*100.0/len(results)
    print birdType," with ", Probability, "%"
    return birdType, Probability

#intialization
segmentor = Segmentor()
classifier = pickle.load(open('techno_model.sav', 'rb'))
outlier_detector = pickle.load(open('outlier_detector_IF.sav', 'rb'))

while True:
    #read from database
    sqldb =  sql()
    temp = sqldb.getRoI()
    if (temp != None):
        roiid , pathToAudio = temp
        results = segmentor.getSegments(pathToAudio)
        birdType, Probability = processPrediction(results)
        sqldb.updateRes(roiid,birdType, Probability)
    else:
        time.sleep(1)
    sqldb.closedb()
   