import numpy
import Fileparser
import RetinaOperations
from statistics import mean,stdev
import scipy

class AnomalyScore:
    # a class for preserving the anomaly scores of text consisting of the sentence number of the start of the anomaly, the end of the anomaly, the corresponding score and the text
    # contained within
    def __init__(self, id, startIdx,endIdx,anomalyScore,text):
        self.id = id
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.anomalyScore = anomalyScore
        self.text=text

    def toString(self):
        print("sentenceNum: " + str(self.id))
        print("startIdx "+ str(self.startIdx))
        print("startIdx " + str(self.endIdx))
        print("anomalyScore " + str(self.anomalyScore))
        print("text " + str(self.text))


def returnAnomalyScore(activeColumns, predictedColumns):
    # calculates the anomaly score from the list of active and predicted columns present at a step
    activeNum=len(activeColumns)
    if activeNum > 0:
        overlapNum = len(RetinaOperations.getFingerprintOverlap(activeColumns,predictedColumns))
        anomalyScore=(activeNum - overlapNum) / float(activeNum)
    else:
        anomalyScore=0.0
    return anomalyScore

def getAnomalyScores(activeColumnsList,predictedColumnsList):
    # calculates the anomaly scores from a list containing the active and predicted columns of every step
    scoreList=[]
    for i in range(len(activeColumnsList)):
        score = returnAnomalyScore(activeColumnsList[i],predictedColumnsList[i])
        scoreList.append(score)
    return scoreList


def getAnomalyScoresMovingAverage(anomalyScores,windowSize):
    # calculates the moving average of the anomaly scores within a window - deprecated
    movingAverageScores=[]
    windowScores=[]
    sum=0
    for i in range(windowSize):
        windowScores.append(anomalyScores[i])
        movingAverageScores.append(mean(windowScores))

    for i in range(windowSize,len(anomalyScores)):
        windowScores.pop(0)
        windowScores.append(anomalyScores[i])


        movingAverageScores.append(mean(windowScores))
    return movingAverageScores

def calculateQfunction(previousScores,currentScore):
    # calculates the q function of the previous anomaly scores and the current anomaly score
    windowMean=mean(previousScores)

    windowStDev=stdev(previousScores)
    if(windowStDev==0):
        return 0
    x=(currentScore-windowMean)/windowStDev
    qVal = scipy.stats.norm.sf(x)
    return qVal


def getAnomalyLikelihoods(anomalyScores,windowSize):
    # calculates the anomaly likelihood scores using a window of a certain size
    likelyhoodScores = []

    windowScores = []
    windowScores.append(anomalyScores[0])
    for i in range(2,windowSize+1):
        windowScores.append(anomalyScores[i-1])
        qVal=calculateQfunction(windowScores,anomalyScores[i])
        likelyhoodScores.append(1-qVal)

    for i in range(windowSize+1, len(anomalyScores)):
        windowScores.pop(0)
        windowScores.append(anomalyScores[i-1])
        qVal = calculateQfunction(windowScores, anomalyScores[i])
        likelyhoodScores.append(1-qVal)

    likelyhoodScores.insert(0,0)
    likelyhoodScores.insert(0, 0)

    return likelyhoodScores


def runDetector(idx, windowSize):
    # for a document, retreives the list of active and predictive columns at every step,
    # calculates the corresponding anomaly and likelihood scores depending of the input windowsize
    active_path = "C:\DiplomaProject\OutputColumns\ActiveColumns" + str(idx) + ".txt"
    predictive_path = "C:\DiplomaProject\OutputColumns\PredictiveColumns" + str(idx) + ".txt"
    anomaly_path="C:\DiplomaProject\AnomalyScores\AnomalyScores" + str(idx) + ".txt"
    likelyHood_path="C:\DiplomaProject\LikelyHoodScores\LikelyHoodScores" + str(idx) + ".txt"
    activeColumnList=Fileparser.get_fingerprint_from_file(active_path)
    predictiveColumnList=Fileparser.get_fingerprint_from_file(predictive_path)

    activeColumnList,predictiveColumnList=RetinaOperations.mergeSplitIndices(activeColumnList,predictiveColumnList,4)
    print(len(activeColumnList))
    print(len(predictiveColumnList))
    scoreList = getAnomalyScores(activeColumnList, predictiveColumnList)
    likelyHoodList = getAnomalyLikelihoods(scoreList, windowSize)
    with open(anomaly_path, 'w') as scoreFile:
        for i in range(len(scoreList)):
            scoreFile.write(str(scoreList[i]) + "|")
    with open(likelyHood_path, 'w') as likelyHoodFile:
        for i in range(len(likelyHoodList)):
            likelyHoodFile.write(str(likelyHoodList[i]) + "|")

for i in range(39,40):
    runDetector(i,5)




