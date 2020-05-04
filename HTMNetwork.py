from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory
from nupic.algorithms.anomaly import Anomaly
from nupic.algorithms.backtracking_tm_cpp import BacktrackingTM

import Fileparser
import RetinaOperations
import numpy


class AnomalyScore:
    # a class for preserving the anomaly scores of text consisting of the sentence number of the start of the anomaly, the end of the anomaly and the corresponding score
    def __init__(self, id, startIdx,endIdx,anomalyScore):
        self.id = id
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.anomalyScore = anomalyScore

    def toString(self):
        print("sentenceNum: " + self.id)
        print("startIdx "+ self.startIdx)
        print("startIdx " + self.endIdx)
        print("anomalyScore " + self.anomalyScore)

        # parameters for the spatial pooler and temporal memory networks, only tm_only is used
sp_layer1 = SpatialPooler(
        inputDimensions=(128, 128),
        columnDimensions=(64, 64),
        potentialPct=0.1,
        potentialRadius=5,
        globalInhibition=False,
        localAreaDensity=0.1,
        numActiveColumnsPerInhArea=3,
        synPermInactiveDec=0.5,
        synPermActiveInc=0.02,
        synPermConnected=0.90,
        boostStrength=0.0,
        wrapAround=False

    )



tm_only =  TemporalMemory(
    inputDimensions=(4096,),
    columnDimensions=(4096,),
    cellsPerColumn=5,
    newSynapseCount = 15,
    activationThreshold = 15,
    initialPermanence=0.7,
    connectedPermanence=0.8,
    minThreshold=8,
    maxNewSynapseCount=30,
    permanenceIncrement=0.04,
    permanenceDecrement=0.01,
    predictedSegmentDecrement=0.005,
    maxSegmentsPerCell=255,
    maxSynapsesPerSegment=255,

  )

tm_withSP = TemporalMemory(
    inputDimensions=(4096,),
    columnDimensions=(4096,),
    cellsPerColumn=5,
    newSynapseCount=15,
    activationThreshold=15,
    initialPermanence=0.6,
    connectedPermanence=0.8,
    minThreshold=8,
    maxNewSynapseCount=30,
    permanenceIncrement=0.03,
    permanenceDecrement=0.03,
    predictedSegmentDecrement=0.03,
    maxSegmentsPerCell=255,
    maxSynapsesPerSegment=255,
)

def getActiveAndPredictiveColumnsSPAndTM(fingerprint_list):
    # trains the network and uses it to form predictions for a list of fingerprints
    # version that uses both the spatial pooler and a temporal memory (with the parameters from tm_withSP)
    inputList=[]
    for i in range(len(fingerprint_list)):
        SDR=RetinaOperations.SDRFromFingerprint(fingerprint_list[i])
        inputList.append(SDR)
    print("Training:")
    for i in range(len(fingerprint_list)):
        activeColumns=numpy.zeros(4096)
        sp_layer1.compute(inputList[i], True, activeColumns)
        activeColumnIndices = numpy.nonzero(activeColumns)[0]
        tm_only.compute(activeColumnIndices, learn=True)
        print(i)
        print(len(activeColumnIndices))
        print(len(tm_only.getPredictiveCells()))
        print("-")


    activeColumnList = []
    predictiveColumnList = []
    tm_withSP.reset()
    print("Inference:")


    for i in range(len(fingerprint_list)):
        activeColumns = numpy.zeros(4096)
        sp_layer1.compute(inputList[i], True, activeColumns)
        activeColumnIndices = numpy.nonzero(activeColumns)[0]
        tm_withSP.compute(activeColumnIndices, learn=True)
        print(i)
        print(len(activeColumnIndices))
        print(len(tm_only.getPredictiveCells()))
        print("-")

        activeColumnList.append(activeColumnIndices);
        print(len(activeColumnIndices))

        predictiveCells = tm_only.getPredictiveCells()
        predictiveColumns = []

        for c in predictiveCells:
            idx = tm_only.columnForCell(c)
            predictiveColumns.append(idx)

        predictiveColumnSet = list(dict.fromkeys(predictiveColumns))
        predictiveColumnList.append(predictiveColumnSet);

        print(len(predictiveColumnSet))
        print("-")






        activeColumnList.pop(0)
        predictiveColumnList.pop()

    return activeColumnList, predictiveColumnList


def getAnomalyScores(activeColumnList, predictiveColumnList):
    # calculates the anomaly scores from the list of active and predicted columns at each step - deprecated
    anomalyScoreList = []
    an = Anomaly()
    for i in range(len(activeColumnList)):
        score = an.computeRawAnomalyScore(activeColumnList[i], predictiveColumnList[i])
        anomalyScoreList.append(score)
    return anomalyScoreList


def getActiveAndPredictiveColumnsTMOnly(fingerprint_list):
    # trains the network and uses it to form predictions for a list of fingerprints
    # version that uses both the spatial pooler and a temporal memory (with the parameters from tm_withSP)
    print("Training:")
    for i in range(len(fingerprint_list)):
        splitPrints,numparts, partLength = RetinaOperations.splitFingerprintIntoParts(fingerprint_list[i],4)
        for j in range(numparts):
            tm_only.compute(splitPrints[j], learn=True)
            print(i)
            print(len(splitPrints[j]))
            print(len(tm_only.getPredictiveCells()))
            print("-")
        tm_only.reset()
    activeColumnList=[]
    predictiveColumnList=[]
    print("Inference:")
    for i in range(len(fingerprint_list)):
        splitPrints, numparts, partLength = RetinaOperations.splitFingerprintIntoParts(fingerprint_list[i], 4)
        for j in range(numparts):
            tm_only.compute(splitPrints[j], learn=True )
            activeColumnList.append(splitPrints[j]);
            print(len(splitPrints[j]))
            predictiveCells = tm_only.getPredictiveCells()
            predictiveColumns = []
            for c in predictiveCells:
                idx = tm_only.columnForCell(c)
                predictiveColumns.append(idx)
            predictiveColumnSet = list(dict.fromkeys(predictiveColumns))
            predictiveColumnList.append(predictiveColumnSet);
            print(len(predictiveColumnSet))
            print("-")



    for i in range(4):
        activeColumnList.pop(0)
        predictiveColumnList.pop()

    return activeColumnList, predictiveColumnList




def runNetwork(idx):
    # uses the HTM algorithm on a single text document with calculating the anomaly scores and outputting them to a file
    fingerprint_path = "C:\DiplomaProject\AlmarimiFingerprints\Fingerprints" + str(idx) + ".txt"
    f_path = "C:\DiplomaProject\CleanFingerprints\Fingerprints" + str(idx) + ".txt"
    active_path="C:\DiplomaProject\OutputColumns\ActiveColumns" + str(idx) + ".txt"
    predictive_path="C:\DiplomaProject\OutputColumns\PredictiveColumns" + str(idx) + ".txt"

    fingerprint_list=Fileparser.get_fingerprint_from_file(fingerprint_path)
    RetinaOperations.saveCleanFingerPrints(fingerprint_list,f_path)
    fingerprint_list=Fileparser.get_clean_fingerprints_from_file(f_path)
    fingerprint_list=RetinaOperations.generateWindowOverlapFingerprints(fingerprint_list,5)
    #print(len(fingerprint_list))


    activeColumnList,predictiveColumnList = getActiveAndPredictiveColumnsTMOnly(fingerprint_list)
    with open(active_path, 'w') as f:
        f.truncate(0)
        f.write("0|0|0|0|")
        for i in range(len(activeColumnList)):
            for j in range(len(activeColumnList[i])):
                f.write(str(activeColumnList[i][j])+",")
            f.write("|")
        f.close()
    with open(predictive_path, 'w') as f:
        f.truncate(0)
        f.write("0|0|0|0|")
        for i in range(len(predictiveColumnList)):
            for j in range(len(predictiveColumnList[i])):
                f.write(str(predictiveColumnList[i][j])+",")
            f.write("|")
        f.close()

for i in range(39,40):
    runNetwork(i)





