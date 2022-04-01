import math
import csv

def schoolToNum(school):
    if school == "GP":
        return 0
    else:
        return 1

def sexToNum(sex):
    if sex == "F":
        return 0
    else:
        return 1

def addressToNum(address):
    if address == "U":
        return 0
    else:
        return 1

def famsizeToNum(famSize):
    if famSize == "LE3":
        return 0
    else:
        return 1

def pstatusToNum(pStatus):
    if pStatus == "T":
        return 0
    else:
        return 1

def jobToNum (job):
    if job == "teacher":
        return 0
    elif job == "health":
        return 1
    elif job == "services":
        return 2
    elif job == "at_home":
        return 3
    else:
        return 4

def reasonToNum(reason):
    if reason == "home":
        return 0
    elif reason == "reputation":
        return 1
    elif reason == "course":
        return 2
    else:
        return 3

def guardianToNum(guardian):
    if guardian == "mother":
        return 0
    elif guardian == "father":
        return 1
    else:
        return 2

def binaryWordsToNum(b):
    if b == "no":
        return 0
    else:
        return 1

def cosineSimilarity(modelOccurVect, curVect, ids):
    similarities = []

    for indx, val in enumerate(modelOccurVect):
        numerator = 0
        denominatorCur = 0
        denominatorModel = 0

        for j, val in enumerate(curVect):
            numerator += curVect[j] * modelOccurVect[indx][j]
            denominatorCur += curVect[j] ** 2
            denominatorModel += modelOccurVect[indx][j] ** 2

        denom = math.sqrt(denominatorCur) * math.sqrt(denominatorModel)

        if numerator == 0 or denom == 0:
            similarities.append({"sim": 0, "id": ids[indx]})
        else:
            similarities.append({"sim": numerator / denom, "id": ids[indx]})

    return similarities

def validNeighbours(sim, nSize, threshold):
    finalNeigh = []

    for indx, val in enumerate(sim):
        if (val['sim'] > threshold):
            finalNeigh.append(val)

    finalNeigh = sorted(finalNeigh, key = lambda d: d['sim'], reverse= True)

    return finalNeigh[ : nSize]

def categorization(neighbours, modelOccurVect, truthVals):
    count = 0

    for indx, val in enumerate(neighbours):
        closeNeigh = val['id']
        count += truthVals[closeNeigh]

    return count / len(neighbours)

def rmserr(predGrade, truth):
    return math.sqrt((truth - predGrade) ** 2)

def createOutFile(fileName):
    header = ["studentID","G3"]

    with open(fileName, mode='w',  newline='',  encoding='utf-8') as doc:
        doc = csv.writer(doc, delimiter=',')
        doc.writerow(header)

def writeResults(id, pred, fileName):
    row = [id,pred]

    with open(fileName, mode='a',  newline='',  encoding='utf-8') as doc:
        doc = csv.writer(doc, delimiter=',')
        doc.writerow(row)

def parseDataTrain(fileName):
    f1 = open(fileName)

    modelOccurVect = []
    truthVals = []
    ids = []

    first = True
    for line in f1:
        if first != True:
            line = line.strip().split(",")
            vec = []

            for indx, attr in enumerate(line):
                if indx == 1:
                    vec.append(schoolToNum(attr))
                elif indx == 2:
                    vec.append(sexToNum(attr))
                elif indx == 4:
                    vec.append(addressToNum(attr))
                elif indx == 5:
                    vec.append(famsizeToNum(attr))
                elif indx == 6:
                    vec.append(pstatusToNum(attr))
                elif indx == 9 or indx == 10:
                    vec.append(jobToNum(attr))
                elif indx == 11:
                    vec.append(reasonToNum(attr))
                elif indx == 12:
                    vec.append(guardianToNum(attr))
                elif (indx >= 16 and indx <= 23) or indx == 34 or indx == 35:
                    vec.append(binaryWordsToNum(attr))
                elif (indx == 33):
                    truthVals.append(int(attr))
                elif indx != 0:
                    vec.append(int(attr))
                else:
                    ids.append(int(attr))
            modelOccurVect.append(vec)
        else:
            first = False

    return modelOccurVect, truthVals, ids

def parseDataTest(fileName):
    f1 = open(fileName)

    modelOccurVect = []
    ids = []

    first = True
    for line in f1:
        if first != True:
            line = line.strip().split(",")
            vec = []

            for indx, attr in enumerate(line):
                if indx == 1:
                    vec.append(schoolToNum(attr))
                elif indx == 2:
                    vec.append(sexToNum(attr))
                elif indx == 4:
                    vec.append(addressToNum(attr))
                elif indx == 5:
                    vec.append(famsizeToNum(attr))
                elif indx == 6:
                    vec.append(pstatusToNum(attr))
                elif indx == 9 or indx == 10:
                    vec.append(jobToNum(attr))
                elif indx == 11:
                    vec.append(reasonToNum(attr))
                elif indx == 12:
                    vec.append(guardianToNum(attr))
                elif (indx >= 16 and indx <= 23) or indx == 33 or indx == 34:
                    vec.append(binaryWordsToNum(attr))
                elif indx != 0:
                    vec.append(int(attr))
                else:
                    ids.append(int(attr))
            modelOccurVect.append(vec)
        else:
            first = False

    return modelOccurVect, ids

def predictionValidation(testSet,  testIds, testSetTruthVals, modelOccurVect, ids, truthVals, kNeighbours, threshold):
    rmse = 0

    for indx, val in enumerate(testSet):
        simVector = cosineSimilarity(modelOccurVect, testSet[indx], ids)
        neighbours = validNeighbours(simVector, kNeighbours, threshold)

        if len(neighbours) != 0:
            predGrade = categorization(neighbours, modelOccurVect, truthVals)
        else:
            denom = 0
            predGrade = 0

            if val[32] == 1:
                predGrade += val[30]
                denom += 1
            if val[33] == 1:
                predGrade += val[31]
                denom += 1

            predGrade = predGrade / denom

        if predGrade > 20:
            predGrade = 20
        if predGrade < 0:
            predGrade = 0

        writeResults(testIds[indx], predGrade, "predictionResults.csv")

        error = rmserr(predGrade, testSetTruthVals[indx])
        rmse += error

    print (math.sqrt(rmse / len(testSet)))

def predictionTest(testSet,  testIds, modelOccurVect, ids, truthVals, kNeighbours, threshold):
    for indx, val in enumerate(testSet):
        simVector = cosineSimilarity(modelOccurVect, testSet[indx], ids)
        neighbours = validNeighbours(simVector, kNeighbours, threshold)

        if len(neighbours) != 0:
            predGrade = categorization(neighbours, modelOccurVect, truthVals)
        else:
            denom = 0
            predGrade = 0

            if val[32] == 1:
                predGrade += val[30]
                denom += 1
            if val[33] == 1:
                predGrade += val[31]
                denom += 1

            predGrade = predGrade / denom

        if predGrade > 20:
            predGrade = 20
        if predGrade < 0:
            predGrade = 0

        writeResults(testIds[indx], predGrade, "testResults.csv")

def main():
    modelOccurVect, truthVals, ids = parseDataTrain("data/train_data.csv")
    testSet, testIds = parseDataTest("data/test_data.csv")

    #Splitting train data into training set and validation set

    validationSet = modelOccurVect[668:]
    validationIds = ids[668: ]
    validationSetTruthVals = truthVals[668:]

    modelOccurVect = modelOccurVect[0:668]
    ids = ids[0: 668]
    truthVals = truthVals[0: 668]

    threshold = 0.95
    kNeighbours = 12

    createOutFile("predictionResults.csv")
    predictionValidation(validationSet, validationIds, validationSetTruthVals, modelOccurVect, ids, truthVals, kNeighbours, threshold)

    createOutFile("testResults.csv")
    predictionTest(testSet, testIds, modelOccurVect, ids, truthVals, kNeighbours, threshold)

main()
