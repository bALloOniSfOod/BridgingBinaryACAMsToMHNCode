# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2026. 
# See https://www.afbeavers.net/drg for more information

# This file holds the code that evaluates -xln(x) via the continuous approximation theorems


from ContinuousDANClass import ContinuousDAN
from xlnx_data import xlnx_list
from sinnpix_for_xlnx import sinnpix_data, sinnpix_for_50_data
import copy
from copy import deepcopy
from tqdm import tqdm
import random
import pandas as pd

nearestNeighborCode = False
UATCode = True


negative_xlnx_data = xlnx_list
negative_xlnx_data_rounded = deepcopy(negative_xlnx_data)
sinnpix_data = sinnpix_data


if nearestNeighborCode:
    negative_xlnx_data_rounded_50 = []
    negative_xlnx_data_rounded_100 = []
    negative_xlnx_data_rounded_500 = []
    negative_xlnx_data_rounded_1000 = deepcopy(negative_xlnx_data_rounded)

    datasetList = [negative_xlnx_data_rounded_50, negative_xlnx_data_rounded_100, negative_xlnx_data_rounded_500, negative_xlnx_data_rounded_1000]

    for i, element in enumerate(negative_xlnx_data_rounded):
        if i % 20 == 0:
            negative_xlnx_data_rounded_50.append(element)

    for i, element in enumerate(negative_xlnx_data_rounded):
        if i % 10 == 0:
            negative_xlnx_data_rounded_100.append(element)

    for i, element in enumerate(negative_xlnx_data_rounded):
        if i % 2 == 0:
            negative_xlnx_data_rounded_500.append(element)

    for i in range(len(negative_xlnx_data)):
        negative_xlnx_data_rounded[i][1] = round(negative_xlnx_data[i][1], 3)

    inputList = [random.random() for _ in range(50)]

    finalOutputList = []

    for dataset in datasetList:
        finalOutputListHolder = []
        xlnx_DAN = ContinuousDAN(dataset, 500)
        print("trained")
        for input in tqdm(inputList):
            finalOutputListHolder.append(xlnx_DAN.getOutput([input, None])[1])
        print(finalOutputListHolder)
        finalOutputList.append(finalOutputListHolder)

    print("Inputs:")
    print(inputList)
    print("********************************")
    print("Outputs:")
    print(finalOutputList)

if UATCode:
    inputList = [random.random() for _ in range(50)]

    continuousDANList = []

    print("training DANs")

    for sinnpixData in tqdm(sinnpix_for_50_data):
        continuousDANList.append(ContinuousDAN(sinnpixData, 50))

    print("applying inputs")

    accuracyList = [1, 5, 10, 50]

    perAccuracyDANList = []

    for accuracy in tqdm(accuracyList):
        finalDANOutputList = []
        for DANIndex in range(accuracy):
            DANOutputList = []
            for input in inputList:
                DANOutputList.append(continuousDANList[DANIndex].getOutput([input, None])[1])
            finalDANOutputList.append(DANOutputList)

        perAccuracyDANList.append(finalDANOutputList)

    print("summing")

    finalList = []

    for finalDANOutList in tqdm(perAccuracyDANList):
        finalListHolder = []
        for featureIndex in range(50):
            totalSum = 0
            for index, DANOutList in enumerate(finalDANOutList):
                totalSum += DANOutList[featureIndex] * sinnpix_data[index][1]
            finalListHolder.append(totalSum)
        finalList.append(finalListHolder)

    print("inputs:")
    print(inputList)
    print("outputs:")
    print(finalList)

    finalDifferenceList = []

    trueOutput = [random.random() for _ in range(100)]

    for output in finalList:
        finalDifferenceListHolder = []
        for index in range(50):
            finalDifferenceListHolder.append(output[index] - trueOutput[index])
        finalDifferenceList.append(finalDifferenceListHolder)

    for element in range(len(finalDifferenceList)):
        for index in range(len(finalDifferenceList[element])):
            if finalDifferenceList[element][index] < 0:
                finalDifferenceList[element][index] *= -1

    df = pd.DataFrame(finalDifferenceList)
    df.to_excel("ContinuousUAT2.xlsx", index=False, header=False)


