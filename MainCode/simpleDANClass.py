
# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This file constructs a DAN class that constructs and operates on a DAN object. binaryDataListOfLists should be the output of 
# ExcelDataToListOfLists.py. Similarly for binaryInputList (however binaryInputList must have the apppropriate feature dimension,
# so its recommended to append any inputs to the original dataset when running it through ExcelDataToListOfLists.py if possible)


import numpy as np

class simpleDAN:

    def __init__(self, binaryDataListOfLists):
        self.binaryDataListOfLists = np.array(binaryDataListOfLists)

    def getOutput(self, binaryInputList, binaryOutput=True):
        numpyBinaryInputList = np.array(binaryInputList)

        initialOutput = self.binaryDataListOfLists @ numpyBinaryInputList

        scaled_matrix = self.binaryDataListOfLists * initialOutput[:, np.newaxis]

        finalOutputNonBinary = np.max(scaled_matrix, axis=0)

        if binaryOutput:
            maxVal = np.max(finalOutputNonBinary)
            return (finalOutputNonBinary == maxVal).astype(int).tolist()
        
        return finalOutputNonBinary.tolist()
    
    def getEnergy(self, binaryInputList):
        numpyBinaryInputList = np.array(binaryInputList)

        initialOutput = self.binaryDataListOfLists @ numpyBinaryInputList

        return -np.max(initialOutput)



    



