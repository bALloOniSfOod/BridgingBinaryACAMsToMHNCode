# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This file imports a dataset of varying shapes, colors, and sizes, trains a DAN on it, and shows that it can reliably retrieve all possible permutations
# of shape, color, and size


from ShapeColorSize import shapeColorSize
from SimpleDANClass import simpleDAN
from ExcelDataToListofLists import ListofListsToBinaryEncodingListOfLists

totalDataset = ListofListsToBinaryEncodingListOfLists(shapeColorSize, returnAllPossibleMembers=20*10*5)

myDAN = simpleDAN(totalDataset)

accuracyList = []
for input in totalDataset:
    theOutput = myDAN.getOutput(input)
    if input == theOutput:
        accuracyList.append(1)
    else:
        accuracyList.append(0)

print("Final Accuracy: ", 100 * (sum(accuracyList) / len(accuracyList)), "%")
