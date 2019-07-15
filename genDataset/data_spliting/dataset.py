import numpy as np
import findOutliers as outliers
import os
import cv2
from enum import Enum
import random


class Dataset:
    class Mode(Enum):
        Mono = 0
        LeftRight = 1
        UpDown = 2
        Trianocular = 3
        SparsePoints = 4

    def __init__(self, name, mode, datasetDirectories, filenamesFileDirectory, filenamesFileName):
        self.overThresh = 8.0
        self.underThresh = 0.5
        self.overPercent = 5
        self.underPercent = 5
        self.pairIds = []
        self.name = name
        self.delimiter = " "
        # For Restyles
        # self.ignoreList = [
        #     "area2",                                # Stanford
        #     "area4",
        #     "fa5f164b48f043c6b2b0bb9e8631a4821",    # Matterport
        #     "a641c3f4647545a2a4f5c50f5f5fbb571",    # Matterport
        #     "e0166dba74ee42fd80fccc26fe3f02c81",    # Matterport
        #     "eb00de2714da4edba8fcd867924c2a271",
        #     "e9510fcbae554d6cb8136a7274521ff31",
        #     "e996abcc45ad411fa7f406025fcf2a631",
        #     "ce4dcbb88c474cc1a0d1a3768062ec5d1",
        #     "602971d3594745e6b1ae71d0a1c6fde61",
        #     "7812e14df5e746388ff6cfe8b043950a1",
        #     "975b8a35009841e6aaec4a0124a3e2ff1",
        #     "0685d2c5313948bd94e920b5b9e1a7b21",
        #     "59e3dc4b2b5848b6a55eb9bc98a42f431",
        #     "2797de2c1a00404faa63b1c722809d0c1",
        #     "65a6bb9bce044fa8bc0c1865820930be1",
        #     "0151156dd8254b07a241a2ffaf0451d41"
        # ]
        # self.validationList = [
        #     "area3",
        #     "85cef4a4c3c244479c56e56d9a723ad21",
        #     "0c334eaabb844eaaad049cbbb2e0a4f21",
        #     "0b724f78b3c04feeb3e744945517073d1",
        #     "fbf6d32ff0e044e88355076d502e160b1",
        #     "edb61af9bebd428aa21a59c4b2597b201"
        # ]
        self.ignoreList = [
            "area3",                                # Stanford
            "fa5f164b48f043c6b2b0bb9e8631a4821",    # Matterport
            "a641c3f4647545a2a4f5c50f5f5fbb571",    # Matterport
            "e0166dba74ee42fd80fccc26fe3f02c81",    # Matterport
            "306f740ae757d660135b3e88cdae39551",    # SunCG
            "308ed5f320429e1295db6c872b27ad0b1"     # SunCG
        ]
        self.validationList = []
        self.dataDict = {}
        self.goodDataDict = {}
        self.trainDict = {}
        self.testDict = {}
        self.validDict = {}
        self.testFilePath = os.path.join(
            filenamesFileDirectory, filenamesFileName + "_evaluation.txt")
        self.trainFilePath = os.path.join(
            filenamesFileDirectory, filenamesFileName + "_train.txt")
        self.validFilePath = os.path.join(
            filenamesFileDirectory, filenamesFileName + "_validation.txt")
        if (mode == "Mono" or mode == "SparsePts"):
            self.mode = self.Mode.Mono
            self.pairIds.append("_Left")
            self.dataDict["_Left_color"] = []
            self.dataDict["_Left_depth"] = []
            self.dataDict["_Left_normals"] = []

            self.goodDataDict["_Left_color"] = []
            self.goodDataDict["_Left_depth"] = []
            self.goodDataDict["_Left_normals"] = []

            self.trainDict["_Left_color"] = []
            self.trainDict["_Left_depth"] = []
            self.trainDict["_Left_normals"] = []

            self.testDict["_Left_color"] = []
            self.testDict["_Left_depth"] = []
            self.testDict["_Left_normals"] = []

            self.validDict["_Left_color"] = []
            self.validDict["_Left_depth"] = []
            self.validDict["_Left_normals"] = []

        if mode == "SparsePts":
            self.mode = self.Mode.SparsePoints
            self.dataDict["_Left_points"] = []
            self.goodDataDict["_Left_points"] = []
            self.trainDict["_Left_points"] = []
            self.testDict["_Left_points"] = []
            self.validDict["_Left_points"] = []

        self.dataDirs = datasetDirectories

    def gatherFiles(self):
        print("Gathering filepaths...")
        for dataDir in self.dataDirs:
            if (not os.path.exists(dataDir)):
                print("Directory: {} does not exist. Skipping.".format(dataDir))
                continue
            filesList = sorted(os.listdir(dataDir))
            for filepath in filesList:
                for pairID in self.pairIds:
                    if pairID in filepath:
                        if "_color_" in filepath and ".png" in filepath:
                            keyType = pairID + "_color"
                            self.dataDict[keyType].append(
                                os.path.join(dataDir, filepath))
                        elif "_depth_" in filepath and ".exr" in filepath:
                            keyType = pairID + "_depth"
                            self.dataDict[keyType].append(
                                os.path.join(dataDir, filepath))
                        elif "_points_" in filepath and ".exr" in filepath and self.mode == self.Mode.SparsePoints:
                            keyType = pairID + "_points"
                            self.dataDict[keyType].append(
                                os.path.join(dataDir, filepath))
                        # elif "_normal_" in filepath and ".exr" in filepath:
                        #     keyType = pairID + "_normals"
                        #     self.dataDict[keyType].append(os.path.join(dataDir, filepath))

    def calcTotalCount(self):
        self.count = 0
        if (self.mode == self.Mode.Mono):
            self.count += len(self.dataDict["_Left_color"]) + \
                len(self.dataDict["_Left_depth"])
        elif self.mode == self.Mode.SparsePoints:
            self.count += len(self.dataDict["_Left_color"]) + len(
                self.dataDict["_Left_depth"]) + len(self.dataDict["_Left_points"])

    def printCountSummary(self):
        print("Dataset {} Summary:".format(self.name))
        print("Number of total filepaths: {}".format(self.calcTotalCount()))
        for pairID in self.pairIds:
            print("Number of " + pairID +
                  " color files: {}".format(len(self.dataDict[pairID + "_color"])))
            print("Number of " + pairID +
                  " depth files: {}".format(len(self.dataDict[pairID + "_color"])))

    def printFileSummary(self):
        print("Dataset {} File Summary:".format(self.name))
        for pairID in self.pairIds:
            print("Number of " + pairID + " color files in Train-set: {}".format(
                len(self.trainDict[pairID + "_color"])))
            print("Number of " + pairID + " depth files in Train-set: {}".format(
                len(self.trainDict[pairID + "_color"])))
            print("Number of " + pairID +
                  " color files in Test-set: {}".format(len(self.testDict[pairID + "_color"])))
            print("Number of " + pairID +
                  " depth files in Test-set: {}".format(len(self.testDict[pairID + "_color"])))

    def outCondition(self, depthMap):
        return (outliers.percentageOverThreshold(depthMap, self.overThresh) > self.overPercent) or (outliers.percentageUnderThreshold(depthMap, self.underThresh) > self.underPercent)

    def getZippedData(self, dictionary):
        if self.mode == self.Mode.Mono:
            zipped = zip(dictionary["_Left_color"], dictionary["_Left_depth"])
        else:
            zipped = zip(
                dictionary["_Left_color"], dictionary["_Left_depth"], dictionary["_Left_points"])
        return zipped

    def removeOutliers(self):
        print("Removing Outliers...")
        # Monocular Mode
        if self.mode == self.Mode.Mono or self.mode == self.Mode.SparsePoints:
            for data in self.getZippedData(self.dataDict):
                depthMap = cv2.imread(data[1], cv2.IMREAD_ANYDEPTH)
                if self.outCondition(depthMap):
                    print("File pair:\n[\t{}\n\t{}\n]is OUT.".format(
                        os.path.basename(data[0]), os.path.basename(data[1])))
                else:
                    self.goodDataDict["_Left_color"].append(data[0])
                    self.goodDataDict["_Left_depth"].append(data[1])
                    if self.mode == self.Mode.SparsePoints:
                        self.goodDataDict["_Left_points"].append(data[2])

    # Splits to Train and Test Dictionaries
    def splitFiles(self):
        for data in self.getZippedData(self.goodDataDict):
            if any(hash in data[0] for hash in self.ignoreList):
                self.testDict["_Left_color"].append(data[0])
                self.testDict["_Left_depth"].append(data[1])
                if self.mode == self.Mode.SparsePoints:
                    self.testDict["_Left_points"].append(data[2])
            elif any(hash in data[0] for hash in self.validationList):
                self.validDict["_Left_color"].append(data[0])
                self.validDict["_Left_depth"].append(data[1])
                if self.mode == self.Mode.SparsePoints:
                    self.validDict["_Left_points"].append(data[2])
            else:
                self.trainDict["_Left_color"].append(data[0])
                self.trainDict["_Left_depth"].append(data[1])
                if self.mode == self.Mode.SparsePoints:
                    self.trainDict["_Left_points"].append(data[2])

    # creates files
    def createFiles(self):
        print("Creating files")
        testFile = open(self.testFilePath, "w")
        trainFile = open(self.trainFilePath, "w")
        validFile = open(self.validFilePath, "w")
        trainLines = []
        testLines = []
        validLines = []

        def printLine(data):
            line = data[0] + self.delimiter + data[1]
            if self.mode == self.Mode.SparsePoints:
                line += self.delimiter + data[2]
            return line + '\n'

        for data in self.getZippedData(self.trainDict):
            trainLines.append(printLine(data))
        for data in self.getZippedData(self.testDict):
            testLines.append(printLine(data))
        for data in self.getZippedData(self.validDict):
            validLines.append(printLine(data))

        random.shuffle(trainLines)
        random.shuffle(testLines)
        random.shuffle(validLines)
        trainFile.writelines(trainLines)
        testFile.writelines(testLines)
        validFile.writelines(validLines)
        trainFile.close()
        testFile.close()
        validFile.close()

    def modeToString(self):
        if (self.mode == self.Mode.Mono):
            return "Monocular"
        elif self.mode == self.Mode.Mono:
            return "Sparse+Mono"
        else:
            return ""

    def printSummary(self):
        print("Generated Dataset:")
        print("Name            : {}".format(self.name))
        print("Dataset Mode    : {}".format(self.modeToString()))
        print("Data Directories: ")
        for dataDir in self.dataDirs:
            print("\t {}".format(dataDir))
        print(
            "Number of file-pairs in train-set: {}".format(len(self.trainDict["_Left_color"])))
        print(
            "Number of file-pairs in test-set : {}".format(len(self.testDict["_Left_color"])))
