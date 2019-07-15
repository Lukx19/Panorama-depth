import numpy as np
import findOutliers as outliers
import os
import os.path as osp
# import cv2
from enum import Enum
import random
from glob import glob
from skimage import io


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
            "area_3",                                # Stanford
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
            self.pairIds.append("")
            self.dataset_paths = []
            self.good_dataset_paths = []
            self.train_dataset_paths = []
            self.test_dataset_paths = []
            self.validate_dataset_paths = []

            self.dataDict["_color"] = []
            self.dataDict["_depth"] = []
            # self.dataDict["_normals"] = []

            self.goodDataDict["_color"] = []
            self.goodDataDict["_depth"] = []
            # self.goodDataDict["_normals"] = []

            self.trainDict["_color"] = []
            self.trainDict["_depth"] = []
            # self.trainDict["_normals"] = []

            self.testDict["_color"] = []
            self.testDict["_depth"] = []
            # self.testDict["_normals"] = []

            self.validDict["_color"] = []
            self.validDict["_depth"] = []
            # self.validDict["_normals"] = []

        if mode == "SparsePts":
            self.mode = self.Mode.SparsePoints
            self.dataDict["_points"] = []
            self.goodDataDict["_points"] = []
            self.trainDict["_points"] = []
            self.testDict["_points"] = []
            self.validDict["_points"] = []

        self.dataDirs = datasetDirectories

    def gatherFiles(self):
        print("Gathering filepaths...")
        for data_dir in self.dataDirs:
            dataset_folder, dataset_name = osp.split(data_dir)
            dir_name_length = len(dataset_folder) + 1
            color_files = sorted(glob(data_dir + '/**/*_color_*.png', recursive=True))
            depth_files = sorted(glob(data_dir + '/**/*_depth_*.tiff', recursive=True))
            for color_file, depth_file in zip(color_files, depth_files):
                self.dataDict["_color"].append(color_file[dir_name_length:])
                self.dataDict["_depth"].append((dataset_folder, depth_file[dir_name_length:]))

            if self.mode == self.Mode.SparsePoints:
                pts_files = sorted(glob(data_dir + '/**/*_points_*.tiff', recursive=True))
                for pts_file in pts_files:
                    self.dataDict["_points"].append(pts_file[dir_name_length:])

    def calcTotalCount(self):
        self.count = 0
        if (self.mode == self.Mode.Mono):
            self.count += len(self.dataDict["_color"]) + \
                len(self.dataDict["_depth"])
        elif self.mode == self.Mode.SparsePoints:
            self.count += len(self.dataDict["_color"]) + len(
                self.dataDict["_depth"]) + len(self.dataDict["_points"])

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
            zipped = zip(dictionary["_color"], dictionary["_depth"])
        else:
            zipped = zip(
                dictionary["_color"], dictionary["_depth"], dictionary["_points"])
        return zipped

    def removeOutliers(self):
        print("Removing Outliers...")
        # Monocular Mode
        if self.mode == self.Mode.Mono or self.mode == self.Mode.SparsePoints:
            for data in self.getZippedData(self.dataDict):
                depthMap = io.imread(osp.join(*data[1]))

                if self.outCondition(depthMap):
                    print("File pair:\n[\t{}\n\t{}\n]is OUT.".format(
                        os.path.basename(data[0]), os.path.basename(data[1][1])))
                else:
                    self.goodDataDict["_color"].append(data[0])
                    self.goodDataDict["_depth"].append(data[1][1])

                    if self.mode == self.Mode.SparsePoints:
                        self.goodDataDict["_points"].append(data[2])

    def useAllData(self):
        for data in self.getZippedData(self.dataDict):
            self.goodDataDict["_color"].append(data[0])
            self.goodDataDict["_depth"].append(data[1][1])
            if self.mode == self.Mode.SparsePoints:
                self.goodDataDict["_points"].append(data[2])

    # Splits to Train and Test Dictionaries
    def splitFiles(self):
        for data in self.getZippedData(self.goodDataDict):
            if any(hash in data[0] for hash in self.ignoreList):
                self.testDict["_color"].append(data[0])
                self.testDict["_depth"].append(data[1])
                if self.mode == self.Mode.SparsePoints:
                    self.testDict["_points"].append(data[2])
            elif any(hash in data[0] for hash in self.validationList):
                self.validDict["_color"].append(data[0])
                self.validDict["_depth"].append(data[1])
                if self.mode == self.Mode.SparsePoints:
                    self.validDict["_points"].append(data[2])
            else:
                self.trainDict["_color"].append(data[0])
                self.trainDict["_depth"].append(data[1])
                if self.mode == self.Mode.SparsePoints:
                    self.trainDict["_points"].append(data[2])

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
            "Number of file-pairs in train-set: {}".format(len(self.trainDict["_color"])))
        print(
            "Number of file-pairs in test-set : {}".format(len(self.testDict["_color"])))
