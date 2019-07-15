import os
import numpy as np
import dataset
import argparse

# add dataset directory prompt


def addDatasetSirectoryPrompt():
    dataDirs = []
    print("Please add a directory with data...")
    dataDirs.append(input())
    option = 'y'
    while (option == 'y'):
        print("Do you want to add another directory with data? [y/n]")
        option = input()
        if (option == 'y'):
            print("Please add another directory")
            dataDir = input()
            dataDirs.append(dataDir)
        elif (option == 'n'):
            break
        else:
            print("Unknown option. Please select [y/n].")
            option = 'y'

    return dataDirs


# mode selection prompt
def modeSelectionPrompt():
    print("Please select one of the following dataset modes:")
    print("1. Monocular")
    print("2. Mono + Sparse points")
    option = input()
    while (option != '1' and option != '2' and option != '3' and option != '4'):
        print("Unknown option. Please select one of the above (1, 2, 3 or 4)")
        option = input()
    if (option == '1'):
        mode = "Mono"
    if option == '2':
        mode = 'SparsePts'
    return mode

# filenames file directory and name prompt


def filenamesPathPrompt():
    print("Please give the directory to save the filenames files...")
    filenamesDir = input()
    print("Please give a name for the filenames files (suffixes '_train' and '_test' are added automatically):")
    filenamesName = input()
    return filenamesDir, filenamesName


def main():
    parser = argparse.ArgumentParser(
        description='Generates test, train and validation splits for 360D')
    parser.add_argument('-d', '--default', action='store_true', default=False,
                        help='Use preset default paths for datasets')
    args = parser.parse_args()
    print("* 360D filenames file generator *")
    if args.default:
        dataDirectories = [
            # '../../datasets/Matterport/',
            # '../../datasets/Stanford/',
            # '../../datasets/SunCG',
            "../../../datasets/Panodepth/Stanford"

        ]
        #  use for validation:
        # '../../datasets/SceneNet',
    else:
        dataDirectories = addDatasetSirectoryPrompt()

    mode = modeSelectionPrompt()
    filenamesDir, name = filenamesPathPrompt()
    filenamesDir = os.path.abspath(filenamesDir)
    os.makedirs(filenamesDir, exist_ok=True)
    print("output dir", filenamesDir)
    data = dataset.Dataset(name, mode, dataDirectories, filenamesDir, name)
    data.gatherFiles()
    data.printCountSummary()
    # data.removeOutliers()
    data.useAllData()
    data.splitFiles()
    data.printFileSummary()
    data.createFiles()
    data.printSummary()
    print("Done.")


if __name__ == "__main__":
    main()
