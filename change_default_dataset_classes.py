import os
import argparse
import numpy as np
import pandas as pd


def getListOfFiles(dirName, level=True):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath) and level:
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)           
    return allFiles

class DB_generator: # Only works fine for classification. TFDS must become an automatic data loader.
    def __init__(self, path):
        self.path = path
        self.ref_size = len(path)
    def Load_data(self, rename=False):
        self.files = getListOfFiles(self.path)
        if rename:
            for file in self.files:
                if file.endswith("jpg"):
                    os.rename(file, file[:file.find(".jpg")]+".jpeg")
        self.files = [f for f in self.files if ".jpeg" in f and not "csv" in f] # not include csv files
        self.classes = os.listdir(self.path) # Target DB "Database/flowers/"
        self.classes = [f for f in self.classes if not ".csv" in f] # not include csv files
        self.classes_dict = {}
        for i,k in enumerate(self.classes): self.classes_dict[k]=i
        # self.labels = [self.classes_dict[item[self.ref_size:self.ref_size+item[self.ref_size:].find("/")]] for item in self.files]
        self.labels = [item[self.ref_size:self.ref_size+item[self.ref_size:].find("/")] for item in self.files]
    def Split_label_and_data(self, val, seed=0):
        np.random.seed(seed)
        idxs = np.arange(len(self.files))
        size_db = len(self.files)
        np.random.shuffle(idxs)
        files = np.array(self.files)[idxs]
        labels = np.array(self.labels)[idxs]
        self.data_train = files[round(size_db*val):]
        self.data_valid = files[:round(size_db*val)]
        self.label_train = labels[round(size_db*val):]
        self.label_valid = labels[:round(size_db*val)]
    def Get_labels_data(self):
        return self.data_train, self.data_valid, self.label_train, self.label_valid
    def Save_label_and_data(self, output_path):
        # Pandas was included for a more efficient link data storage
        self.data_train = pd.DataFrame({"imgs":self.data_train, "labels":self.label_train})
        self.data_valid = pd.DataFrame({"imgs":self.data_valid, "labels":self.label_valid})
        self.data_train.to_csv(output_path+"training.csv")
        self.data_valid.to_csv(output_path+"validation.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='split training and testing sets')
    parser.add_argument('--path_data',  type=str,  help='a path where data is located')
    parser.add_argument('--output_path',  type=str,  help='a path where data is saved')
    parser.add_argument('--validation', default=0.5, type=float,  help='a float to choose the size of validation set')
    parser.add_argument('--seed', default=0, type=int,  help='an integer to define the seed in numpy random')
    args = parser.parse_args()

    
    DBgenerator = DB_generator(args.path_data)
    DBgenerator.Load_data(True)
    DBgenerator.Split_label_and_data(args.validation, args.seed)
    # DBgenerator.Get_labels_data()
    DBgenerator.Save_label_and_data(args.output_path)

    
    




