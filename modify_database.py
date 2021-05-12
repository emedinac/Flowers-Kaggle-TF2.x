import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='split training and testing sets')
    parser.add_argument('--path_data',  type=str,  help='a path where dataset file is located')
    parser.add_argument('--dataset_name',  type=str,  help='a path where dataset name is located')
    parser.add_argument('--path_mod',  type=str,  help='a path where path mod is located')
    args = parser.parse_args()

    # Change Class names
    with open(args.dataset_name+"/"+args.dataset_name+".py") as f:
        data = f.read()
    init_txt = "tfds.features.ClassLabel(names="
    end_txt = "]),"
    lblidx1 = data.find(init_txt)
    lblextra = len(init_txt)
    lblidx2 = data[lblidx1+lblextra:].find(end_txt)+1
    ref_txt = data[lblidx1+lblextra:lblidx1+lblextra+lblidx2] # Ref code from tfds (current version) source
    problem_classes = os.listdir(args.path_data) # Target DB
    problem_classes = str([f for f in problem_classes if not ".csv" in f])
    data = data.replace(ref_txt,problem_classes)

    # Change _split_generators and _generate_examples
    init_txt = "    path = dl_manager.download_and_extract"
    end_txt = "def _generate_examples(self, path):"
    lblidx1 = data.find(init_txt)
    # lblextra = len(init_txt)
    # lblidx2 = data[lblidx1+lblextra:].find(end_txt)+1
    ref1_txt = data[lblidx1:-1]


    init_txt = "import tensorflow_datasets as tfds"
    lblidx1 = data.find(init_txt)
    lblextra = len(init_txt)
    lblidx2 = lblextra
    ref_txt = data[lblidx1:lblidx1+lblextra] # Ref code from tfds (current version) source
    problem_classes = "import tensorflow_datasets as tfds\n\
import tensorflow as tf\n\
import csv # Temporal csv reader\n"
    data = data.replace(ref_txt,problem_classes)

    with open(args.path_mod) as f:
        target_data = f.read()
    init_txt = "path ="
    end_txt = " # Set this manually inside dataset folder to avoid automation"
    lblidx1 = target_data.find(init_txt)
    lblextra = len(init_txt)
    lblidx2 = target_data[lblidx1+lblextra:].find(end_txt)+1
    ref2_txt = target_data[lblidx1+lblextra:lblidx1+lblextra+lblidx2]
    target_data = target_data.replace(ref2_txt, "\'"+args.path_data+"\' ")
    data = data.replace(ref1_txt, target_data)
    file = open(args.dataset_name+"/"+args.dataset_name+".py","w")
    file.write(data)
    file.close()