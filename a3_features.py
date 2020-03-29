import os
import sys
from glob import glob
import argparse
import numpy as np
import pandas as pd
# Whatever other imports you need
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import csv
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    
    # Do what you need to read the documents here.
    author_folder_path = glob("{}/*".format(args.inputdir))

    all_file_path = []
    for author in author_folder_path:
        all_file_path += glob("{}/*".format(author))

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))

    words_index_dict = {}
    words_list = []
    author_name_list = []
    for filename in all_file_path:
        author_name_list.append(filename.split('/')[-2])
        with open(filename, "r") as thefile:
            file_words = []
            for line in thefile:
                words_tmp = line.lower().split()
                words = [word for word in words_tmp if word.isalpha()]
                file_words += words
                i = 0
                for word in words:
                    if word not in words_index_dict:
                        words_index_dict[word] = i
                        i += 1
            words_list.append(file_words)

    unique_word_list = tuple(words_index_dict.keys())

    # create feature ndarray, rows represent author files, columns represent features
    features = np.zeros((len(all_file_path), len(unique_word_list)))

    file_index = 0  
    for words in words_list:
        file_list = np.zeros(len(unique_word_list))

        for word in words:
            if word in words_index_dict.keys():
                index = words_index_dict[word]
                file_list[index] += 1

        features[file_index] = file_list
        file_index += 1

    svd = TruncatedSVD(n_components=args.dims)
    X = svd.fit_transform(features)
    y = author_name_list
    test = args.testsize/100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test)

    # build table
    table_train = pd.DataFrame(X_train)
    table_train.insert(0, "data_type", "train", True)
    table_train.insert(0, "author", y_train, True)
    
    table_test = pd.DataFrame(X_test)
    table_test.insert(0, "data_type", "test", True)
    table_test.insert(0, "author", y_test, True)

    table = pd.concat([table_train, table_test])

    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    table.to_csv(args.outputfile)

    print("Done!")
    
