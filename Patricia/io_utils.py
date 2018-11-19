############################################
# For file input and output
#      Patricia at 11.18, 2018
############################################

import pandas as pd
import os
import numpy as np

class IO:
    def __init__(self, data_directory="./", out_directory="./"):
        self.data_directory = data_directory
        self.out_directory = out_directory
    def load_file(self, filename):
        filepath = os.path.join(self.data_directory, filename)
        return pd.read_csv(filepath)
    def load_users(self, user_file = "users.csv"):
        return self.load_file(user_file)
    def load_business(self, buss_file = "business.csv"):
        return self.load_file(buss_file)
    def load_train(self, train_file="train_reviews.csv"):
        return self.load_file(train_file)
    def load_train_simple(self, train_file="train_reviews.csv"):
        train_set = self.load_train(train_file=train_file)
        uid_lst = train_set["user_id"]
        bid_lst = train_set["business_id"]
        label_lst = train_set["stars"]
        return uid_lst, bid_lst, label_lst
    def load_valid(self, valid_file="validate_queries.csv"):
        valid_set = self.load_file(valid_file)
        valid_x = valid_set.loc[:, ["user_id", "business_id"]]
        valid_y = valid_set["stars"]
        return valid_x, valid_y
    def load_test(self, test_file="test_queries.csv"):
        return self.load_file(test_file)
    def write_file(self, dataframe, outfile_name):
        outfile_path = os.path.join(self.out_directory, outfile_name)
        dataframe.to_csv(outfile_path)

    def evaluate(self, y_true, y_pred): # the l_2 difference
        diff = y_true - y_pred
        print(diff)
        return np.dot(diff.T, diff) / float(len(y_true))
