import os
import sys
import numpy as np
from shutil import copyfile


class OutputSummary:

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.rmse_all = []
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        copyfile(os.path.abspath(sys.argv[0]), self.out_dir + '/main.py')

    def add_outputs(self, outputs):
        self.rmse_all.append(outputs.get_last_rmse())

    def write_summary(self):
        rmse_all = np.asarray(self.rmse_all)
        if rmse_all[0] is not None:
            text_file = open(self.out_dir + '/summary.txt', 'w')
            text_file.write("RMSE\n====\n\n")
            text_file.write("Runs:\n")
            for val in rmse_all:
                text_file.write("  %f\n" % val)
            text_file.write("Mean: %f\n" % np.mean(rmse_all))
            text_file.write("Std:  %f\n" % np.std(rmse_all))
            text_file.close()
        else:
            print("RMSE summary skipped")
