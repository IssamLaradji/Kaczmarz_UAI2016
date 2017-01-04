import dataset_utils as du
import numpy as np
import sys
import argparse
import pandas as pd
import os
from kaczmarz import Kaczmarz

from pretty_plot import pretty_plot
from itertools import product
import json
import shlex


def save_csv(path, csv_file):
    create_dirs(path)
    csv_file.to_csv(path + ".csv", index=False) 

    print "csv file saved in %s" % (path)


def create_dirs(fname):
    if "/" not in fname:
        return
        
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass 

if __name__ == "__main__":
    np.random.seed(0)
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default ="exp1")  
    parser.add_argument('-s','--selection_rules', nargs='+', default =["greedy_res"])
    parser.add_argument('-o','--objectives', nargs='+', default=["squared_loss"])
    parser.add_argument('-n','--n_iters', default=10, type=int)
    parser.add_argument('-fig','--show_fig', type=int, default=1)
    parser.add_argument('-e','--experiment', default=None)
    parser.add_argument('-t','--timeit', type=int, default=0)
    parser.add_argument('-v','--verbose', type=int, default=0)
    parser.add_argument('-title','--title', default=None)
    parser.add_argument('-ylabel','--ylabel', default=None)
    parser.add_argument('-xlabel','--xlabel', default=None)

    io_args = parser.parse_args()
    exp = io_args.experiment

    ### 0. GET ALGORITHM PAIRS
    if exp != None:
        # LOAD EXPERIMENTS
        with open('experiments.json') as data_file:
            exp_dict = json.loads(data_file.read())

        argString = exp_dict[exp]
        io_args = parser.parse_args(shlex.split(argString))

    dataset_name = io_args.dataset

    selection_rules = io_args.selection_rules        
    objectives = io_args.objectives
    
    n_iters = io_args.n_iters + 1
    verbose = io_args.verbose
    timeit = io_args.timeit
    title = io_args.title
    xlabel = io_args.xlabel
    ylabel = io_args.ylabel

    # 1. Load Dataset
    dataset = du.load_dataset(dataset_name)
    A = dataset["A"]
    b = dataset["b"]
        
    for objective in objectives:
        results = pd.DataFrame()
        for s_rule in selection_rules:
            name = "%s" % (s_rule)
            print "\n%s\n" %(name)

            np.random.seed(1)

            clf = Kaczmarz(selection_rule=s_rule, objective=objective,
                           verbose=verbose, n_iters=n_iters)
            clf.fit(A, b)

            results[name] = clf.results

        pp = pretty_plot.PrettyPlot(title=title, ylabel=ylabel, xlabel=xlabel)
        pp.plot_DataFrame(results / results.max().max())
        pp.show()

        if exp is not None:
            fpath = ("experiments/%s" % (exp))
            create_dirs(fpath)
            save_csv(fpath, results)
            pp.save(fpath)

        else:
            pp.show()