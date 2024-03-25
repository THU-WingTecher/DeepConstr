
## collect the number of constr for each operator
import os
import yaml
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from typing import *

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 17

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE - 1)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

MIN_FAC = 0.85
MAX_FAC = 1.02

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{xfrac}")
# Define the path where to search for YAML file

def load_data(path) :
    data_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check if the file ends with .yaml
            if file.endswith(".yaml"):
                file_path = os.path.join(root, file)
                # Open and parse the YAML file
                with open(file_path, 'r') as stream:
                    try:
                        data = yaml.safe_load(stream)
                        if data.get("rules", None) is not None : # data already trained
                            data_list.append(data)
                    except yaml.YAMLError as exc:
                        print(exc)
    print(len(data_list))
    return data_list


"""
args:
  dtype:
  - Tensor
  - Tensor
  is_pos:
  - true
  - false
  name:
  - self
  - other
  required:
  - true
  - true
name: torch.Tensor.atan2
package: torch
pass_rate: 17.4
rules:
- - cot: ''
    length: 1
    target:
      choosen_dtype:
        other: tensor
        self: tensor
      msg: The size of tensor a (2) must match the size of tensor b (9) at non-singleton
        dimension 5
      package: torch
    txt: self.shape[5] == other.shape[5]
  - f1_score: 69.06077348066299
    overall_score: 100
    precision: 100.0
    recall: 52.742616033755276
- - cot: divided
    length: 1
    target:
      choosen_dtype:
        other: tensor
        self: tensor
      msg: 'Too large tensor shape: shape = [6, 9, 7, 8, 9, 9, 9, 9, 9]'
      package: torch
    txt: self.shape == other.shape
  - f1_score: 67.79661016949153
    overall_score: 100
    precision: 100.0
    recall: 51.28205128205129

"""
def get_constr_stats(data_list):
    constr_stats = {
        "synthesized": 0,
        "divided": 0,
        "other": 0
    }
    constr_len = []
    constr_operator = {}
    constr_f1 = []
    constr_prec = []
    constr_recall = []
    for data in data_list:
        for rule in data["rules"]:
            if "synthesized" == rule[0]["cot"]:
                constr_stats["synthesized"] += 1
            elif "divided" == rule[0]["cot"]:
                constr_stats["divided"] += 1
            else:
                constr_stats["other"] += 1
            constr_len.append(rule[0].get("length", 1))
            constr_f1.append(rule[1]["f1_score"])
            constr_prec.append(rule[1]["precision"])
            constr_recall.append(rule[1]["recall"])
    return constr_stats, constr_len, constr_operator, constr_f1, constr_prec, constr_recall

# -> visualize distribution of length of cosntr, f1_ prec_ recall of constr, pie chart of constr type
def get_num_of_constr_with_pass_rate(data_list) :
    pass_rate_num_of_constr = []
    for data in data_list:
        pass_rate_num_of_constr.append(
            (data["pass_rate"], len(data["rules"]))
        )
    return pass_rate_num_of_constr

def viz_gen_way_of_constrs(constr_stats):
    print(constr_stats)
    for key in constr_stats.keys():
        print("key", constr_stats[key] / sum(constr_stats.values()))

def viz_constr_len(constr_len : List[int]):
    plt.hist(constr_len, bins=range(1, max(constr_len) + 1))
    plt.show()

colors = ['green', 'pink']
plt.rcParams['text.usetex'] = True
def viz_constr_f1(constr_f1 : List[float], constr_prec, acc_f1, acc_prec , path = "/results/"):
    plt.figure(figsize=(10, 5))
    # PyTorch
    plt.subplot(1, 2, 1)
    plt.scatter(acc_f1, acc_prec, c=colors[1], label=r'\textsc{DeepConstr$^{s}$}', alpha=0.6, edgecolors='w', linewidths=0.5)
    plt.scatter(constr_f1, constr_prec, c=colors[0], label=r'\textsc{DeepConstr}', alpha=0.6, edgecolors='w', linewidths=0.5)
    plt.title('\\textit{PyTorch}')
    plt.xlabel('\\textit{Soundness}')
    plt.ylabel('\\textit{Completeness}')
    plt.legend()
    plt.savefig(path + f"dist-time.png")
if __name__ == "__main__" : 
    path = "/artifact/data/records/torch"
    acc_path = "/artifact/data/only_acc/torch"
    save_path = "/artifact/results/"
    data_list = load_data(path)
    acc_data_list = load_data(acc_path)
    constr_stats, constr_len, constr_operator, constr_f1, constr_prec, constr_recall = get_constr_stats(data_list)
    acc_constr_stats, acc_constr_len, acc_constr_operator, acc_constr_f1, acc_constr_prec, acc_constr_recall = get_constr_stats(acc_data_list)
    print("all")
    viz_gen_way_of_constrs(constr_stats)
    print("only_acc")
    viz_gen_way_of_constrs(acc_constr_stats)
    viz_constr_f1(constr_recall, constr_prec, acc_constr_recall, acc_constr_prec, save_path)