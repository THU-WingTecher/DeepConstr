
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
plt.rc("xtick", labelsize=23)  # fontsize of the tick labels
plt.rc("ytick", labelsize=23)  # fontsize of the tick labels
plt.rc("legend", fontsize=BIGGER_SIZE - 1)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

MIN_FAC = 0.85
MAX_FAC = 1.02

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{xfrac}")
colors = ['blue', 'red']
plt.rcParams['text.usetex'] = True
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
  - LLM
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
        LLM: tensor
        self: tensor
      msg: The size of tensor a (2) must match the size of tensor b (9) at non-singleton
        dimension 5
      package: torch
    txt: self.shape[5] == LLM.shape[5]
  - f1_score: 69.06077348066299
    overall_score: 100
    precision: 100.0
    recall: 52.742616033755276
- - cot: divided
    length: 1
    target:
      choosen_dtype:
        LLM: tensor
        self: tensor
      msg: 'Too large tensor shape: shape = [6, 9, 7, 8, 9, 9, 9, 9, 9]'
      package: torch
    txt: self.shape == LLM.shape
  - f1_score: 67.79661016949153
    overall_score: 100
    precision: 100.0
    recall: 51.28205128205129

"""
def get_constr_stats(data_list):
    constr_stats = {
        "processed": 0,
        "LLM": 0
    }
    constr_len = []
    constr_operator = {}
    constr_f1 = []
    constr_prec = []
    constr_recall = []
    for data in data_list:
        # if any([rule[0]["cot"] in ["processed", "divided"] for rule in data["rules"]]):
        #     constr_stats["processed"] += 1
        # else :
        #     # for r in data["rules"] :
        #     #     print(r[0]["cot"])
        #     constr_stats["LLM"] += 1
        for rule in data["rules"]:
            if "processed" == rule[0]["cot"] or "divided" == rule[0]["cot"]:
                constr_stats["processed"] += 1
            else :
                constr_stats["LLM"] += 1
            constr_len.append(rule[0].get("length", 1))
            constr_f1.append(rule[1]["f1_score"])
            constr_prec.append(rule[1]["precision"])
            constr_recall.append(rule[1]["recall"])
    return constr_stats, constr_len, constr_operator, constr_f1, constr_prec, constr_recall
def mean(numbers):
    return sum(numbers) / len(numbers)
import statistics
# -> visualize distribution of length of cosntr, f1_ prec_ recall of constr, pie chart of constr type
def viz_passrate(data_list, acc_data_list, name, path = "/artifact/results/") :
    all = []
    for i, data_li in enumerate([data_list, acc_data_list]):
        pass_rate_num_of_constr = []
        for data in data_li:
            pass_rate_num_of_constr.append(
                (data["pass_rate"], len(data["rules"]))
            )
        pass_rate, num_of_constraints = zip(*pass_rate_num_of_constr)
        all.extend(pass_rate)
        print("mean passrate", mean(pass_rate))
        print("median passrate", statistics.median(pass_rate))

    print("overall passrate")
    print("mean passrate", mean(all))
    print("median passrate", statistics.median(all))

def viz_gen_way_of_constrs(constr_stats):
    print("### GEN WAY OF CONSTRAINS ###")
    print(constr_stats)
    print("### GEN WAY OF CONSTRAINS ###")

def viz_constr_len(constr_len : List[int]):
    plt.hist(constr_len, bins=range(1, max(constr_len) + 1))
    plt.show()

def viz_constr_f1(constr_recall : List[float], constr_prec, acc_recall, acc_prec , path = "/artifact/results/", name = "PyTorch"):

    plt.figure(figsize=(9,9))
    plt.legend(loc='lower right')
    # PyTorch
    plt.scatter(acc_prec, acc_recall, alpha=0.8, edgecolors='black', label=r'\textsc{DeepConstr$^{s}$}', linewidth=0.6, s=80, c='blue', marker='x')
    plt.scatter(constr_prec, constr_recall, alpha=0.8, edgecolors='black', label=r'\textsc{DeepConstr}', linewidth=0.6, s=80, c='red', marker='+')
    # plt.scatter(acc_prec, acc_recall, alpha=0.8, edgecolors='black', label=r'\textsc{DeepConstr$^{s}$}', linewidth=0.6, s=80, c='blue', marker='o')
    # plt.scatter(constr_prec, constr_recall, alpha=0.8, edgecolors='black', label=r'\textsc{DeepConstr}', linewidth=0.6, s=80, c='red', marker='o')
    print("mean acc_prec", mean(acc_prec))
    print("median acc_prec", statistics.median(acc_prec))
    print("mean acc_recall", mean(acc_recall))
    print("median acc_recall", statistics.median(acc_recall))
    print("mean constr_prec", mean(constr_prec))
    print("median constr_prec", statistics.median(constr_prec))
    print("mean constr_recall", mean(constr_recall))
    print("median constr_recall", statistics.median(constr_recall))
    # if name == "torch" :
    #     plt.title('\\textit{PyTorch}')
    # else :
    #     plt.title('\\textit{TensorFlow}')

    plt.xlabel('\\textit{Soundness}', fontsize=20)
    plt.ylabel('\\textit{Completeness}', fontsize=20)
    plt.legend()
    plt.xticks(range(0, 101, 10))
    plt.yticks(range(0, 101, 10))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(path + f"5_dist_{name}.pdf")
    plt.savefig(path + f"5_dist_{name}.png")

if __name__ == "__main__" : 
    record_dir = "/artifact/data/"
    frameworks = ["torch", "tf"]
    kinds = ["records", "only_acc"]

    for framework in frameworks:
        data = []
        path = os.path.join(record_dir, "records", framework)
        acc_path = os.path.join(record_dir, "only_acc", framework)
        data_list = load_data(path)
        acc_data_list = load_data(acc_path)
        constr_stats, constr_len, constr_operator, constr_f1, constr_prec, constr_recall = get_constr_stats(data_list)
        acc_constr_stats, acc_constr_len, acc_constr_operator, acc_constr_f1, acc_constr_prec, acc_constr_recall = get_constr_stats(acc_data_list)

    for framework in frameworks:
        data = []
        path = os.path.join(record_dir, "records", framework)
        acc_path = os.path.join(record_dir, "only_acc", framework)
        data_list = load_data(path)
        acc_data_list = load_data(acc_path)
        constr_stats, constr_len, constr_operator, constr_f1, constr_prec, constr_recall = get_constr_stats(data_list)
        acc_constr_stats, acc_constr_len, acc_constr_operator, acc_constr_f1, acc_constr_prec, acc_constr_recall = get_constr_stats(acc_data_list)
        print("all")
        viz_gen_way_of_constrs(constr_stats)
        print("only_acc")
        viz_gen_way_of_constrs(acc_constr_stats)
        viz_constr_f1(constr_recall, constr_prec, acc_constr_recall, acc_constr_prec, name=framework)
        viz_passrate(data_list, acc_data_list, name=framework)
        # print(pass_rate_num_of_constr)
