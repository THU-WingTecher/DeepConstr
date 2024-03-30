
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
            if "synthesized" == rule[0]["cot"] or "divided" == rule[0]["cot"]:
                constr_stats["synthesized"] += 1
            else :
                constr_stats["other"] += 1
            constr_len.append(rule[0].get("length", 1))
            constr_f1.append(rule[1]["f1_score"])
            constr_prec.append(rule[1]["precision"])
            constr_recall.append(rule[1]["recall"])
    return constr_stats, constr_len, constr_operator, constr_f1, constr_prec, constr_recall

# -> visualize distribution of length of cosntr, f1_ prec_ recall of constr, pie chart of constr type
def viz_num_of_constr_with_pass_rate(data_list, acc_data_list, name, path = "/artifact/results/") :
    plt.figure(figsize=(9,12))
    for i, data_li in enumerate([data_list, acc_data_list]):
        pass_rate_num_of_constr = []
        for data in data_li:
            pass_rate_num_of_constr.append(
                (data["pass_rate"], len(data["rules"]))
            )
        if i == 0 :
            edge_color = 'black'
            line_color = 'blue'
            marker = 'x'
            color = 'blue'
            label = r'\textsc{DeepConstr}'
        else :
            edge_color = 'black'
            marker = 'x'
            color = 'red'
            line_color = 'red'
            label = r'\textsc{DeepConstr$^{s}$}'
        pass_rate, num_of_constraints = zip(*pass_rate_num_of_constr)

        # Convert to numpy arrays for easier manipulation
        pass_rate = np.array(pass_rate)
        
        num_of_constraints = np.array(num_of_constraints)
        z2 = np.polyfit(num_of_constraints, pass_rate, 2)
        p2 = np.poly1d(z2)

        # Generating points for the trend line
        # trend_line_x = np.linspace(min(num_of_constraints), max(num_of_constraints), 100)
        # trend_line_y = p2(trend_line_x)
        # Re-plotting the scatter and the improved trend line
        plt.scatter(num_of_constraints, pass_rate, alpha=0.8, edgecolors=edge_color, linewidth=0.6, s=80, c=color, marker=marker)
        # plt.plot(trend_line_x, trend_line_y, "r-", label=label, color=line_color)

    # Adjusting y-axis to not display negative values
    plt.ylim(bottom=0)

    # Adding professional plot enhancements with no negative y-axis values
    # if name == "torch" :
    #     plt.title('\\textit{PyTorch}')
    # else :
    #     plt.title('\\textit{TensorFlow}')
    plt.xlabel('\\textit{Number of Constraints}')
    plt.ylabel('\\textit{Pass Rate(\%)}')
    # plt.xticks()
    # plt.yticks()
    plt.legend()

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()

    # Show the adjusted plot
    plt.savefig(path + f"num_constr_passrate-{name}.pdf")
    plt.savefig(path + f"num_constr_passrate-{name}.png")

def viz_gen_way_of_constrs(constr_stats):
    print(constr_stats)
    for key in constr_stats.keys():
        print("key", constr_stats[key] / sum(constr_stats.values()))

def viz_constr_len(constr_len : List[int]):
    plt.hist(constr_len, bins=range(1, max(constr_len) + 1))
    plt.show()

def viz_constr_f1(constr_f1 : List[float], constr_prec, acc_f1, acc_prec , path = "/artifact/results/", name = "PyTorch"):
    plt.figure(figsize=(9,12))
    # PyTorch
    plt.scatter(acc_f1, acc_prec, c=colors[1], label=r'\textsc{DeepConstr$^{s}$}', alpha=0.6, edgecolors='w', linewidths=0.5)
    plt.scatter(constr_f1, constr_prec, c=colors[0], label=r'\textsc{DeepConstr}', alpha=0.6, edgecolors='w', linewidths=0.5)
    # if name == "torch" :
    #     plt.title('\\textit{PyTorch}')
    # else :
    #     plt.title('\\textit{TensorFlow}')

    plt.xlabel('\\textit{Soundness}')
    plt.ylabel('\\textit{Completeness}')
    plt.legend()
    plt.xticks(range(0, 101, 10))
    plt.yticks(range(0, 101, 10))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(path + f"dist-time-{name}.pdf")
    plt.savefig(path + f"dist-time-{name}.png")

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
        print("all")
        viz_gen_way_of_constrs(constr_stats)
        print("only_acc")
        viz_gen_way_of_constrs(acc_constr_stats)
        viz_constr_f1(constr_recall, constr_prec, acc_constr_recall, acc_constr_prec, name=framework)
        viz_num_of_constr_with_pass_rate(data_list, acc_data_list, name=framework)
        # print(pass_rate_num_of_constr)