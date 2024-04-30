        completed_list = get_completed_list()
        if self.cfg["model"]["type"] == "torch" :
            train_list = check_left_api(
                    pt_neuri_data_path,
                    pt_data_paths
                ) + check_left_api(
                    pt_deepconstr_data_path,
                    pt_data_paths
                )
        elif self.cfg["model"]["type"] == "tensorflow" :
            train_list = check_left_api(
                    tf_neuri_data_path,
                    tf_data_paths
                ) + check_left_api(
                    tf_deepconstr_data_path,
                    tf_data_paths
                )
            
def get_completed_list(path = "/artifact/experiments/results/completed.json") : 
    data = set()
    # with open(path, "r") as f:
    #     data.extend(json.load(f))
    
    #load csv 
    csv_paths = [
        "/artifact/experiments/results/merged_torch_v2.csv",
        "/artifact/experiments/results/merged_tf_v2.csv"
        ]
    for csv_path in csv_paths :
        with open(csv_path, "r") as f:
            for i, line in enumerate(f.readlines()) :
                if i==0 : continue # head 
                data.add(line.split(",")[0].replace(".models",""))
    return list(set(data))

def check_left_api(api_data_path, saved_data_paths) :
    with open(api_data_path, "r") as file:
        data = json.load(file)
    data = list(set(data))
    for saved_data_path in saved_data_paths :
        with open(saved_data_path, "r") as f:
            for i, line in enumerate(f.readlines()) :
                if i==0 :
                    pass
                else :
                    columns = line.split(",")
                    api_name = columns[0].replace(".models","")
                    if api_name in data :
                        data.remove(api_name)
    
    return data

def check_record(api_names, record_dir) :
    todo = []
    for name in api_names :
        path = os.path.join(record_dir, name.replace(".","/")+"-0.yaml")
        if not os.path.exists(path) :
            todo.append(name)
            print(f"{path} is not exist")
    return todo
            
pt_data_paths = [
    "/artifact/experiments/results/merged_torch_v2.csv",
]
pt_neuri_data_path = "/artifact/data/torch_overall_apis.json"
pt_deepconstr_data_path = "/artifact/data/torch_deepconstr.json"
tf_data_paths = [
    "/artifact/experiments/results/merged_tf_v2.csv"
]
tf_neuri_data_path = "/artifact/data/tf_overall_apis.json"
tf_deepconstr_data_path = "/artifact/data/tf_deepconstr.json"

# print("neuri - pt")
# neuri_pt = check_left_api(
#     pt_neuri_data_path,
#     pt_data_paths
# )
# check_record(neuri_pt, "/artifact/data/records/torch")
# print("neuri - pt")
# neuri_pt = check_left_api(
#     pt_neuri_data_path,
#     pt_data_paths
# )
# check_record(neuri_pt, "/artifact/data/records/torch")

# print(check_left_api(
#     pt_neuri_data_path,
#     pt_data_paths
# ))
# print("neuri - pt")
# neuri_pt = check_left_api(
#     pt_neuri_data_path,
#     pt_data_paths
# )

# print(check_record(neuri_pt, "/artifact/data/records"))
# print("neuri - tf")
# neuri_tf = check_left_api(
#     tf_neuri_data_path,
#     tf_data_paths
# )
# print(check_record(neuri_tf, "/artifact/data/records"))

# print("deepconstr - pt")
# deepconstr_pt = check_left_api(
#     pt_deepconstr_data_path,
#     pt_data_paths
# )

# print(check_record(deepconstr_pt, "/artifact/data/records"))
# print("deepconstr - tf")
# deepconstr_tf = check_left_api(
#     tf_deepconstr_data_path,
#     tf_data_paths
# )
# print(check_record(deepconstr_tf, "/artifact/data/records"))