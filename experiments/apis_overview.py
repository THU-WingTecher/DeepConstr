
import json
import os

import yaml


def get_trained_list(record_path, path) :
    data = set()
    for root, dirs, files in os.walk(record_path):
        for file in files:
            if file.endswith(".yaml"):
                with open(os.path.join(root, file), "r") as f:
                    record = yaml.safe_load(f)
                if record["pass_rate"] > 0 :
                    data.add(record["name"])
    return list(data)

def get_prepared_list() : 
    data = set()
    prepared_path = "/artifact/data/records"
    for root, dirs, files in os.walk(prepared_path+"/torch"):
        for file in files:
            if file.endswith(".yaml"):
                api_name = os.path.join(root, file).replace(prepared_path+"/","").replace("/",".").split('-')[0]
                data.add(api_name)
    return list(data)

def save_to_json(data, path) : 
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__" : 
    record_path = "/artifact/data/records"
    tf_trained = get_trained_list(record_path+"/tf", "tf")
    print("Number of trained tf apis: ", len(tf_trained))
    print("Saving trained tf apis to json file")
    save_to_json(tf_trained, "/artifact/data/tf_trained.json")
    tf_prepared = get_prepared_list()
    tf_untrained = set(tf_prepared) - set(tf_trained)
    print(tf_untrained)
    print(len(tf_untrained))
    save_to_json(list(tf_untrained), "/artifact/data/tf_untrained.json")
    print("done")