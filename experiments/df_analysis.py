import os
import pickle
import pandas as pd
import json
import datetime
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
folder_suffix = ".models"
def parse_directory_name(dirname):
    """
    Parses the directory name to classify information into columns and API names.

    Args:
    - dirname (str): The directory name to parse.

    Returns:
    - tuple: A tuple containing the column name and API name.
    """
    parts = dirname.split('-')
    if len(parts) >= 3:
        column = parts[1]  # Column name (e.g., 'neuri')
        api_name = parts[-1].replace(folder_suffix, "")  # API name (e.g., 'torch.acos')
        return column, api_name
    else:
        return None, None

def cov_summerize(data, tlimit=None, gen_time=None):
    """
    Summarizes coverage data, calculating branch coverage over time and the final branch full.

    Args:
    - data (dict): The coverage data dictionary.
    - tlimit (float, optional): Optional time limit for summarization. Defaults to None.
    - gen_time (np.array, optional): Optional generation time array for adjusting time values. Defaults to None.

    Returns:
    - tuple: A tuple containing branch_by_time and final_bf.
    """
    model_total = 0
    branch_by_time = [[0, 0, 0]]
    final_bf = 0

    for time, value in data.items():
        bf = 0
        n_model = value["n_model"]
        cov = value["merged_cov"]
        model_total += n_model

        if gen_time is not None:
            time -= gen_time[0][:model_total].sum()

        branch_cov = 0
        for fname in cov:
            branch_cov += len(cov[fname]["branches"])
            bf += cov[fname]["bf"]

        branch_by_time.append([time, model_total, branch_cov])
        final_bf = max(final_bf, bf)

        if tlimit is not None and time > tlimit:
            break

    return branch_by_time, final_bf

def traverse_and_classify(root_dir, api_data):
    """
    Traverses the root directory to identify relevant subdirectories and classifies the information.

    Args:
    - root_dir (str): The root directory to traverse.

    Returns:
    - dict: A dictionary with keys as API names and values as dicts with columns and paths.
    """
    for root, dirs, files in os.walk(root_dir):
        for dirname in dirs:
            if dirname.endswith(folder_suffix):
                column, api_name = parse_directory_name(dirname)
                if column and api_name:
                    if api_name not in api_data:
                        api_data[api_name] = {}
                    assert api_data[api_name].get(column) is None, f"the data of {api_name}-{column} is duplicated"
                    api_data[api_name][column] = os.path.join(root, dirname, 'coverage', 'merged_cov.pkl')

def get_model_cnt(pickle_path) : 
    cnt = 0
    dir_path = os.path.dirname(os.path.dirname(pickle_path)) #../*.models/coverage/merged_cov.pkl
    dir_path = dir_path.replace(folder_suffix, "")
    csv_path = os.path.join(dir_path, "status.csv")
    with open(csv_path, 'r') as file:
        data = file.readlines() 
    for row in data :
        cols = row.split(",")
        if cols and cols[1] == "ok" :
            cnt +=1
    return cnt

def process_pickle_files(api_coverage_data):
    """
    Processes each relevant pickle file using the cov_summerize function.

    Args:
    - api_coverage_data (dict): Dictionary with API names as keys and dicts with columns and pickle file paths as values.

    Returns:
    - dict: A dictionary with API names as keys and summarized coverage data as values.
    """
    summarized_data = {}
    for api_name, columns in api_coverage_data.items():
            api_summary = {}
            for column, pickle_path in columns.items():
                # Load the pickle file

                try :
                    with open(pickle_path, 'rb') as file:
                        data = pickle.load(file)
                        # Summarize the data
                        branch_by_time, final_bf = cov_summerize(data)
                        api_summary[column] = {
                            'branch_by_time': branch_by_time,
                            'final_bf': max([bf for _,_,bf in branch_by_time]),
                            'model_n': get_model_cnt(pickle_path)
                        }
                except FileNotFoundError :
                    api_summary[column] = {
                        'branch_by_time': [[0, 0, 0], [0, 0, 0]],
                        'final_bf': 0,
                        'model_n': 0,
                    }
                    print("File not found : ", pickle_path)              
            summarized_data[api_name] = api_summary

    return summarized_data

def aggregate_summarized_data(processed_data):
    """
    Aggregates the summarized data across different APIs and columns.

    Args:
    - processed_data (dict): The processed data from pickle files.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the aggregated data.
    """
    # Create a list to hold all data rows
    aggregated_data = []

    # Iterate over the processed data and aggregate
    for api_name, api_summary in processed_data.items():
        for column, summary in api_summary.items():
            for time_data in summary['branch_by_time']:
                time, model_count, branch_cov = time_data
                aggregated_data.append({
                    'API': api_name,
                    'Column': column,
                    'Time': time,
                    'Model Count': summary['model_n'],
                    'Branch Coverage': branch_cov,
                    'Final BF': summary['final_bf']
                })

    # Convert the list of dicts to a DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)
    return aggregated_df

def revise_complete_data(save_dir, api_names):
    with open(save_dir, 'r') as file:
        data = json.load(file)
    
    for name in api_names :
        if name in data : 
            pass 
        else :
            data.append(name) 
    
    data = list(set(data))
    data.sort()
    with open(save_dir, 'w') as file:
        json.dump(data, file)

def summarize_final_bf(aggregated_df):
    """
    Summarizes the final_bf values and formats the table to highlight the largest value in each row.

    Args:
    - aggregated_df (pd.DataFrame): The aggregated data in a pandas DataFrame.

    Returns:
    - str: A string representation of the summarized table.
    """
    completed_data = []
    cnt = 0
    # print(aggregated_df.head())
    final_bf_summary = aggregated_df.pivot_table(index='API', columns='Column', values='Final BF', aggfunc='max')
    model_cnt = aggregated_df.pivot_table(index='API', columns='Column', values='Model Count', aggfunc='max')
    model_cnt = model_cnt.add_suffix('_cnt')
    for idx in range(final_bf_summary.shape[0]):
        max_column = final_bf_summary.iloc[idx].idxmax()
        if max_column == 'deepconstr' :
            completed_data.append(final_bf_summary.iloc[idx]._name.replace(folder_suffix, ''))
            cnt+=1

    all_data = final_bf_summary.shape[0]
    revise_complete_data("/DeepConstr/experiments/results/completed.json", completed_data)
    print("Total APIs with deepconstr as the largest final BF: ", cnt, "from", all_data)
    print(f"Increase ratio of deepconstr as the largest final BF: {cnt/all_data}")
    return pd.concat([final_bf_summary, model_cnt], axis=1), completed_data 

def save_data(final_bf, completed_data, save_dir="/DeepConstr/experiments/results"):
    save_path = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
    complete_data_path = os.path.join(save_dir, "completed.json")
    final_bf.to_csv(save_path)
    revise_complete_data(complete_data_path, completed_data)

def merge_csvs(*csv_paths, save_path=None):
    dfs = []
    for arg in csv_paths : 
        dfs.append(pd.read_csv(arg))
    
    merged_df = pd.concat(dfs)
    # filtered_df = merged_df[~(merged_df == 0).any(axis=1)]

    if 'deepconstr' in merged_df.columns:
        filtered_df = merged_df.sort_values('deepconstr', ascending=False).drop_duplicates(subset='API').sort_index()
    else:
        # If 'deepconstr' column does not exist, just remove duplicates based on all columns
        filtered_df = merged_df.drop_duplicates()

    sorted_columns_df = filtered_df.sort_values(by="API")
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df = filtered_df.sort_values(by="API")
    if save_path is not None :
        filtered_df.to_csv(save_path, index=False)
    return filtered_df

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
                    api_name = columns[0].replace(folder_suffix,"")
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

def extract_unnormal_val(df) :
    mask = df < 0

    # Step 2: Find the row and column labels where the condition is True
    rows, cols = np.where(mask)

    # Step 3: Iterate through the rows and cols to save the (row, col) tuples where the condition is met
    # Note: This uses the actual index and column labels, not integer positions
    positions = [(df.index[row], df.columns[col]) for row, col in zip(rows, cols)]            
    with open("/DeepConstr/results/unnormal_val.json", "w") as file:
        json.dump(positions, file)

def gen_table4_from_df(*args, nnsmith_path, neuri_path, type="torch") :
    dfs = []
    for path in args :
        if isinstance(path, str) :
            dfs.append(pd.read_csv(path, index_col="API"))
        elif isinstance(path, pd.DataFrame) :
            dfs.append(path)
    df = pd.concat(dfs, axis=1)
    with open(nnsmith_path, 'r') as file:
        apis = json.load(file)
    nnsmith_columns = [name for name in apis]
    with open(neuri_path, 'r') as file:
        apis = json.load(file)
    neuri_columns = [name for name in apis]
    nnsmith_none, neuri_none = [], []
    improvements = []
    print(len(nnsmith_columns), len(list(set(nnsmith_columns + neuri_columns))))
    # for col in nnsmith_columns:
    #     if col not in df.API.values:
    #         nnsmith_none.append(col.replace(folder_suffix, ''))
    #         df.loc[len(df)] = [col, 0, 0, 0, 0, 0, 0, 0, 0]  # Or another default value as appropriate
    # for col in neuri_columns:
    #     if col not in df.API.values:
    #         neuri_none.append(col.replace(folder_suffix, ''))
    #         df.loc[len(df)] = [col, 0, 0, 0, 0, 0, 0, 0, 0]  # Or another default value as appropriate
    # print(df.head())
    columns_to_subtract = ['deepconstr', 'neuri', 'symbolic', 'deepconstr_2']
    for col in columns_to_subtract:
        if col in df.columns:
            default_val = 8937 if type == "tensorflow" else 35809# 31304 # 8937
            df[col] = df[col] - default_val

    extract_unnormal_val(df)
    for tool in ["deepconstr"] :
        for baseline in ["neuri", "symbolic", "deepconstr_2"] :
            columns_to_compare = [baseline] if all(col in df.columns for col in [baseline]) else []
            if baseline == "symbolic":
                extracted_columns_df_with_models = df.loc[df.index.intersection(nnsmith_columns)]
                total_rows = len(nnsmith_columns)
                added = total_rows - extracted_columns_df_with_models.shape[0]
            else:
                extracted_columns_df_with_models = df.loc[df.index.intersection(neuri_columns)]
                total_rows = len(neuri_columns)
                added = total_rows - extracted_columns_df_with_models.shape[0]
            print(extracted_columns_df_with_models.head())
            for col in columns_to_compare:
                improvement_col_name = f"improvement_ratio_{tool}_vs_{col}"

                # Calculate the improvement ratio and ensure it's a single value per row (Series)
                extracted_columns_df_with_models[improvement_col_name] = extracted_columns_df_with_models.apply(
                    lambda row: (row[tool] - row[col]) / row[col] if row[tool] > 0 and row[col] > 0 else 0, axis=1)
                
                # Correctly update the original DataFrame
                df.loc[extracted_columns_df_with_models.index, improvement_col_name] = extracted_columns_df_with_models[improvement_col_name]
            
            rows_where_deepconstr_is_highest = extracted_columns_df_with_models.apply(
                lambda row: row[tool] > row[columns_to_compare], axis=1).sum()
            
            print(f"rows_where_deepconstr_is_highest {tool} vs {baseline}: highest {rows_where_deepconstr_is_highest}, intersected {total_rows - added}")
    df = df.sort_values(by='improvement_ratio_deepconstr_vs_neuri', ascending=True)
    for col in df.columns:
        if col == "API":
            continue
        print(col)
        print(df[df[col] != 0][col].mean())
    sorted_df = df.reset_index()
    return sorted_df

def merge_with_original_data(df_original, aggregated_df) : 
    final_bf_summary = aggregated_df.pivot_table(index='API', columns='Column', values='Final BF', aggfunc='max')
    model_cnt = aggregated_df.pivot_table(index='API', columns='Column', values='Model Count', aggfunc='max')
    new_data = pd.concat([final_bf_summary, model_cnt], axis=1)
    print(new_data)
    # df_original_copy = df_original.copy()
    # df_original.update(new_data)
    # # Identifying and printing the changed values
    # changed_values = df_original != df_original_copy
    # for col in changed_values.columns:
    #     for row in changed_values.index:
    #         if changed_values.at[row, col]:
    #             print(f"Changed value at row {df_original.loc[row,'API']}, column '{col}': {df_original_copy.at[row, col]} -> {df_original.at[row, col]}")

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folders", type=str, nargs="+", help="bug report folder"
    )
    # parser.add_argument("--tags", type=str, nargs="+", help="tags")
    # parser.add_argument(
    #     "-n", "--name", type=str, help="file_name"
    # )
    parser.add_argument(
        "-o", "--output", type=str, help="save name"
    )
    parser.add_argument(
        "-p", "--package", type=str, help="save name"
    )
    # parser.add_argument("-t", "--tlimit", type=int, default=4 * 3600, help="time limit")
    # parser.add_argument("--pdf", action="store_true", help="use pdf as well")
    # parser.add_argument("--minfac", type=float, default=0.85, help="min factor")
    # parser.add_argument("--maxfac", type=float, default=1.02, help="max factor")
    args = parser.parse_args()

    assert args.package == "torch" or args.package == "tensorflow"
    dir = "/DeepConstr/results/"
    if args.package == "torch" : 
        nnsmith_path = "/DeepConstr/data/torch_nnsmith.json"
        neuri_path = "/DeepConstr/data/torch_neuri.json"
    elif args.package == "tensorflow" :
        nnsmith_path = "/DeepConstr/data/tf_nnsmith.json"
        neuri_path = "/DeepConstr/data/tf_neuri.json"
    result_df = gen_table4_from_df(
        *[os.path.join(dir, folder) for folder in args.folders],  
        nnsmith_path=nnsmith_path,
        neuri_path=neuri_path,
        type=args.package)
    result_df.to_csv(os.path.join(dir, args.output+".csv"), index=False)
    
    # result_df = gen_table4(
    #     final_bf_summary,  
    #     "/DeepConstr/data/torch_nnsmith.json",
    #     "/DeepConstr/data/torch_neuri.json",
    #     type="torch")
    # result_df.to_csv(os.path.join("/DeepConstr/results/", args.output+".csv"), index=False)
    # print("tf")
    # csv_paths = [
    #     "/DeepConstr/experiments/results/final_tf.csv",
    # ]
    # df = merge_csvs(*csv_paths, save_path="/DeepConstr/experiments/results/merged_tf_v3.csv")
    # gen_table4(
    #     df,
    #     "/DeepConstr/data/tf_nnsmith.json",
    #     "/DeepConstr/data/tf_neuri.json",
    #     type="tf")

    # print("torch")
    # df = merge_csvs(*csv_paths)
    # gen_table4(
    #     df,  
    #     nnsmith_path="/DeepConstr/data/torch_nnsmith.json",
    #     neuri_path="/DeepConstr/data/torch_overall_apis.json",
    #     type="torch")
    # print("tf")
    # csv_paths = [
    #     "/DeepConstr/experiments/results/merged_tf_v2.csv",
    #     "/DeepConstr/experiments/results/20240402-235616.csv",
    #     "/DeepConstr/experiments/results/20240404-124049.csv"
    # ]
    # df = merge_csvs(*csv_paths, save_path="/DeepConstr/experiments/results/merged_tf_v3.csv")
    # gen_table4(
    #     df,
    #     "/DeepConstr/data/tf_nnsmith.json",
    #     "/DeepConstr/data/tf_overall_apis.json",
    #     type="torch")

# pt_data_paths = [
#     "/DeepConstr/experiments/results/merged_torch_v2.csv",
# ]
# pt_neuri_data_path = "/DeepConstr/data/torch_overall_apis.json"
# pt_nnsmith_data_path = "/DeepConstr/data/torch_nnsmith.json"
# tf_data_paths = [
#     "/DeepConstr/experiments/results/merged_tf_v2.csv"
# ]
# tf_neuri_data_path = "/DeepConstr/data/tf_overall_apis.json"
# tf_nnsmith_data_path = "/DeepConstr/data/tf_nnsmith.json"

# def exclude_intestable() :
#     neuri_pt = check_left_api(
#         pt_neuri_data_path,
#         pt_data_paths
#     )

#     neuri_pt_rec = check_record(neuri_pt, "/DeepConstr/data/records")
#     neuri_tf = check_left_api(
#         tf_neuri_data_path,
#         tf_data_paths
#     )
#     neuri_tf_rec = check_record(neuri_tf, "/DeepConstr/data/records")

#     nnsmith_pt = check_left_api(
#         pt_nnsmith_data_path,
#         pt_data_paths
#     )

#     nnsmith_pt_rec = check_record(nnsmith_pt, "/DeepConstr/data/records")
#     nnsmith_tf = check_left_api(
#         tf_nnsmith_data_path,
#         tf_data_paths
#     )
#     nnsmith_tf_rec = check_record(nnsmith_tf, "/DeepConstr/data/records")
#     return list(set(neuri_pt_rec + neuri_tf_rec + nnsmith_pt_rec + nnsmith_tf_rec))