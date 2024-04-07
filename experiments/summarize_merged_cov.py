import os
import pickle
import pandas as pd
import json
import datetime

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
                dir_path = os.path.dirname(os.path.dirname(pickle_path)) #../*.models/coverage/merged_cov.pkl
                model_n = len(os.listdir(dir_path)) - 1
                try :
                    with open(pickle_path, 'rb') as file:
                        data = pickle.load(file)
                        # Summarize the data
                        branch_by_time, final_bf = cov_summerize(data)
                        api_summary[column] = {
                            'branch_by_time': branch_by_time,
                            'final_bf': max([bf for _,_,bf in branch_by_time]),
                            'model_n': model_n
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
    final_bf_summary = aggregated_df.pivot_table(index='API', columns='Column', values='Final BF', aggfunc='max')
    model_cnt = aggregated_df.pivot_table(index='API', columns='Column', values='Model Count', aggfunc='max')
    model_cnt.add_suffix('_cnt')
    for idx in range(final_bf_summary.shape[0]):
        max_column = final_bf_summary.iloc[idx].idxmax()
        if max_column == 'constrinf' :
            completed_data.append(final_bf_summary.iloc[idx]._name.replace(folder_suffix, ''))
            cnt+=1

    all_data = final_bf_summary.shape[0]
    revise_complete_data("/artifact/experiments/results/completed.json", completed_data)
    print("Total APIs with constrinf as the largest final BF: ", cnt, "from", all_data)
    print(f"Increase ratio of constrinf as the largest final BF: {cnt/all_data}")
    return pd.concat([final_bf_summary, model_cnt], axis=1), completed_data 

def save_data(final_bf, completed_data, save_dir="/artifact/experiments/results"):
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

    if 'constrinf' in merged_df.columns:
        filtered_df = merged_df.sort_values('constrinf', ascending=False).drop_duplicates(subset='API').sort_index()
    else:
        # If 'constrinf' column does not exist, just remove duplicates based on all columns
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
            

def gen_table4(df, nnsmith_path, neuri_path, type="torch") :

    with open(nnsmith_path, 'r') as file:
        apis = json.load(file)
    nnsmith_columns = [name + folder_suffix for name in apis]
    with open(neuri_path, 'r') as file:
        apis = json.load(file)
    neuri_columns = [name + folder_suffix for name in apis]
    nnsmith_none, neuri_none = [], []
    improvements = []
    print(len(nnsmith_columns), len(list(set(nnsmith_columns + neuri_columns))))
    for col in nnsmith_columns:
        if col not in df.API.values:
            nnsmith_none.append(col.replace(folder_suffix, ''))
            df.loc[len(df)] = [col, 0, 0, 0, 0, 0, 0, 0, 0]  # Or another default value as appropriate
    # for col in neuri_columns:
    #     if col not in df.API.values:
    #         neuri_none.append(col.replace(folder_suffix, ''))
    #         df.loc[len(df)] = [col, 0, 0, 0, 0, 0, 0, 0, 0]  # Or another default value as appropriate
    for tool in ["constrinf"] :
        for baseline in ["neuri", "symbolic", "constrinf_2"] :
            columns_to_compare = [baseline] if all(col in df.columns for col in [baseline]) else []
            if baseline == "symbolic":
                total_rows = len(nnsmith_columns)
                added = len(nnsmith_none)
                extracted_columns_df_with_models = df[df['API'].isin(nnsmith_columns)]
            else :
                total_rows = df.shape[0]
                added = len(neuri_none)
                extracted_columns_df_with_models = df[df['API'].isin(neuri_columns)]
            rows_where_constrinf_is_highest = extracted_columns_df_with_models.apply(lambda row: row[tool] > row[columns_to_compare], axis=1).sum()
            print("rows_where_constrinf_is_highest", tool, "vs", baseline, ":", "highest", rows_where_constrinf_is_highest, "intersected", total_rows - added)
    for col in df.columns:
        if col == "API":
            continue
        print(col)
        print(df[df[col] != 0][col].mean())

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
    # parser.add_argument(
    #     "-o", "--output", type=str, default="results", help="results folder"
    # )
    # parser.add_argument("-t", "--tlimit", type=int, default=4 * 3600, help="time limit")
    # parser.add_argument("--pdf", action="store_true", help="use pdf as well")
    # parser.add_argument("--minfac", type=float, default=0.85, help="min factor")
    # parser.add_argument("--maxfac", type=float, default=1.02, help="max factor")
    args = parser.parse_args()
    data = {}
    for folder in args.folders:
        traverse_and_classify(folder, data)

    processed_data = process_pickle_files(data)
    aggregated_df = aggregate_summarized_data(processed_data)
    final_bf_summary, completed_data = summarize_final_bf(aggregated_df)
    # print(final_bf_summary)
    save_data(final_bf_summary, completed_data, "/artifact/experiments/results")

    # merge_with_original_data(orig_df, aggregated_df)
    # csv_paths = [
    #     "/artifact/experiments/results/merged_torch_v3.csv",
    # ]
    # # orig_df = merge_csvs(*csv_paths)
    # print("torch")
    # csv_paths = [
    #     "/artifact/experiments/results/final_torch.csv",
    # ]
    # df = merge_csvs(*csv_paths)
    # gen_table4(
    #     df,  
    #     "/artifact/data/torch_nnsmith.json",
    #     "/artifact/data/torch_neuri.json",
    #     type="torch")
    # print("tf")
    # csv_paths = [
    #     "/artifact/experiments/results/final_tf.csv",
    # ]
    # df = merge_csvs(*csv_paths, save_path="/artifact/experiments/results/merged_tf_v3.csv")
    # gen_table4(
    #     df,
    #     "/artifact/data/tf_nnsmith.json",
    #     "/artifact/data/tf_neuri.json",
    #     type="tf")

    # print("torch")
    # df = merge_csvs(*csv_paths)
    # gen_table4(
    #     df,  
    #     nnsmith_path="/artifact/data/torch_nnsmith.json",
    #     neuri_path="/artifact/data/torch_overall_apis.json",
    #     type="torch")
    # print("tf")
    # csv_paths = [
    #     "/artifact/experiments/results/merged_tf_v2.csv",
    #     "/artifact/experiments/results/20240402-235616.csv",
    #     "/artifact/experiments/results/20240404-124049.csv"
    # ]
    # df = merge_csvs(*csv_paths, save_path="/artifact/experiments/results/merged_tf_v3.csv")
    # gen_table4(
    #     df,
    #     "/artifact/data/tf_nnsmith.json",
    #     "/artifact/data/tf_overall_apis.json",
    #     type="torch")

# pt_data_paths = [
#     "/artifact/experiments/results/merged_torch_v2.csv",
# ]
# pt_neuri_data_path = "/artifact/data/torch_overall_apis.json"
# pt_nnsmith_data_path = "/artifact/data/torch_nnsmith.json"
# tf_data_paths = [
#     "/artifact/experiments/results/merged_tf_v2.csv"
# ]
# tf_neuri_data_path = "/artifact/data/tf_overall_apis.json"
# tf_nnsmith_data_path = "/artifact/data/tf_nnsmith.json"

# def exclude_intestable() :
#     neuri_pt = check_left_api(
#         pt_neuri_data_path,
#         pt_data_paths
#     )

#     neuri_pt_rec = check_record(neuri_pt, "/artifact/data/records")
#     neuri_tf = check_left_api(
#         tf_neuri_data_path,
#         tf_data_paths
#     )
#     neuri_tf_rec = check_record(neuri_tf, "/artifact/data/records")

#     nnsmith_pt = check_left_api(
#         pt_nnsmith_data_path,
#         pt_data_paths
#     )

#     nnsmith_pt_rec = check_record(nnsmith_pt, "/artifact/data/records")
#     nnsmith_tf = check_left_api(
#         tf_nnsmith_data_path,
#         tf_data_paths
#     )
#     nnsmith_tf_rec = check_record(nnsmith_tf, "/artifact/data/records")
#     return list(set(neuri_pt_rec + neuri_tf_rec + nnsmith_pt_rec + nnsmith_tf_rec))