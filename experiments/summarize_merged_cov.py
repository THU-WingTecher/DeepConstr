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
    revise_complete_data("/artifact/experiments/results/completed.json", completed_data)
    print("Total APIs with deepconstr as the largest final BF: ", cnt, "from", all_data)
    print(f"Increase ratio of deepconstr as the largest final BF: {cnt/all_data}")
    return pd.concat([final_bf_summary, model_cnt], axis=1), completed_data 

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folders", type=str, nargs="+", help="bug report folder"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="save name"
    )
    args = parser.parse_args()

    data = {}
    for folder in args.folders:
        traverse_and_classify(folder, data)
    processed_data = process_pickle_files(data)
    aggregated_df = aggregate_summarized_data(processed_data)
    final_bf_summary, completed_data = summarize_final_bf(aggregated_df)
    print(final_bf_summary.head())
    final_bf_summary.reset_index().to_csv(os.path.join("/artifact/results/", args.output+".csv"), index=False)