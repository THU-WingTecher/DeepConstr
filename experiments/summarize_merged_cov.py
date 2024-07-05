import os
import pickle
import pandas as pd
import json
import datetime
import numpy as np
import traceback
from experiments.process_pickle import process_pickle
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
    column = None 
    api_name = None
    parts = dirname.split('-')
    if len(parts) == 1:
        column = "temp"  # Column name (e.g., 'neuri')
    elif len(parts) == 2:
        column = parts[0]  # Column name (e.g., 'neuri')
    elif len(parts) >= 3:
        column = parts[1]  # Column name (e.g., 'neuri')
    else:
        raise ValueError(f"Invalid directory name: {dirname}")
    api_name = parts[-1].replace(folder_suffix, "")  # API name (e.g., 'torch.acos')
    return column, api_name

def traverse_and_classify(root_dir, api_data, package):
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
                    assert api_data[api_name].get(column) is None, f"the data of {api_name}-{column} is duplicated, duplicated with {api_data[api_name].get(column)}"
                    if column == "acetest" :
                        api_data[api_name][column] = os.path.join(root, dirname, f"output_{package}_0", api_name, "non_crash", 'coverage', 'merged_cov.pkl')
                    else :
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

def process_pickle_files(api_coverage_data, package):
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
                    api_summary[column] = process_pickle(pickle_path)
                    if column in ["acetest", "temp", "doctor"] : 
                        suffix = "info" if package == "tf" else "profraw"
                        files = [info_file_nm for info_file_nm in os.listdir(os.path.dirname(pickle_path)) if info_file_nm.endswith(suffix)]
                        if len(files) < 1 :
                            raise FileNotFoundError(f"{suffix} file is not exist in {os.path.dirname(pickle_path)}")
                        api_summary[column]['model_n'] = int(float(files[0].replace("."+suffix,"")))
                    else :
                        api_summary[column]['model_n'] = get_model_cnt(pickle_path)            
                except FileNotFoundError :
                    api_summary[column] = {
                        'branch_by_time': [[0, 0, 0], [0, 0, 0]],
                        'final_bf': 0,
                        'model_n': 0,
                    }
                    print("Error while reading : ", pickle_path, "Please refer /data/unnormal_vals* file and readme(unnormal values))")  
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

def summarize_final_bf(aggregated_df, pivot):
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
        if max_column == pivot :
            completed_data.append(final_bf_summary.iloc[idx]._name.replace(folder_suffix, ''))
            cnt+=1

    all_data = final_bf_summary.shape[0]
    # revise_complete_data("/DeepConstr/experiments/results/completed.json", completed_data)
    print(final_bf_summary.head())
    return pd.concat([final_bf_summary, model_cnt], axis=1), completed_data 

def get_default_coves(model_type) :
    if model_type == "torch" :
        path = os.path.join("experiments/torch_default2/coverage/merged_cov.pkl")
    elif model_type in ["tensorflow", "tf"] :
        path = os.path.join("experiments/tf_default/coverage/merged_cov.pkl")
    else :
        return 
    return process_pickle(path)


def extract_and_remove_unnormal_val(df, package):
    # Step 1: Create a mask where any value is less than 0
    unnormal_vals = {}
    mask = df <= 0

    # Step 2: Find the row labels where the condition is True for any column
    rows_to_remove = np.any(mask, axis=1)

    # Step 3: Save the positions of unnormal values
    rows, cols = np.where(mask)
    for col in cols :
        with open(f"/DeepConstr/results/unnormal_val_{df.columns[col]}_{package}.json", "w") as file:
            json.dump(list(set([df.index[row] for row in rows])), file)

    # positions = [(df.index[row], df.columns[col]) for row, col in zip(rows, cols) if not str(df.columns[col]).endswith("cnt")]


    # Step 4: Remove rows containing unnormal values from the dataframe
    df_clean = df[~rows_to_remove]

    return df_clean

def get_api_list_saved_path(tool, package) : 
    if "symbolic" in tool :
        tool = "nnsmith" 
    if "deepconstr_2" in tool or \
        "deepconstr" in tool:
        tool = "deepconstr"
    return f"{package}_{tool}.json"

def get_intersected(tool1, tool2, package) : 
    data = []
    root_dir = "./data"
    for tool in [tool1, tool2] :
        file_name = get_api_list_saved_path(tool, package)
        file_name = os.path.join(root_dir, file_name)
        with open("/DeepConstr/data/torch_dc_doctor.json", "r") as f :
            data.append(json.load(f))

    intersected = list(set(data[0]).intersection(set(data[1])))
    return intersected
def customize_concat(dataframes):
    """
    Concatenate a list of dataframes side by side (column-wise) and resolve any duplicate columns
    by keeping only the maximum value for each (index, column) pair.

    Args:
    dataframes (list of pd.DataFrame): List of DataFrames to concatenate.

    Returns:
    pd.DataFrame: A single DataFrame after handling duplicates by selecting maximum values.
    """
    # Concatenate the dataframes with keys to track their origin
    df_concat = pd.concat(dataframes, axis=1, keys=[f'df{i}' for i in range(len(dataframes))])

    # Flatten the MultiIndex columns for easier manipulation
    df_concat.columns = ['_'.join(col).strip() for col in df_concat.columns.values]

    # Identify duplicated based on the base column name (ignoring df1_, df2_, etc. prefixes)
    base_cols = [col.split('_')[1] for col in df_concat.columns]
    duplicates = set(col for col in base_cols if base_cols.count(col) > 1)

    # Resolve duplicates by taking the max across the original dataframes' columns
    for col in duplicates:
        max_col = df_concat[[f'df{i}_{col}' for i in range(len(dataframes))]].max(axis=1)
        df_concat[col] = max_col    # Assign max values to a new column or overwrite existing one
        # Drop the original duplicated columns
        for i in range(len(dataframes)):
            df_concat.drop([f'df{i}_{col}'], axis=1, inplace=True)

    return df_concat

def gen_table4_from_df(*args, pivot="deepconstr", package="torch") :

    dfs = []
    for path in args :
        if isinstance(path, str) :
            dfs.append(pd.read_csv(path, index_col="API"))
        elif isinstance(path, pd.DataFrame) :
            dfs.append(path)
    # df = customize_concat(dfs)
    df = pd.concat(dfs, axis=1)
    df = df.set_index('API', drop=True)
    print(df.head())
    default_val = get_default_coves(package)['final_bf']
    cov_col_names = [col for col in df.columns if "cnt" not in col]
    for col in cov_col_names :
        # print(df[col], default_val)
        df[col] = df[col] - default_val
    df = extract_and_remove_unnormal_val(df, package)

    for tool in [pivot] :
        for baseline in [base for base in cov_col_names if base != tool] :
            columns_to_compare = [baseline] if all(col in df.columns for col in [baseline]) else []
            intersected = get_intersected(pivot, baseline, package)

            extracted_columns_df_with_models = df.loc[df.index.intersection(intersected)]

            for col in columns_to_compare:
                improvement_col_name = f"improvement_ratio_{tool}_vs_{col}"

                # Calculate the improvement ratio and ensure it's a single value per row (Series)
                extracted_columns_df_with_models[improvement_col_name] = extracted_columns_df_with_models.apply(
                    lambda row: (row[tool] - row[col]) / row[col] if row[tool] > 0 and row[col] > 0 else 0, axis=1)
                
                # Correctly update the original DataFrame
                df.loc[extracted_columns_df_with_models.index, improvement_col_name] = extracted_columns_df_with_models[improvement_col_name]
            
            rows_where_pivot_is_highest = extracted_columns_df_with_models.apply(
                lambda row: row[tool] > row[columns_to_compare], axis=1).sum()
            
            print(f"rows_where_{pivot}_is_highest {tool} vs {baseline}: highest {rows_where_pivot_is_highest}, intersected {len(intersected)} apis")
            print(f"After removed unnormal values, we have {extracted_columns_df_with_models.shape[0]} apis")
            # print(f"Total APIs with {pivot} as the largest final BF: ", cnt, "from", all_data)
            print(f"Increase ratio of {pivot} as the largest single-operator coverage: {rows_where_pivot_is_highest/extracted_columns_df_with_models.shape[0]}")
        
    df = df.sort_values(by=f'improvement_ratio_{pivot}_vs_{col}', ascending=True)
    for col in df.columns:
        print(col)
        print(df[df[col] != 0][col].mean())
    sorted_df = df.reset_index()
    return sorted_df

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folders", type=str, nargs="+", help="bug report folder"
    )
    parser.add_argument(
        "-p", "--pivot", type=str, help="pivot column to compare"
    )
    parser.add_argument(
        "-k", "--package", type=str, help="package name"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="save name"
    )
    parser.add_argument(
        "-l", "--load", type=str, default=None, help="load file"
    )
    args = parser.parse_args()
    assert args.output.endswith(".csv"), "output argument should be a csv file name"
    assert args.load is None or args.load.endswith(".csv") , "load argument should be a csv file name and with raw_file"
    data = {}
    for folder in args.folders:
        traverse_and_classify(folder, data, args.package)
    # print(data)
    processed_data = process_pickle_files(data, args.package)
    aggregated_df = aggregate_summarized_data(processed_data)
    final_bf_summary, completed_data = summarize_final_bf(aggregated_df, args.pivot)
    final_bf_summary = final_bf_summary.reset_index()
    final_bf_summary.to_csv(os.path.join("/DeepConstr/results/", f"raw_{args.output}"), index=False)
    gen_table4_from_df(final_bf_summary, args.load, pivot=args.pivot, package=args.package).to_csv(os.path.join("/DeepConstr/results/", f"{args.output}"), index=False)