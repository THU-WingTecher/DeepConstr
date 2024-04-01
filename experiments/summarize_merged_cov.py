import os
import pickle
import pandas as pd
import json
import datetime

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
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
        api_name = parts[-1]  # API name (e.g., 'torch.acos')
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

def traverse_and_classify(root_dir):
    """
    Traverses the root directory to identify relevant subdirectories and classifies the information.

    Args:
    - root_dir (str): The root directory to traverse.

    Returns:
    - dict: A dictionary with keys as API names and values as dicts with columns and paths.
    """
    api_data = {}
    for root, dirs, files in os.walk(root_dir):
        for dirname in dirs:
            if dirname.endswith('.models'):
                column, api_name = parse_directory_name(dirname)
                if column and api_name:
                    if api_name not in api_data:
                        api_data[api_name] = {}
                    api_data[api_name][column] = os.path.join(root, dirname, 'coverage', 'merged_cov.pkl')
    return api_data

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
    # Extracting and reshaping the final_bf values
    # print(aggregated_df.head())
    final_bf_summary = aggregated_df.pivot_table(index='API', columns='Column', values='Final BF', aggfunc='max')
    model_cnt = aggregated_df.pivot_table(index='API', columns='Column', values='Model Count', aggfunc='max')

    # Formatting the table to mark the largest value in each row with an asterisk
    for idx in range(final_bf_summary.shape[0]):
        # max_value = final_bf_summary.iloc[idx].max()
        # print(final_bf_summary.iloc[idx])
        max_column = final_bf_summary.iloc[idx].idxmax()
        if max_column == 'constrinf' :
            completed_data.append(final_bf_summary.iloc[idx]._name.replace(".models", ''))
            cnt+=1
        # final_bf_summary.loc[idx] = final_bf_summary.iloc[idx].apply(lambda x: f"*{x}" if x == max_value else x)

    all_data = final_bf_summary.shape[0]
    revise_complete_data("/artifact/experiments/results/completed.json", completed_data)
    print("Total APIs with constrinf as the largest final BF: ", cnt)
    print(f"Increase ratio of constrinf as the largest final BF: {cnt/all_data}")
    return pd.concat([final_bf_summary, model_cnt], axis=1), completed_data 

def save_data(final_bf, completed_data, save_dir="/artifact/experiments/results"):
    save_path = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
    complete_data_path = os.path.join(save_dir, "completed.json")
    final_bf.to_csv(save_path)
    revise_complete_data(complete_data_path, completed_data)

def merge_csvs(save_path, *args) :
    # adjusted_column_names = [name + ".models" for name in json_content]
    # extracted_columns_df_with_models = filtered_df[filtered_df.columns.intersection(adjusted_column_names)]

    import pandas as pd
    dfs = []
    for arg in args : 
        dfs.append(pd.read_csv(arg))
    
    merged_df = pd.concat(dfs)
    filtered_df = merged_df[~(merged_df == 0).any(axis=1)]

    if 'constrinf' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('constrinf', ascending=False).drop_duplicates(subset='API').sort_index()
    else:
        # If 'constrinf' column does not exist, just remove duplicates based on all columns
        filtered_df = filtered_df.drop_duplicates()

    filtered_df.reset_index(drop=True, inplace=True)

    total_rows = filtered_df.shape[0]
    for tool in ["constrinf", "constrinf_2"] :
        for baseline in ["neuri", "symbolic"] :
            columns_to_compare = [baseline] if all(col in filtered_df.columns for col in [baseline]) else []
            if tool in filtered_df.columns and columns_to_compare:
                rows_where_constrinf_is_highest = filtered_df.apply(lambda row: row[tool] > row[columns_to_compare], axis=1).sum()
            else:
                rows_where_constrinf_is_highest = "Required columns are missing or incorrectly specified."
            print("percentage_constrinf_highest of ", tool, "vs", baseline, ":", (rows_where_constrinf_is_highest / total_rows) * 100, "%")
    
    sorted_columns_df = filtered_df.sort_values(by="API")
    print(sorted_columns_df.head())
    sorted_columns_df.to_csv(save_path, index=False)

# csv_paths = [
#     "/artifact/experiments/results/merged_torch.csv",
#     "/artifact/experiments/results/torch1.csv",
#     "/artifact/experiments/results/torch2.csv",
#     "/artifact/experiments/results/torch3.csv",
#     "/artifact/experiments/results/torch4.csv",
#     "/artifact/experiments/results/torch5.csv",
# ]
# merge_csvs("/artifact/experiments/results/merged_torch_v2.csv", *csv_paths)

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         root_dir = sys.argv[1]
#     else:
#         root_dir = '/artifact/gen_tf/'

#     api_coverage_data = traverse_and_classify(root_dir)
#     processed_data = process_pickle_files(api_coverage_data)
#     aggregated_df = aggregate_summarized_data(processed_data)
#     final_bf_summary, completed_data = summarize_final_bf(aggregated_df)
#     save_data(final_bf_summary, completed_data, "/artifact/experiments/results")
    # print(final_bf_summary)