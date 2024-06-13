import sys, pickle


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

def process_pickle(path) : 
    with open(path, 'rb') as file:
        data = pickle.load(file)
        # Summarize the data
        branch_by_time, final_bf = cov_summerize(data)
        return {
            'branch_by_time': branch_by_time,
            'final_bf': max([bf for _,_,bf in branch_by_time]),
        }

if __name__ == "__main__" :
    import os
    for path in sys.argv[1:]:
        print(os.path.dirname(path), process_pickle(path))

# find exp/acetest/tf_wo_xla/ -type f -name "merged_cov.pkl" -exec python experiments/process_pickle.py {} +