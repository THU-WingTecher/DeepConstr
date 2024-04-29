import re
from typing import Any, Dict, List, Literal
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from deepconstr.grammar.dtype import materalize_dtypes
from deepconstr.utils import formatted_dict
from deepconstr.logger import TRAIN_LOG

def delete_after(msg, chr, include=False) : 
    if chr not in msg :
        return msg
    end = msg.index(chr) if include else msg.index(chr)-1  # Find the start of the JSON object
    res = msg[:end]
    return res

class ErrorMessage:
    def __init__(self, msg: str, traceback_msg : str, values, choosen_dtype, package : Literal["torch", "tensorflow"] = "torch"):
        """
        Initialize an instance with error details.

        :param error_message: The textual error message.
        :param args: The positional arguments associated with the error.
        :param kwargs: The keyword arguments associated with the error.
        """
        self.error_type = None 
        self.values = None
        self.msg = msg
        if isinstance(msg, Exception) :
            self.error_type = type(msg)
            self.msg = str(msg)
        else :
            self.error_type = None
        self.traceback = traceback_msg
        if values is not None :
            self.values : Dict[str, Any] = values
        self.chooen_dtype : Dict[str, Any] = choosen_dtype
        self.package = package
    
    def dump(self) :
        return {
            "msg" : self.msg,
            "choosen_dtype" : {name : dtype.dump() for name, dtype in self.chooen_dtype.items() \
                               if dtype is not None},
            "package" : self.package
        }
    @staticmethod
    def load(data) :
        return ErrorMessage(data["msg"], 
                            "loaded",
                            None, 
                            {name : materalize_dtypes(dtype)[0] for name, dtype in data["choosen_dtype"].items()}, 
                            data["package"])
    def __repr__(self):
        return f"{self.error_type}({self.get_core_msg()})[{formatted_dict(self.get_values_map())}]"
    
    def get_values_map(self) -> Dict[str, Any]:
        """
        Retrieve the error details.

        :return: A tuple containing the error message, args, and kwargs.
        """
        if self.values is None :
            TRAIN_LOG.warning(f"Loaded ErrorMessage has no values")
            return {}
        return self.values
    
    def get_summarized(self) -> str :
        return f"{self.get_core_msg()}[{formatted_dict(self.get_values_map())}]"
    
    def get_dtypes(self, names) -> List[Any] :
        dtypes = [None]*len(names)
        for i, name in enumerate(names) :
            dtypes[i] = self.chooen_dtype[name]
        return dtypes

    def get_dtypes_map(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Retrieve the error details.

        :return: A tuple containing the error message, args, and kwargs.
        """
        return self.chooen_dtype
    
    def sculpt_msg(self):
        """
        Sculpt the error message.

        :return: The sculpted error message.
        """
        pass
    def clean_errmsg(self, msg : str) -> str : 
        # msg = self.delete_after(msg, '\t')
        msg = delete_btw(msg, '{', '}')
        msg = delete_btw(msg, '<', '>')
        msg = msg.replace('\t',' ').replace('  ',' ')
        msg = delete_after(msg, '\n')
        return msg.strip()

    def get_core_msg(self) -> str :
        msg = self.msg
        if msg == "no error" :
            return msg
        first_pos = msg.find("Error:")
        if self.package == 'tensorflow' :
            msg = self.clean_errmsg(msg) #tf special
        if first_pos != -1 and self.msg.count('a') > 1:
            msg = msg[first_pos:].strip()
            return msg 
        else :
            return msg

def delete_btw(msg, start_chr, end_chr, include=False):
    stack = []
    to_delete = []
    
    for i, c in enumerate(msg):
        if c == start_chr:
            stack.append(i)
        elif c == end_chr and stack :
            start = stack.pop()
            if not stack:
                end = i
                if not include:
                    to_delete.append((start, end))
                else:
                    to_delete.append((start + 1, end - 1))
    
    # Reverse the list so that we delete from the end of the string first
    to_delete.reverse()
    
    for start, end in to_delete:
        msg = msg[:start] + msg[end + 1:]
    
    return msg

def sentence_similarity(sentence1, sentence2):
    # Create the Document Term Matrix
    vectorizer = TfidfVectorizer().fit_transform([sentence1, sentence2])

    # Get the vectors for each sentence
    vector1 = vectorizer[0] 
    vector2 = vectorizer[1]

    # Calculate and return the cosine similarity
    return cosine_similarity(vector1, vector2)[0, 0]
  
def is_similar(sen1, sen2, threshold=0.5) :
    return sentence_similarity(masking(sen1), masking(sen2)) >= threshold

def masking(sentence, mask_char="#"):
    return mask_numbers(mask_strings(sentence, mask_char), mask_char)

def mask_numbers(sentence, mask_char="#"):
    return re.sub(r'-?\d+', mask_char, sentence)

def mask_strings(sentence, mask_char="#"):
    masked = re.sub(r'"[^"]*"', f'"{mask_char}"', sentence)
    return masked

def sort_sentences_by_similarity(target_sentence, sentence_list):
    vectorizer = TfidfVectorizer()

    # Combine the target sentence and the list of sentences to vectorize them together
    combined_list = [target_sentence] + sentence_list
    vectorized_sentences = vectorizer.fit_transform(combined_list)
    
    # Calculate the cosine similarity between the target sentence and all other sentences
    cosine_similarities = cosine_similarity(vectorized_sentences[0:1], vectorized_sentences[1:]).flatten()
    
    # Pair each sentence with its similarity score
    paired_with_similarity = [(sentence, score) for sentence, score in zip(sentence_list, cosine_similarities)]

    # Sort the list of sentences by similarity to target_sentence
    sorted_by_similarity = sorted(paired_with_similarity, key=lambda x: x[1], reverse=True)

    return sorted_by_similarity

from sklearn.metrics import silhouette_score

def find_optimal_clusters(data, max_clusters=10):
    """
    Finds the optimal number of clusters for KMeans clustering based on the silhouette score.

    Parameters:
    - data: The TF-IDF matrix of the data.
    - max_clusters (int): The maximum number of clusters to consider.

    Returns:
    - int: The optimal number of clusters.
    """
    silhouette_scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n_clusters in range(2, min(max_clusters, data.shape[0]) + 1):
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            cluster_labels = clusterer.fit_predict(data)
            if len(set(cluster_labels)) == 1:
                return 1  # Indicates clustering is not meaningful
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append((n_clusters, silhouette_avg))
        # Finding the number of clusters with the highest silhouette score
        optimal_clusters = sorted(silhouette_scores, key=lambda x: x[1], reverse=True)[0][0]
    return optimal_clusters

from sklearn.metrics import silhouette_score


# dummy_data_count = 5  # Number of dummy strings to add
# DUMMY_STR = ["DUMMY_DATA_FOR_CLUSTERING",  # Dummy string for padding
#                     "shape should consistence matrix, but we got nothing but you"
# ]
# dummy_strings = DUMMY_STR * dummy_data_count


def map_error_messages_to_clusters_dynamic(raw_error_messages, threshold=0.5):
    """
    Maps each error message to a cluster based on structural characteristics using TF-IDF and dynamically
    determined KMeans clustering based on silhouette scores.

    Parameters:
    - raw_error_messages (list of str): A list containing the raw error messages.

    Returns:
    - dict: A dictionary where keys are cluster labels and values are lists of error messages belonging to each cluster.
    """
    # cosine similarity
    clusters : List[List[str]] = [] 
    found = None
    for err_msg in raw_error_messages :
        found = False
        if clusters : 
            for _cls in clusters : 
                try :
                    if is_similar(err_msg, _cls[0], threshold) :
                        _cls.append(err_msg)
                        found = True
                        break
                except : 
                    TRAIN_LOG.warning(f"Error in comparing {err_msg} and {_cls[0]}, may be two strings are empty")
                    continue 
            if not found :
                if err_msg :
                    clusters.append([err_msg])
        else :
            clusters.append([err_msg])
    return {i: cls for i, cls in enumerate(clusters)}
    # k-mean clusters
    # raw_error_messages += dummy_strings
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # tfidf_matrix = tfidf_vectorizer.fit_transform(raw_error_messages)

    # # Dynamically determining the optimal number of clusters
    # optimal_clusters = find_optimal_clusters(tfidf_matrix, max_clusters=10)
    # km = KMeans(n_clusters=optimal_clusters, random_state=42, n_init="auto")
    # km.fit(tfidf_matrix)

    # clusters = km.labels_.tolist()
    # cluster_mapping = {}
    # for cluster_label, error_message in zip(clusters, raw_error_messages):
    #     if error_message in DUMMY_STR:  # Skip dummy data in final output
    #         continue
    #     if cluster_label not in cluster_mapping:
    #         cluster_mapping[cluster_label] = [error_message]
    #     else:
    #         cluster_mapping[cluster_label].append(error_message)

    # return cluster_mapping

