import re
from typing import Any, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ErrorMessage:
    def __init__(self, msg: str, values, choosen_dtype):
        """
        Initialize an instance with error details.

        :param error_message: The textual error message.
        :param args: The positional arguments associated with the error.
        :param kwargs: The keyword arguments associated with the error.
        """
        self.msg = msg
        self.values : Dict[str, Any] = values
        self.chooen_dtype : Dict[str, Any] = choosen_dtype
    
    def __repr__(self):
        return f"ErrMeg({self.get_core_msg()}, {self.values})"
    
    def get_values_map(self) -> Dict[str, Any]:
        """
        Retrieve the error details.

        :return: A tuple containing the error message, args, and kwargs.
        """
        return self.values
    
    def get_dtypes(self) -> List[Any] :
        return list(self.chooen_dtype.values())

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
        return msg.strip()
    def delete_after(self, msg, chr, include=False) : 
        if chr not in msg :
            return msg
        end = msg.index(chr) if include else msg.index(chr)-1  # Find the start of the JSON object
        res = msg[:end]
        return res

    def get_core_msg(self) -> str :
        if self.msg == "no error" :
            return self.msg
        first_pos = self.msg.find("Error:")
        if first_pos != -1 and self.msg.count('a') > 1:
            error = self.msg[first_pos:].strip()
            if self.package == 'tf' :
                error = self.clean_errmsg(error) #tf special
            return error 
        else :
            return self.msg 


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
    return sentence_similarity(mask_numbers(sen1), mask_numbers(sen2)) >= threshold

def mask_numbers(sentence, mask_char="#"):
    return re.sub(r'-?\d+', mask_char, sentence)

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