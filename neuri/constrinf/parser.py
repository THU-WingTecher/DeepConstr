from typing import Tuple

def parse_from_raw_txt(raw_infer) -> Tuple[str, str] :
    """
    raw_infer txt -> answer_output, cot str
    """
    ans_flag = '```'
    if ans_flag in raw_infer :
        cot, answers = raw_infer.split(ans_flag)[0], raw_infer.split(ans_flag)[1:]
        return ';'.join([ans.replace('python','') for ans in answers]), cot.strip()
    else : 
        return '', ''
    
def segment_constr(target : str) : 
    
    """
    txt : a or b -> a, b 
    txt : a and b -> a, b
    txt : a or (b or c) -> a, b or c
    txt : (a) or (b or c and (d and e)) -> a, b or c and (d and e)
    """
    parts = []
    split_hints = [' or ', ' and '] 
    split_mark = "##"
    if not any(hint in target for hint in split_hints) :
        return [] 
    for hint in split_hints :
        target = target.replace(hint, split_mark)
    
    for part in target.split(split_mark) :
        while part.count('(') != part.count(')') :
            if part.count('(') == 0 and part.count(')') == 0 : break
            if part.count('(') > part.count(')') :
                pos = part.index('(')
                part = part[:pos] + part[pos+1:]
                
            if part.count('(') < part.count(')') :
                pos = part.rindex(')')
                part = part[:pos] + part[pos+1:]
        parts.append(part)
    return parts