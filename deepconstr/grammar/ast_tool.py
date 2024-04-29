import ast
import re 
from typing import List
import operator as op

from deepconstr.error import IncorrectConstrError 

class AstNameFinder(ast.NodeVisitor):  # Changed to NodeVisitor as we only need to visit nodes
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.id)
        self.generic_visit(node)
    def visit_Constant(self, node):
        self.names.add(node.value)
        self.generic_visit(node)

def clean_txt(line : str) -> str :
    # line = re.sub(r'^[-\d]+', '', line)
    return line.strip()

def identify_related_args(astree, arg_names) : 
    finder = AstNameFinder()
    finder.visit(astree)
    arg_names = finder.names.intersection(arg_names)
    if not arg_names :
        raise IncorrectConstrError(f"Cannot find any related args in the constraint: {ast.dump(astree)}")
    return arg_names

def is_same_ast_name(left_ast : str, right_ast) : 
    return left_ast == right_ast.__name__

def get_operator(astop : str):
    if is_same_ast_name(astop, ast.Eq):
        return lambda a,b : op.eq(a, b)
    elif is_same_ast_name(astop, ast.NotEq):
        return lambda a,b : op.ne(a, b)
    elif is_same_ast_name(astop, ast.Lt):
        return lambda a,b : op.lt(a, b)
    elif is_same_ast_name(astop, ast.LtE):
        return lambda a,b : op.le(a, b)
    elif is_same_ast_name(astop, ast.Gt):
        return lambda a,b : op.gt(a, b)
    elif is_same_ast_name(astop, ast.USub):
        return lambda a: -a
    elif is_same_ast_name(astop, ast.GtE):
        return lambda a,b : op.ge(a, b)
    elif is_same_ast_name(astop, ast.Is):
        return lambda a,b : op.eq(a, b)
    elif is_same_ast_name(astop, ast.IsNot):
        return lambda a,b : op.ne(a, b)
    elif is_same_ast_name(astop, ast.In):
        from deepconstr.grammar import SMTFuncs
        return lambda a,b : SMTFuncs.in_(a,b)
    elif is_same_ast_name(astop, ast.NotIn):
        from deepconstr.grammar import SMTFuncs
        return lambda a,b : SMTFuncs.not_in(a,b)
    elif is_same_ast_name(astop, ast.Add):
        return lambda a,b : op.add(a, b)
    elif is_same_ast_name(astop, ast.Sub):
        return lambda a,b : op.sub(a, b)
    elif is_same_ast_name(astop, ast.Mult):
        return lambda a,b : op.mul(a, b)
    elif is_same_ast_name(astop, ast.MatMult):
        return lambda a,b : op.matmul(a, b)
    elif is_same_ast_name(astop, ast.Div):
        return lambda a,b : op.truediv(a, b)
    elif is_same_ast_name(astop, ast.FloorDiv):
        return lambda a,b : op.truediv(a, b)
    elif is_same_ast_name(astop, ast.Mod):
        return lambda a,b : op.mod(a, b)
    elif is_same_ast_name(astop, ast.Pow):
        return lambda a,b : op.pow(a, b)
    else:
        raise ValueError(f"Unknown operator: {astop}")
