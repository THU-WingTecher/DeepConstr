import ast
from typing import List, Union, Dict, Any, Callable, Tuple, Optional
from logger import LOGGER
from specloader import TENSOR_ATTRS_MAPPING
import yaml

class ArgsSearcher(ast.NodeTransformer):
    def __init__(self, arg_names : List[str]) -> None:
        self.arg_names = arg_names 
        self.related_args = set()
    def visit_Name(self, node):
        if node.id in self.arg_names :
            self.related_args.add(node.id)
            # new_node = ast.Subscript(
            #     value = ast.copy_location(ast.Name(id='args', ctx=ast.Load()), node),
            #     slice=ast.copy_location(ast.Constant(value=node.id), node),
            #     ctx=ast.Load()
            # )
            # LOGGER.debug(f'replaced the Node {ast.dump(node)} with {ast.dump(new_node)}')
            # return ast.copy_location(new_node, node)
        return self.generic_visit(node)
    def visit_Constant(self, node):
        if node.value in self.arg_names :
            self.related_args.add(node.value)
            # new_node = ast.Subscript(
            #     value = ast.copy_location(ast.Name(id='args', ctx=ast.Load()), node),
            #     slice=node,
            #     ctx=ast.Load()
            # )
            # LOGGER.debug(f'replaced the Node {ast.dump(node)} with {ast.dump(new_node)}')
            # return ast.copy_location(new_node, node)
        return self.generic_visit(node)
    def get(self) : return list(self.related_args)

class Oppositor(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node, ast.Expression) :
            new_node = ast.Call(
                func=ast.copy_location(ast.Name(id=TENSOR_ATTRS_MAPPING[node.attr], ctx=ast.Load()), node),
                args=[node.value],
                keywords=[],
            )
            # print(f'replaced the Node {ast.dump(node)} with {ast.dump(new_node)}')
            LOGGER.debug(f'replaced the Node {ast.dump(node)} with {ast.dump(new_node)}')
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)
    
class NodeReplacer(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if hasattr(node, 'attr') and node.attr in TENSOR_ATTRS_MAPPING.keys():
            new_node = ast.Call(
                func=ast.copy_location(ast.Name(id=TENSOR_ATTRS_MAPPING[node.attr], ctx=ast.Load()), node),
                args=[node.value],
                keywords=[],
            )
            # print(f'replaced the Node {ast.dump(node)} with {ast.dump(new_node)}')
            LOGGER.debug(f'replaced the Node {ast.dump(node)} with {ast.dump(new_node)}')
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and hasattr(node.func, 'attr') and node.func.attr in TENSOR_ATTRS_MAPPING.keys():
            new_node = ast.Call(
                func=ast.copy_location(ast.Name(id=TENSOR_ATTRS_MAPPING[node.func.attr], ctx=ast.Load()), node),
                args=[node.func.value],
                keywords=[],
            )
            # print(f'replaced the Node {ast.dump(node)} with {ast.dump(new_node)}')
            LOGGER.debug(f'replaced the Node {ast.dump(node)} with {ast.dump(new_node)}')
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)

def parse_and_transform(code):
    tree = ast.parse(code, mode='eval')
    transformer = NodeReplacer()
    return transformer.visit(tree)


def set_location(node, lineno, col_offset):
    node.lineno = lineno
    node.col_offset = col_offset
    for child in ast.iter_child_nodes(node):
        set_location(child, lineno, col_offset)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

