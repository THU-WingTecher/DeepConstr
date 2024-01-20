import ast
from typing import List, Union, Any, Callable, Tuple, Optional
import z3 
from neuri.abstract.tensor import AbsTensor
from specloader import SHAPE_ATTR, RANK_ATTR, LEN_ATTR, \
        TYPE_ATTRS, gen_arr_len_z3
from specloader import Z3TENSOR

SYM_LEN = {}
class Select : 
    def __init__(self, name, idx = None, attr = None) :
        self.name = name 
        self.idx = idx 
        self.binops : List[Tuple[Callable, Union[int, z3.Int]]] = []
        self.attr = attr
        self.funcs : List[Callable] = [] 
    def __repr__(self) : return f'Select:{self.name}'
    def set_func(self, func) : self.funcs.append(func)
    def set_idx(self, idx) : self.idx = idx
    def set_binops(self, ops) : self.binops.append(ops) # op, right
    def set_attr(self, con) : self.attr = con
    def has_symbol(self) : return not (self.idx is None or isinstance(self.idx, int))
    def gen_z3_obj(self, dtype_dict) : # numel implement
        if not isinstance(self.name, str) : return self.name # end recursion
        dtype = dtype_dict[self.name]
        if isinstance(dtype, AbsTensor) : 
            z3 = dtype.z3()(self.name)
            if self.attr is None :
                return Z3TENSOR.shape(z3)
            elif self.attr == SHAPE_ATTR :
                return Z3TENSOR.shape(z3)
            elif self.attr in TYPE_ATTRS :
                return Z3TENSOR.dtype(z3)
            elif self.attr in [LEN_ATTR, RANK_ATTR] :
                return Z3TENSOR.rank(z3)
            else : #attr not assigned -> default : shape
                return Z3TENSOR.shape(z3)
        else : 
            if self.attr is None :
                return dtype.z3()(self.name)
            elif self.attr == LEN_ATTR : 
                return gen_arr_len_z3(self.name)
            else :
                return dtype.z3()(self.name)
    def change_sym_pos(self, iter_name) : 
        # name : i, idx : None -> name : iter.name, idx : i
        self.idx = self.name
        self.name = iter_name
    def export_len_var(self, dtype_dict) :
        dtype = dtype_dict[self.name]
        if isinstance(dtype, AbsTensor) :
            return Z3TENSOR.rank(dtype.z3()(self.name))
        else :
            return gen_arr_len_z3(self.name)
    def gen_idx(self, length) :
        idx = self.idx
        for binop in self.binops : 
            idx = binop[0](idx, binop[1])
        return z3.If(idx>=0, idx, length+idx)
    def concrete(self, dtype_dict) :
        z3_obj = self.name
        while isinstance(z3_obj, Select) or \
            isinstance(self.idx, Select) :
            if isinstance(z3_obj, Select) : 
                z3_obj = z3_obj.concrete(dtype_dict)
            if isinstance(self.idx, Select) : 
                self.idx = self.idx.concrete(dtype_dict)
        
        length = self.export_len_var(dtype_dict)
        SYM_LEN.update({self.name : length})
        z3_obj = self.gen_z3_obj(dtype_dict)
        if self.idx is None : 
            z3_obj = z3_obj
        else :
            idx = self.gen_idx(length)
            z3_obj = z3_obj[idx]
        if self.funcs :
            for func in self.funcs : 
                z3_obj = func(z3_obj, length = length)

        return z3_obj
        

class IRexpr : 
    def __init__(self, ops : ast.Expr, values : List[Any]) : 
        self.ops = ops 
        self.values : List[Union[IRexpr, IRcompare]] = values
        self.whereis_symbols : List[Optional[Select]] = []
    def __repr__(self) : return f'IRexpr:[{self.values}]'
    def find_sym(self, obj) :
        for value in self.values : 
            if hasattr(value, 'find_sym') :
                value.find_sym(obj)
    def add_sym_pos(self, obj, attr) :
        self.whereis_symbols.append((obj, attr))
    def assign(self, sym : z3.Int) :
        for value in self.values : 
            value.assign(sym)
        for whereis_symbol in self.whereis_symbols :
            obj, attr = whereis_symbol # attr == name 
            setattr(obj, attr, sym)

    def concrete(self, dtype_dict, iters = None) : 
        from specloader.ast2z3 import convert_boolop_to_z3
        res = []
        op = convert_boolop_to_z3(self.ops)
        for value in self.values : 
            if hasattr(value, 'concrete') :
                res += value.concrete(dtype_dict, iters)
            else :
                res += value if hasattr(value, '__len__') else [value] 

        return [op(*res)]
    
class IRcompare(IRexpr) : 
    def __init__(self, left, ops, comparators) :
        self.left : Select = left 
        self.comparators : List[Optional[Select]] = comparators  
        self.whereis_symbols = []
        self.ops = ops 
    def __repr__(self) : return f'ir[{self.left}-{self.comparators}]'
    def add_sym_pos(self, obj, attr) :
        self.whereis_symbols.append((obj, attr))
    def assign(self, sym : z3.Int) :
        for whereis_symbol in self.whereis_symbols : 
            obj, attr = whereis_symbol # attr == name 
            setattr(obj, attr, sym)
    def _find_sym(self, obj, sym) :
        if isinstance(obj, Select) :
            if isinstance(obj.idx, Select) :
                self._find_sym(obj.idx, sym)
            if obj.idx == sym :
                self.add_sym_pos(obj, 'idx')  
    def find_sym(self, sym) :
        self._find_sym(self.left, sym)
        for comparator in self.comparators :
            self._find_sym(comparator, sym)  
        if self.left == sym :
            self.add_sym_pos(self, 'left')
    def concrete(self, dtype_dict, iters = None, dim = 0) :
        res = []
        left_objs = self.left.concrete(dtype_dict) if hasattr(self.left, 'concrete') else self.left
        for right_obj in self.comparators :
            right_obj = right_obj.concrete(dtype_dict) if hasattr(right_obj, 'concrete') else right_obj
            res+= [self.ops(left_objs, right_obj)]
        return res 

def symbolize_idx(idx) : 
    return z3.Int(idx)