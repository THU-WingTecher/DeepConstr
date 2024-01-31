import ast
from typing import List, Union, Dict, Any, Callable, Tuple, Optional, get_args
import z3
import operator as op
import typing 
# 'numel', not defined 
SHAPE_ATTR = 'shape'
RANK_ATTR = 'rank'
LEN_ATTR = 'len'
TYPE_ATTRS = ['type', 'dtype']
TYPE_ATTR = 'type'
MAX_ARR_LEN = 6
# MAX_SHAPE_SUM = 2 * 1024**3 / 16
MAX_SHAPE_SUM = 2 * 1024**2 / 16
MIN_VALUE = -4
MAX_VALUE = 9
NONE_VALUE = -999
Z3_SOLVERABLE_FUNCS = ['all', 'any', 'min', 'max', 'abs']
TYPE_FUNCS = ['type', 'isinstance']

TENSOR_SIZE_EXPRS = ('shape', 'size', 'numel')

TENSOR_ATTRS_MAPPING = {
    'dim' : 'rank',
    'dims' : 'rank',
    'ndim' : 'rank',
    'ndims' : 'rank'
}
NOT_SIGN = "~"
COMPARISON_KINDS = [
    z3.Z3_OP_GE,  # Greater than or equal
    z3.Z3_OP_GT,  # Greater than
    z3.Z3_OP_LE,  # Less than or equal
    z3.Z3_OP_LT,  # Less than
    z3.Z3_OP_EQ,  # Equal
    z3.Z3_OP_DISTINCT  # Not equal
]
BOOL_KINDS = [
    z3.Z3_OP_NOT,
    z3.Z3_OP_AND,
    z3.Z3_OP_OR
]
OP_POOLS = [
           op.lt,
           op.le,
           op.gt,
           op.ge]
POS_OPS = [ast.In, ast.Is, ast.Eq]
NEG_OPS = [ast.NotIn, ast.IsNot, ast.NotEq]
BOOL_POOLS = [op.ne, op.eq,]

###### TENSOR DTYPE DEINITION ######

## Define : tensor, int, float, bool, complex, str
## Define : Array(tensor, int, float, bool, complex, str)
## Will be updated to support Dict, object

Z3DTYPE = z3.Datatype('DType')
Z3DTYPE.declare('float32')
Z3DTYPE.declare('int32')
Z3DTYPE.declare("qint8")
Z3DTYPE.declare("qint16")
Z3DTYPE.declare("qint32")
Z3DTYPE.declare("bfloat16")
Z3DTYPE.declare("float16")
Z3DTYPE.declare("float32")
Z3DTYPE.declare("float64")
Z3DTYPE.declare("uint8")
Z3DTYPE.declare("uint16")
Z3DTYPE.declare("uint32")
Z3DTYPE.declare("uint64")
Z3DTYPE.declare("int8")
Z3DTYPE.declare("int16")
Z3DTYPE.declare("int32")
Z3DTYPE.declare("int64")
Z3DTYPE.declare("bool")
Z3DTYPE.declare("complex32")
Z3DTYPE.declare("complex64")
Z3DTYPE.declare("complex128")
Z3DTYPE = Z3DTYPE.create()
TensorZ3 = z3.Datatype('z3tensor')
TensorZ3.declare('tensor_instance', 
                ('shape', z3.ArraySort(z3.IntSort(), z3.IntSort())),
                ('dtype', Z3DTYPE),
                ('rank', z3.IntSort())),
TensorZ3 = TensorZ3.create()

###### TENSOR DTYPE DEINITION ######
class z3_funcs():
    func_names = ['all', 'len', 'any', 'min', 'max', 'abs', 'in_', 'not_in']
    # @staticmethod
    # def numel(vs, length=None):
    #     numel = z3.Int('numel')
    #     z3.ForAll([z3.Int('i')], 
    #               z3.Implies(z3.And(0 <= z3.Int('i'), z3.Int('i') < length), A[z3.Int('i')] != 0)))

    # re
    @staticmethod
    def sorted(vs, length=None):
        if isinstance(vs, int):
            return abs(vs)
        elif hasattr(vs, '__len__'):
            return [vs[i] <= vs[i + 1] for i in range(len(vs) - 1)]
        else:
            return [z3.If(vs[i] >= 0, vs[i], -vs[i]) for i in range(MAX_ARR_LEN-1)]

    @staticmethod
    def abs(vs, length=None):
        if isinstance(vs, int):
            return abs(vs)
        elif hasattr(vs, '__len__'):
            return [z3.If(x >= 0, x, -x) for x in vs]
        else:
            return [z3.If(vs[i] >= 0, vs[i], -vs[i]) for i in range(MAX_ARR_LEN)]
    # Return minimum of a vector; error if empty
    @staticmethod
    def min(vs, length=None):
        if not hasattr(vs, '__len__') :
            min_idx = z3.Int('min_idx')
            i = z3.Int('i')
            max_constraint = z3.ForAll([i], z3.Implies(z3.And(i >= 0, i < length), vs[min_idx] <= vs[i]))
            exists_in_array = z3.And(min_idx >= 0, min_idx < length)

            CONSTRAINTS.extend([exists_in_array, max_constraint])
            return vs[min_idx] 
        else :
            if len(vs) == 0 :
                return None
            if len(vs) == 1 :
                return vs[0]

            m = vs[0]
            for v in vs[1:]:
                m = z3.If(v < m, v, m)
            return m

    # Return maximum of a vector; error if empty
    @staticmethod
    def max(vs, length=None):
        if not hasattr(vs, '__len__') :
            max_idx = z3.Int('max_idx')
            i = z3.Int('i')
            max_constraint = z3.ForAll([i], z3.Implies(z3.And(i >= 0, i < length), vs[max_idx] >= vs[i]))
            exists_in_array = z3.And(max_idx >= 0, max_idx < length)

            CONSTRAINTS.extend([exists_in_array, max_constraint])
            return vs[max_idx]

        else :
            if len(vs) == 0 :
                return None
            if len(vs) == 1 :
                return vs[0]
            m = vs[0]
            for v in vs[1:]:
                m = z3.If(v > m, v, m)
            return m

    @staticmethod
    def in_(a,b) : 
        from specloader.irs import SYM_LEN
        if isinstance(b[0], z3.ArrayRef) : 
            b_len = SYM_LEN[b[0].arg(0).decl().name()] if b[0].num_args() else SYM_LEN[b[0].decl().name()]
            i = z3.Int('i')
            exists = z3.Exists([i], z3.Implies(z3.And(i >= 0, i < b_len), b[0][i] == a))
            return exists
        elif hasattr(b, '__len__') :
            return z3.Or([a == v for v in b])
        else :
            return z3.Or([a == b[i] for i in range(MAX_ARR_LEN)])
    
    @staticmethod
    def not_in(a,b) : 
        if isinstance(b[0], z3.ArrayRef) : 
            from specloader.irs import SYM_LEN
            b_len = SYM_LEN[b[0].arg(0).decl().name()] if b[0].num_args() else SYM_LEN[b[0].decl().name()]
            i = z3.Int('i')
            exists = z3.ForAll([i], z3.Implies(z3.And(i >= 0, i < b_len), b[0][i] != a))
            return exists
        elif hasattr(b, '__len__') :
            return z3.And([a != v for v in b])
        else :
            return z3.And([a != b[i] for i in range(MAX_ARR_LEN)])

def len_name(arg_name) :
    return arg_name + '_len'
def gen_arr_len_z3(arg_name : str) : 
    return z3.Int(len_name(arg_name))

def gen_len_obj(obj : z3.ExprRef, suff_conds_needed : Dict[str, bool] = {}) -> z3.ExprRef :
    """ 
    we are building suff_cond by the index of array.
    however, for generatorexpr which are a[0]>1 ... a[MAX_LEN]>1, this way of building would be wrong.
    Therefore, for generatorexpr, we disable suff_cond building.
    """
    if len(suff_conds_needed) != 0 :
        if obj.decl().name() == TensorZ3.shape.name() :
            if suff_conds_needed[obj.arg(0).decl().name()] :
                return TensorZ3.rank(obj.arg(0))
            else :
                return None
        else : 
            if suff_conds_needed[obj.decl().name()] :
                return gen_arr_len_z3(obj.decl().name())
    else :
        if obj.decl().name() == TensorZ3.shape.name() :
            return TensorZ3.rank(obj.arg(0))
        else : 
            return gen_arr_len_z3(obj.decl().name())
def length_default_constraints(length) : 
    return z3.And(length >= 0, length <= MAX_ARR_LEN)
def length_not_zero_constraints(length) : 
    return z3.And(length > 0, length <= MAX_ARR_LEN)
def pos_constraints(z3obj, len_var, include_zero) : 
    i = z3.Int('i')
    if include_zero :
        return z3.ForAll([i], z3.Implies(z3.And(i>=0, i<=len_var), z3obj[i] >= 0))
    else :
        return z3.ForAll([i], z3.Implies(z3.And(i>=0, i<=len_var), z3obj[i] > 0))
    
# def pos_constraints(z3obj, len_var, include_zero) : 
#     return z3_funcs.not_in(0, [z3obj[i] for i in range(MAX_ARR_LEN)])
def check_numel_constr(shape, len_var=None) :
    # Create a Z3 array of integers

    # Recursive function to calculate the product of elements in the array
    def array_product(arr, length):
        if length == 0:
            return 1
        else:
            return arr[length - 1] * array_product(arr, length - 1)

    # Z3 variables

# The constraint that the product of the shape array is less than MAX_SHAPE_SUM
    if len_var is None : len_var = MAX_ARR_LEN
    product_constraint = array_product(shape, len_var) < MAX_SHAPE_SUM
    return product_constraint
