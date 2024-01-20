import ast
from typing import List, Union, Dict, Any, Callable, Tuple, Optional, get_args
import traceback
import operator as op
from specloader.materalize import materalize_dtypes, materalize_dtype
from logger import AUTOINF_LOG
from neuri.abstract.tensor import AbsTensor
from abstract.dtype import AbsDType, AbsIter, AbsLiteral, DType
import z3
from specloader import RANK_ATTR, TYPE_FUNCS, \
        NOT_SIGN, POS_OPS, NEG_OPS, LEN_ATTR, \
        TYPE_ATTRS, TYPE_ATTR, z3_funcs, gen_len_obj
from neuri.abstract.op import __MAX_RANK__
from specloader.irs import Select, IRcompare, IRexpr, symbolize_idx
from specloader.smt import gen_sufficient_condition
class Ast2z3 : 
    
    def __init__(self, arg_names, _ast, args_type_dict) -> None:
        
        self.arg_names = arg_names
        self.is_sliced= {arg_name : [] for arg_name in arg_names}
        self.must_not_None = {arg_name : False for arg_name in arg_names}
        self.min_len = {arg_name : 0 for arg_name in arg_names}
        self.must_be_seq = {arg_name : False for arg_name in arg_names}
        self.is_subscriptable = {arg_name : False for arg_name in arg_names}
        self.args_types : Dict[str, Union[AbsDType,AbsIter,AbsLiteral,AbsTensor]] = {arg_name : None for arg_name in arg_names}
        self.args_length = {arg_name : None for arg_name in arg_names}
        self.is_rank_rules = {arg_name : False for arg_name in arg_names}
        self.is_suff_cond_need = {arg_name : True for arg_name in arg_names}
        self.args_type_dict = args_type_dict
        self.unsolverable = False
        self.ast = _ast
        self.error = False
        self.type_constraints = None
        self.types_map : Dict[str, List[str, Union[AbsDType, AbsLiteral, AbsTensor]]] = dict()
        self.irs : List[IRcompare] = []
        self.inspect_ast_tree()     
        self.set_err_flag()
        self.constraints = None
        self.inspecting = False

    def set_rank_rule_flag(self, arg_name) : 
        self.is_rank_rules[arg_name] = True
    def is_rank_rule(self, arg_name) :
        return self.is_rank_rules[arg_name] 
    def info(self) -> str :
        info = ""
        info += f"arg_names : {self.arg_names}\n"
        info += f"is_sliced : {self.is_sliced}\n"
        info += f"min_len : {self.min_len}\n"
        info += f"is_subscriptable : {self.is_subscriptable}\n"
        info += f"args_types : {self.args_types}\n"
        info += f"args_length : {self.args_length}\n"
        info += f"unsolverable : {self.unsolverable}\n"
        info += f"args_type_dict : {self.args_type_dict}\n"
        info += f"ast : {ast.dump(self.ast)}\n"
        info += f"error : {self.error}\n"
        info += f"types_map : {self.types_map}\n"
        return info
    def set_args_types(self, args_types : Dict[str, Any]) -> None :
        for arg_name in self.arg_names :
            self.args_types[arg_name] = args_types[arg_name]
    def set_args_length(self, args_length : Dict[str, Any]) -> None :
        for arg_name in self.arg_names :
            if args_length[arg_name] is not None : 
                self.args_length[arg_name] = args_length[arg_name]
    def get_mode(self) : 
        if self.is_converting() :
            return '[convert-mode]'
        else :
            return '[inspect-mode]'
    def set_model_unsolverable_flag(self, switch : bool = True) : 
        AUTOINF_LOG.debug(f"{self.get_mode()} would not solved by z3. types_map : {self.types_map}")
        self.unsolverable = switch
    def set_types_map(self, left : str, 
                      right : Any = None , 
                      sign : Optional[str] = None) : 
        if left not in self.types_map.keys() : 
            self.types_map[left] = []
        if right is not None :
            self.types_map[left].append((sign, right))
    def inspect_ast_tree(self) -> None :
        self.args_types = {arg_name : None for arg_name in self.arg_names}
        self.inspecting = True 
        self.convert(self.ast)
        self.inspecting = False
        if self.is_in_types_map('undefined') and len(self.arg_names) == 1 :
            item = self.types_map['undefined']
            self.set_types_map(self.arg_names[0], item[1], item[0])
            del self.types_map['undefined']
    def _is_suff_cond_need(self) -> None :
        """ 
        we are building suff_cond by the index of array.
        however, for generatorexpr which are a[0]>1 ... a[MAX_LEN]>1, this way of building would be wrong.
        Therefore, for generatorexpr, we disable suff_cond building.
        """
    def gen_constraints(self) -> None : 
        try :
            constraints = []
            if self.is_unsolverable() : 
                return True
            ir = self.convert(self.ast)
            if isinstance(ir, bool) : 
                self.set_model_unsolverable_flag()
                return ir
            if hasattr(ir, 'concrete') :
                z3_equations = ir.concrete(self.args_types)
            else : 
                if hasattr(ir, '__len__') : 
                    for i in range(len(ir)) :
                        ir[i] = ir[i].concrete(self.args_types)[0] if hasattr(ir[i], 'concrete') else z3.And(ir[i])
                z3_equations = z3.And(ir)

            z3_equations = ir.concrete(self.args_types) if hasattr(ir, 'concrete') else z3.And(ir)
            if not hasattr(z3_equations, '__iter__') :
                z3_equations = [z3_equations]
            for z3_equation in z3_equations :
                sufficient_conds = gen_sufficient_condition(z3_equation, self.is_suff_cond_need)
                if sufficient_conds :
                    z3_equation = z3.Implies(z3.And(sufficient_conds), z3_equation)
                # AUTOINF_LOG.debug(f"z3_equation : {z3_equation}, \nsuff_conds : {sufficient_conds}")
                constraints.append(z3_equation)
            # return z3.And(constraints)
            return z3.simplify(z3.And(constraints))
        except :
            AUTOINF_LOG.error(f"{ast.dump(self.ast, indent=3)}\n{self.info()}\n{traceback.format_exc()}")
            self.set_err_flag()
            return None
    def type_rule_behavior(self,
                           left : str, 
                           comparators : List[Union[str, AbsLiteral]],
                           op : ast.Expr,
                           ) -> None :

        ## if rights == Literal , we assume that if dtype is kindof AbsLiteral, it cannot be exist with other dtype
        if any(isinstance(op, neg_op) for neg_op in NEG_OPS) : 
            sign = NOT_SIGN
        elif any(isinstance(op, pos_op) for pos_op in POS_OPS) :
            sign = None
        
        infered_types=[]
        for comparator in comparators :
            if type(comparator) == str :
                materalized = materalize_dtypes(comparator, merge_tensor=False) 
                if materalized is not None :
                    infered_types.extend(materalized)
            else :
                infered_types.append(comparator)
        for dtype in infered_types : 
            self.set_types_map(left, dtype, sign)

    def is_types_map_inited(self) : 
        return len(self.types_map) != 0
    def is_converting(self) : return self.inspecting == False 
    def set_iter_rule_flag(self, arg_name) : 
        self.must_be_seq[arg_name] = True
        self.is_subscriptable[arg_name] = True
    def clean_types_map(self) :
        to_rm=[]
        for key in self.types_map.keys() :
            if len(self.types_map[key]) == 0 :
                to_rm.append(key)
        for key in to_rm :
            del self.types_map[key]
    def is_unsolverable_dtype(self, orig_type, materalized) :
        ## materalized dtype is one of literal or iter(Tuple, list) -> z3 unsolverable ex) type(a) == list[int], or type(a) == Literal['a', 'b']
        ## if orig_type is not tensor, materalized dtype is absdtype -> z3 unsolverable ex) type(a) == int (if tensor -> to_tensor_dtype)
        return any(isinstance(materalized, dtype) for dtype in \
                   [AbsLiteral,
                    AbsIter]) or \
                    (not any(isinstance(orig_dtype, AbsTensor) for orig_dtype in self.args_type_dict[orig_type]) and \
                     any(isinstance(materalized, dtype) for dtype in \
                    [AbsDType]))
    def set_idx_constraints(self, arg_name, sliced) : 
        self.is_sliced[arg_name] = [sliced]
        self.min_len[arg_name] = max(self.min_len[arg_name], sliced+1 if sliced >= 0 else -sliced)
    def is_unsolverable(self) :
        return self.unsolverable
    def is_in_types_map(self, left) :
        return left in self.types_map.keys()
    def is_tensor(self, arg_name : str) :
        # Tensor dtype arg_name would not be changed dtype to int, or list[int], or bool.
        # dtype would be retrieved with args_type_dict that has been given when generating the rule.
        # Therefore, the type_actvated new dtype would not be shown here.
        arg_name = self.get_arg_name(arg_name)
        return isinstance(self.args_types[arg_name], AbsTensor)
        
    def get_arg_name(self, obj) :
        return str(obj)
    def is_generator_exp(self, left) :
        if isinstance(left, Select) :
            return left.has_symbol()
        elif isinstance(left, str) :
            return left in ['i', 'j']
        else :
            return False 
    def is_z3_obj(self, arg_name) :
        if isinstance(arg_name, str) and arg_name in self.arg_names :
            return False 
        else :
            return True 
    def check_tensor_dtype(self, arg_name) :
        if isinstance(self.args_types[arg_name], AbsTensor) :
            self._is_tensor[arg_name] = True
        else :
            self._is_tensor[arg_name] = False
    def materalize_dtype(self, dtype) : 
        if dtype not in self.arg_names : 
            return materalize_dtype(dtype)
        else :
            return dtype 
    def convert(self,
               expr : Union[ast.Expression,List[ast.Expression]]
               ) -> Union[IRcompare, IRexpr, bool] :
        """
        This function do two things : 
            1.collect arg info from given ast (first time executed) -> return bool
            2.gen z3 obj according to given ast -> return z3.ExprRef
        it return "bool" only when collecting arg_info, which only happens at the first time.
        Other times, it should return z3.ExprRef
        
        """
        if isinstance(expr, ast.Expression):
            return self.convert(expr.body)
        if isinstance(expr, ast.Expr):
            return self.convert(expr.value) ## if it is not body?
        elif isinstance(expr, ast.Module):
            return self.convert(expr.body[0])
        elif isinstance(expr, ast.UnaryOp):
            # Handle Boolean operations (e.g., And, Or)
            operand_to_oppo = self.convert(expr.operand)
            if isinstance(operand_to_oppo, str) : 
                return True 
            else :
                return IRexpr(expr.op, [operand_to_oppo])
        elif isinstance(expr, ast.BoolOp):
            # Handle Boolean operations (e.g., And, Or)
            return IRexpr(expr.op, [self.convert(value) for value in expr.values])
        elif isinstance(expr, ast.Tuple) :
            return [self.convert(value) for value in expr.elts]
        elif isinstance(expr, ast.Compare) : ## return left op comparators ex) 'output_size'<0 or 'input_size'
            res=[]
            left=expr.left 
            ops=expr.ops
            astop = ops[0]
            comparators=expr.comparators
            left = self.convert_vars(left) # -> filter out type/len constraints
            rights = self.convert_comparators_to_z3(comparators)
            op = convert_ops(ops)
            if self.is_in_types_map(left) and not self.is_converting() : # type()
                dtype_instances = [] 
                for right in rights :
                    if isinstance(right, str) :
                        right = self.materalize_dtype(right)
                    dtype_instances.append(right)
                for dtype in dtype_instances :
                    if self.is_unsolverable_dtype(left, dtype) :
                        self.type_rule_behavior(left, rights, astop)
                        self.set_model_unsolverable_flag()
                
            if self.is_unsolverable() or not self.is_converting()  : 
                return True 
            if self.is_generator_exp(left) : ## interpret ir in Generator part 
                ir = IRcompare(left, op, rights)
                return ir
            else : ## interpret ir 
                if isinstance(left, Select) and left.attr in TYPE_ATTRS : # comparator dtype interpreting if left attr == dtype 
                    left = left.concrete(self.args_types)
                    z3dtypes = set()
                    for _dtype in rights :
                        # string, or Select obj
                        ## currently only allowed btw tensor dtypes.
                        if isinstance(_dtype, Select) : 
                            z3dtypes.add(_dtype.concrete(self.args_types))
                            continue
                        dtype = self.materalize_dtype(_dtype)
                        if not isinstance(dtype, DType) :
                            ## currently only allowed btw tensor dtypes.
                            if hasattr(dtype, 'to_tensor_dtype') :
                                if isinstance(astop, ast.Eq) and len(dtype.to_tensor_dtype())>1: 
                                    astop = ast.In()
                                    op = convert_ops([astop])
                                temp = dtype.to_tensor_dtype()
                                z3s = [d.z3() for d in temp]
                                z3dtypes.update(z3s)
                            elif hasattr(dtype, 'z3') : 
                                z3dtypes.add(dtype.z3())
                            else :
                                z3dtypes.add(dtype)
                        else :
                            z3dtypes.add(dtype.z3())
                    z3dtypes = list(z3dtypes)
                    if len(z3dtypes) == 1 : z3dtypes = z3dtypes[0]
                    if is_compatiable(astop, left, z3dtypes) :
                        res.append(op(left, z3dtypes))
                    else :
                        if any(isinstance(astop, eq_noteq) for eq_noteq in [ast.Eq, ast.NotEq]) : 
                            for z3dtype in z3dtypes :
                                res.append(convert_ops([astop])(left, z3dtype))
                        else :
                            AUTOINF_LOG.debug(f"{self.get_mode()} Uncompatiable - left : {left}, z3dtypes : {z3dtypes}")
                            return True
                else : # left.attr in [RANK_ATTR, LEN_ATTR] or other cases
                    concreted = []
                    left = left.concrete(self.args_types) if hasattr(left, 'concrete') else left
                    for right in rights :
                        if isinstance(right, Select) : 
                            concreted.append(right.concrete(self.args_types))
                        else :
                            concreted.append(right)
                    if is_compatiable(astop, left, concreted) :
                        res.append(op(left, concreted))
                    elif hasattr(left, '__len__') :
                        if is_compatiable(astop, left[0], concreted) :
                            res.append([op(left[i], concreted) for i in range(len(left))])
                    else :
                        if hasattr(left, 'sort') and left.sort().kind() in [6, 5] : # Z3TENSOR).sort()
                            if len(concreted) == 1 and hasattr(concreted[0], 'sort') and concreted[0].sort().kind() in [6, 5] :
                                res.append(op(left, concreted[0]))
                                res.append(gen_len_obj(left)==gen_len_obj(concreted[0]))
                            else :
                                len_obj = gen_len_obj(left)
                                res.append(len_obj==len(concreted))
                                r = [op(left[i], concreted[i]) for i in range(len(concreted))]
                                res.extend(r)
                        else :
                            for con in concreted : 
                                if is_compatiable(astop, left, con) :
                                    res.append(op(left, con))
            
            return IRexpr(ast.And(), [res])
        elif isinstance(expr, ast.Call):
            # Handle function calls
            converted = self.convert_vars(expr)
            if isinstance(converted, str) : return True
            else : return converted
        elif isinstance(expr, ast.comprehension):
            target = self.convert_vars(expr.target)
            iters = self.convert(expr.iter)
            # if expr.iter.func.id == 'range' :
            if len(expr.ifs) > 0 : raise(NotImplementedError)
            return (target, iters) 

        else:
            return self.convert_vars(expr)
    

    def convert_vars(self, expr : ast.Expression) -> Union[str, z3.ExprRef, bool] : 
        """
        return (arg_name, func_name, slices )
        expr : ast.Expression which should contain the information of expr 
        types : Dict[str, Any] which contains the information of type of each variable(var name should be same with left arg_name)
        """

        if isinstance(expr, ast.Name) :
            if self.is_converting() :
                if expr.id in self.arg_names :
                    return Select(expr.id)
            return expr.id
        elif isinstance(expr, ast.Constant) :
            # Handle constant values
            if self.is_converting() :
                if expr.value in self.arg_names :
                    return Select(expr.value)
            return expr.value 
        elif isinstance(expr, ast.UnaryOp) :
            if isinstance(expr.op, ast.USub) :
                left = self.convert_vars(expr.operand)
                if self.is_converting() :
                    if isinstance(left, Select) :
                        return - left.concrete(self.args_types)
                    elif isinstance(left, str) : # 'i' symbol
                        return (ast.Mult(), left, -1)
                    else :
                        return -left 
                else :
                    return left
        elif isinstance(expr, ast.Tuple) :
            return self.convert(expr)
        elif isinstance(expr, ast.Subscript) :
            arg_name = self.convert_vars(expr.value)
            sliced = self.convert_vars(expr.slice)
            if self.is_converting() :
                if isinstance(sliced, int) :
                    arg_name.set_idx(sliced)
                elif isinstance(sliced, tuple) : 
                    astop, left, right = sliced 
                    arg_name.set_idx(left)
                    arg_name.set_binops((get_operator(astop), right))
                else : # str (plain 'i') or Select
                    arg_name.set_idx(sliced)
                return arg_name
            else : # inspecting 
                if isinstance(arg_name, str) and arg_name.lower() == 'literal' :
                    self.check_types_map(create=True)
                    if type(sliced) == str : # 'same' -> ['same']
                        sliced = [sliced]
                    type_name = AbsLiteral(sliced)
                    return type_name
                elif isinstance(arg_name, str) and arg_name.lower() == 'list' :
                    self.check_types_map(create=True)
                    return arg_name.lower() + '[' + str(sliced) + ']'
                else :
                    if arg_name in self.arg_names : #just for inspecting.
                        return arg_name
                    else :
                        AUTOINF_LOG.info(f"{self.get_mode()} Unsupported subscript of {arg_name}")
                        return arg_name      
        elif isinstance(expr, ast.Attribute) :

            arg_name = self.convert_vars(expr.value)
            if self.is_converting() :
                if hasattr(arg_name, 'set_attr') :
                    arg_name.set_attr(expr.attr)
                    return arg_name
                else :
                    return arg_name + '.' + expr.attr
                    # raise Exception(f"{self.get_mode()} wrong rule generated, {arg_name} not in arg_names")
            else :
                if expr.attr in TYPE_ATTRS :
                    self.set_types_map(arg_name)
                return arg_name   
        elif isinstance(expr, ast.Call) :
            if isinstance(expr.func, ast.Attribute) :
                return self.convert_vars(expr.func)
            call = expr 
            if call.func.id in TYPE_FUNCS :
                arg_names = [self.convert_vars(arg) for arg in expr.args]
                assert len(arg_names) == 1 ## we restrict to only accept type() of dtype rule
                arg_name = arg_names[0]  
                if self.is_converting() :
                    ## arg_name should be the instance of Select 
                    arg_name.set_attr(TYPE_ATTR)
                    return arg_name
                else :
                    ## some checking if needed 
                    self.set_types_map(arg_name)
                    return arg_name  
            elif call.func.id == 'range' : # -> (inspecting)arg_name, (converting)Int
                # -> strongly interleave with generaterexp,
                # it should return list in conventional, for simplicity, we only return int here.
                range_args = [] 
                for arg in expr.args : # will return args [start,end,step]
                    range_args.append(self.convert_vars(arg))
 
                if self.is_converting() :
                    return range_args
                else :
                    return self.get_arg_name(range_args)
            elif hasattr(z3_funcs, call.func.id) :
                args = []
                for arg in expr.args : 
                    args.append(self.convert_vars(arg))
                
                if self.is_converting() :
                    if len(args) == 1 : args = args[0]
                    func = getattr(z3_funcs, call.func.id)
                    if isinstance(args, Select) :
                        args.set_func(func)
                        return args
                    else : 
                        for i in range(len(args)) :
                            if isinstance(args[i], Select) :
                                args[i] = args[i].concrete(self.args_types)
                        return func(args)
                else : # inspecting 
                    return self.get_arg_name(args)

            elif call.func.id in ['all', 'any'] :
                if self.is_converting() : 
                    constraints = self.convert_vars(expr.args[0])
                    if call.func.id == 'all' :
                        constraints = IRexpr(ast.And(), constraints)
                    elif call.func.id == 'any' :
                        constraints = IRexpr(ast.Or(), constraints)
                    return constraints
                else : # inspecting
                    return True 
            elif call.func.id in [LEN_ATTR, RANK_ATTR] :
                    arg_name = self.convert_vars(expr.args[0]) #FIXME : it should be 
                    if self.is_converting() : 
                        arg_name.set_attr(LEN_ATTR)
                        return arg_name
                    else : 
                        self.set_iter_rule_flag(arg_name)
                        return arg_name
            else :
                AUTOINF_LOG.error(f"{self.get_mode()} Unsupported function {call.func.id}")
                self.set_err_flag()

        elif isinstance(expr, ast.GeneratorExp) or isinstance(expr, ast.ListComp) :
            ## interpret generators -> idx = symbol , name cannot change 
            if not self.is_converting() : return True 
            syms = []
            idx_constraints = []
            start, end, step = 0, __MAX_RANK__, 1 
            irexpr = self.convert(expr.elt)
            if len(expr.generators) > 1 : raise NotImplementedError(f"{self.get_mode()} Not implemented yet, multiple generators")
            idx_nm, iters = self.convert(expr.generators[0])
            irexpr.find_sym(idx_nm)
            sym_idx = symbolize_idx(idx_nm)
            syms.append(sym_idx)
            if isinstance(iters, Select) : # all(i>0 for i in input.shape)
                conc = iters.concrete(self.args_types)
                length = iters.export_len_var(self.args_types)
                new_sym_idx = symbolize_idx(iters.name+'_i')
                syms.append(new_sym_idx)
                idx_constraints.extend([new_sym_idx>=0, new_sym_idx<length] )
                idx_constraints.append(sym_idx == conc[new_sym_idx])
            else :
                if len(iters) == 1 : #only have end / all(input.shape[i]>0 for i in range(len(input.shape)))
                    if isinstance(iters[0], Select) : # if list -> ?
                        if iters[0].attr in [LEN_ATTR, RANK_ATTR] :
                            end = iters[0].concrete(self.args_types)
                    else : # int
                        end = iters[0]

                elif len(iters) == 2 : # range -> start, end 
                    start = iters[0]
                    end = iters[1]

                else :  # range -> start, end, step 
                    step = iters[2]
                    assert(isinstance(step, int) and abs(step) == 1)
                    if step < 0 : #range(a,b,-1) -> range(b-1, a-1, 1)
                        start = iters[1] + 1
                        end = iters[0] + 1
                    else :
                        start = iters[0]
                        end = iters[1]

                idx_constraints.append(sym_idx<end)
                idx_constraints.append(sym_idx>=start)
                
            self.mark_suff_cond_needed(self.locate_symbol(irexpr))
            irexpr.assign(sym_idx)
            generator_expr = irexpr.concrete(self.args_types, sym_idx) 
            combined = [z3.ForAll(syms, z3.Implies(z3.And(idx_constraints), z3.And(generator_expr)))]
            return combined
        
        elif isinstance(expr, ast.BinOp) :
            if self.is_converting() :
                left = self.convert_vars(expr.left)
                binop = get_operator(expr.op)
                rights = self.convert_comparators_to_z3([expr.right])
                right = rights[0]
                if isinstance(left, str) : # 'i + 1' 
                    return (expr.op, left, right)
                if hasattr(left, 'concrete') : left = left.concrete(self.args_types)
                if hasattr(right, 'concrete') : right = right.concrete(self.args_types)
                return binop(left, right)
            else : # inspecting 
                return True 
        else : 
            self.set_err_flag()
            AUTOINF_LOG.error(f"{self.get_mode()} Unsupported generator expression {expr}")
    def get_arg_name(self, args : List[Any]) :
        for arg_name in args :
            if arg_name in self.arg_names :
                return arg_name            
    def mark_suff_cond_needed(self, sym_pos : List[Tuple[Select, str]]) : 
        for pos in sym_pos : 
            if pos[1] == 'idx' : 
                self.is_suff_cond_need[pos[0].name] = False
            else :
                pass 

    def locate_symbol(self, irexprs) :
        pos = []
        if isinstance(irexprs, IRcompare) :
            pos.extend(irexprs.whereis_symbols)
        elif isinstance(irexprs, IRexpr) :
            for irexpr in irexprs.values :
                pos.extend(self.locate_symbol(irexpr))
        return pos

    def check_types_map(self, create : bool = False) :
        res = True  
        if not self.is_types_map_inited() :
            AUTOINF_LOG.warning(f"{self.get_mode()} unconsidered behavior, generated by obvious unsolverable rule.")
            res = False 
        if create : 
            arg_name = self.arg_names[0]
            self.set_types_map(arg_name)
        return res 
    def set_idx(self, idx, iters) : 
        if type(iters) == int :
            return range(iters) 
        else : 
            return iters 
    def set_err_flag(self) : 
        self.error = True
    def convert_comparators_to_z3(self, comparators : List[ast.Expression]) -> List[ast.Expression] : 
  
        converted_comparators=[]
        if len(comparators) > 1 :
            for comparator in comparators :
                converted_comparators.extend(self.convert_comparators_to_z3([comparator]))
            return converted_comparators
        elif len(comparators) < 1 :
            return []
        else : 
            comparators = comparators[0]
            if isinstance(comparators, ast.Tuple) or isinstance(comparators, ast.List) :
                converted_comparators.extend(self.convert_comparators_to_z3(comparators.elts))
                return converted_comparators
            else : return [self.convert_vars(comparators)]

    def gen_z3_obj(self, arg_name) -> z3.ExprRef :
        
        return gen_z3_obj(arg_name,
                          self.args_types[arg_name],
                            )


def gen_z3_obj(arg_name,
               arg_type : Union[AbsDType, AbsLiteral, AbsTensor, AbsLiteral],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
) -> Optional[z3.ExprRef] :
    z3_obj = None
    if arg_type in [AbsDType.none, AbsDType.str] :
        pass
    elif hasattr(arg_type, 'z3') : 
        z3_obj = arg_type.z3()(arg_name)
    elif isinstance(arg_type, AbsTensor) : 
        z3_obj = arg_type.z3()(arg_name)
    else :
        raise NotImplementedError(f"Unsupported {arg_name} type {arg_type}")
    return z3_obj

def is_compatiable(op : ast, a, b) -> bool :
    if not hasattr(a, '__iter__') and (not isinstance(b, str) and hasattr(b, '__iter__')) :
        return any(isinstance(op, allowed) for allowed in [ast.In, ast.NotIn])
    elif not hasattr(a, '__iter__') and (isinstance(b, str) or not hasattr(b, '__iter__')) :
        return any(isinstance(op, allowed) for allowed in [
                                                        ast.Eq,
                                                        ast.NotEq,
                                                        ast.Lt,
                                                        ast.LtE,
                                                        ast.Gt,
                                                        ast.GtE,
                                                        ast.Is,
                                                        ])
  
    else :
        return False 
def convert_ops(ops: List[ast.Expression]) -> Callable : # operator of ast 

    if len(ops) > 1 :
        # for op in ops :
        #     converted_ops.extend(op)
        # return converted_ops
        raise NotImplementedError(f"Unsupported multiple ops {' '.join([type(op) for op in ops])}")
    elif len(ops) < 1 :
        raise NotImplementedError(f"Unsupported empty ops")
    else :
        ops = ops[0]
        if isinstance(ops, ast.Eq):
            return lambda a,b : op.eq(a, b)
        elif isinstance(ops, ast.NotEq):
            return lambda a,b : op.ne(a, b)
        elif isinstance(ops, ast.Lt):
            return lambda a,b : op.lt(a, b)
        elif isinstance(ops, ast.LtE):
            return lambda a,b : op.le(a, b)
        elif isinstance(ops, ast.Gt):
            return lambda a,b : op.gt(a, b)
        elif isinstance(ops, ast.GtE):
            return lambda a,b : op.ge(a, b)
        elif isinstance(ops, ast.Is):
            return lambda a,b : op.eq(a, b)
        elif isinstance(ops, ast.IsNot):
            return lambda a,b : op.ne(a, b)
        elif isinstance(ops, ast.In):
            return lambda a,b : z3_funcs.in_(a,b)
        elif isinstance(ops, ast.NotIn):
            return lambda a,b : z3_funcs.not_in(a,b)
        else:
            raise ValueError(f"Unknown comparison operation: {op}")

def get_operator(operator):
    if isinstance(operator, ast.Add):
        return lambda a,b : op.add(a, b)
    elif isinstance(operator, ast.Sub):
        return lambda a,b : op.sub(a, b)
    elif isinstance(operator, ast.Mult):
        return lambda a,b : op.mul(a, b)
    elif isinstance(operator, ast.MatMult):
        return lambda a,b : op.matmul(a, b)
    elif isinstance(operator, ast.Div):
        return lambda a,b : op.truediv(a, b)
    elif isinstance(operator, ast.FloorDiv):
        return lambda a,b : op.truediv(a, b)
    elif isinstance(operator, ast.Mod):
        return lambda a,b : op.mod(a, b)
    elif isinstance(operator, ast.Pow):
        return lambda a,b : op.pow(a, b)
    # elif isinstance(operator, ast.LShift):
    #     return op.lshift
    # elif isinstance(operator, ast.RShift):
    #     return op.rshift
    # elif isinstance(operator, ast.BitOr):
    #     return op.or_
    # elif isinstance(operator, ast.BitXor):
    #     return op.xor
    # elif isinstance(operator, ast.BitAnd):
    #     return op.and_
    # elif isinstance(operator, ast.FloorDiv): ## FIXME : z3 does not support floor div
    #     return op.truediv 
    else:
        raise ValueError(f"Unknown operator: {operator}")

def convert_boolop_to_z3(op : ast.BoolOp) -> z3.ExprRef :
    if isinstance(op, ast.Or):
        return z3.Or
    elif isinstance(op, ast.Not):
        return z3.Not
    elif isinstance(op, ast.And):
        return z3.And
    else :
        raise NotImplementedError(f"Unsupported boolop type {type(op)}")   

def flatten_list(nested_list : List[Any]) -> List[Any] :
    flattened_list = []
    [flattened_list.extend(sublist) for sublist in nested_list]
    return flattened_list
