import argparse
import os
import time
from collections import defaultdict
from copy import deepcopy
from random import randint

from z3 import *

from nnsmith.abstract.arith import *
from nnsmith.autoinf.inference.const import GEN_DIR
from nnsmith.logger import AUTOINF_LOG
from nnsmith.autoinf.inference.utils import str_equivalent, wrap_time

testcases = defaultdict(list)

TreeArgCount = 5


def generate_testcase(argMaxCount: int):
    # one symbol
    for num in range(2, 12):
        testcases[1].append(tuple([num]))
    for num in range(100, 105):
        testcases[1].append(tuple([num]))
    for argCount in range(1, argMaxCount + 1):
        # 3 easy tests + 3 hard tests
        for _ in range(3):
            testcase = []
            for argid in range(argCount):
                testcase.append(randint(2, 11))
            testcases[argCount].append(tuple(testcase))
        for _ in range(3):
            testcase = []
            for argid in range(argCount):
                testcase.append(randint(80, 120))
            testcases[argCount].append(tuple(testcase))
        for i in range(argCount):
            testcase = []
            for argid in range(argCount):
                testcase.append(randint(2, 4))
            testcase[i] = 101
            testcases[argCount].append(tuple(testcase))


class ArithExpNode:
    def __init__(self, type=None, value=None):
        self.type = type
        self.value = value
        self.opNum = 0
        self.cntDiv = 0
        self.ArgSet = 0 if type == "CONSTANT" else (1 << value)
        self.cntMinMax = 0
        if self.type == "CONSTANT":
            self.cntConst = 1
            self.maxarg = -self.value
        else:
            self.cntConst = 0
            self.maxarg = self.value

    def is_simple(self):
        return True

    def display(self):
        if self.type == "CONSTANT":
            return str(self.value)
        else:
            return "s" + str(self.value)

    def evaluate(self, ArgValue) -> int:
        if self.type == "CONSTANT":
            return self.value
        else:
            if self.value >= len(ArgValue):
                raise Exception("No argument value")
            return ArgValue[self.value]

    def discrete_evaluate(self, ArgValue, enableCache) -> int:
        if self.type == "CONSTANT":
            return self.value
        elif f"s{self.value}" in ArgValue:
            return ArgValue[f"s{self.value}"]
        else:
            raise Exception("No argument value")

    def nnsmith_evaluate(self, ArgValue):
        if self.type == "CONSTANT":
            return self.value
        elif f"s{self.value}" in ArgValue:
            return ArgValue[f"s{self.value}"]
        else:
            raise Exception("No argument value")


class ArithExpTree:
    def __init__(self, op=None, lson=None, rson=None, ArgSet=None):
        self.op = op
        self.lson = lson
        self.rson = rson
        self.ArgSet = ArgSet
        self.maxarg = self.lson.maxarg
        self.cntDiv = self.lson.cntDiv + (op == "/") + (op == "%")
        self.opNum = self.lson.opNum + (op != None)
        self.cntConst = self.lson.cntConst
        self.cntMinMax = self.lson.cntMinMax + (op == "max") + (op == "min")
        if self.rson != None:
            self.maxarg = max(self.maxarg, self.rson.maxarg)
            self.cntDiv += self.rson.cntDiv
            self.opNum += self.rson.opNum
            self.cntConst += self.rson.cntConst
            self.cntMinMax += self.rson.cntMinMax
        self.Cache = dict()
        # self.CacheDict = self.CacheVal = None

    def strip(self):
        del self.maxarg
        del self.cntDiv
        del self.opNum
        del self.cntConst
        del self.cntMinMax

    def is_simple(self):
        if self.op is not None and self.op not in ["+", "-", "*"]:
            return False
        if self.lson is not None and not self.lson.is_simple():
            return False
        if self.rson is not None and not self.rson.is_simple():
            return False
        return True

    def display(self):
        if self.op == None:
            return self.lson.display()
        # elif self.op == 'FLOOR':
        # return f'floor({self.lson.display()})'
        # elif self.op == 'CEIL':
        # return f'ceil({self.lson.display()})'
        elif self.op == "min" or self.op == "max":
            return f"{self.op}({self.lson.display()},{self.rson.display()})"
        else:
            return "(" + self.lson.display() + self.op + self.rson.display() + ")"

    def display_nnsmith(self):
        if self.op == None:
            return self.lson.display()
        lval, rval = self.lson.display_nnsmith(), self.rson.display_nnsmith()
        if self.op == "+":
            return f"nnsmith_add({lval}, {rval})"
        elif self.op == "-":
            return f"nnsmith_sub({lval}, {rval})"
        elif self.op == "*":
            return f"nnsmith_mul({lval}, {rval})"
        elif self.op == "/":
            return f"nnsmith_div({lval}, {rval})"
        elif self.op == "%":
            return f"nnsmith_mod({lval}, {rval})"
        elif self.op == "min":
            return f"nnsmith_min({lval}, {rval})"
        elif self.op == "max":
            return f"nnsmith_max({lval}, {rval})"
        else:
            raise Exception("Unknown op")

    def nnsmith_evaluate(self, ArgValue):
        LValue = 0 if self.lson == None else self.lson.nnsmith_evaluate(ArgValue)
        RValue = 0 if self.rson == None else self.rson.nnsmith_evaluate(ArgValue)
        if self.op == None:
            return LValue
        elif self.op == "+":
            return nnsmith_add(LValue, RValue)
        elif self.op == "-":
            return nnsmith_sub(LValue, RValue)
        elif self.op == "*":
            return nnsmith_mul(LValue, RValue)
        elif self.op == "/":
            return nnsmith_div(LValue, RValue)
        elif self.op == "%":
            return nnsmith_mod(LValue, RValue)
        elif self.op == "min":
            return nnsmith_min(LValue, RValue)
        elif self.op == "max":
            return nnsmith_max(LValue, RValue)
        else:
            raise Exception("Unknown op")

    @staticmethod
    def compute_value(LValue, RValue, op):
        if op == None:
            Value = LValue
        elif LValue == "Nan" or RValue == "Nan":
            Value = "Nan"
        elif op == "+":
            Value = LValue + RValue
        elif op == "-":
            Value = LValue - RValue
        elif op == "*":
            Value = LValue * RValue
        elif op == "/":
            if RValue == 0:
                Value = "Nan"
            else:
                Value = LValue // RValue
        elif op == "%":
            if RValue == 0:
                Value = "Nan"
            else:
                Value = LValue % RValue
        elif op == "min":
            Value = min(LValue, RValue)
        elif op == "max":
            Value = max(LValue, RValue)
        else:
            raise Exception("Unrecognized operator")
        return Value

    def evaluate(self, ArgValue):
        LValue = 0 if self.lson == None else self.lson.evaluate(ArgValue)
        RValue = 0 if self.rson == None else self.rson.evaluate(ArgValue)
        Value = ArithExpTree.compute_value(LValue, RValue, self.op)
        return Value

    def discrete_evaluate(self, ArgValue, enableCache=False):
        # if tuple(ArgValue.values()) == self.CacheDict: return self.CacheVal
        # Recursive evaluation
        if enableCache:
            evaltuple = tuple(ArgValue.values())
            if tuple(ArgValue.values()) in self.Cache:
                return self.Cache[evaltuple]
        if enableCache:
            LArgValue, RArgValue = dict(), dict()
            for argName, val in ArgValue.items():
                argNum = int(argName[1:])
                if self.lson.ArgSet & (1 << argNum):
                    LArgValue[argName] = val
                else:
                    RArgValue[argName] = val
        else:
            LArgValue, RArgValue = ArgValue, ArgValue
        LValue = (
            0
            if self.lson == None
            else self.lson.discrete_evaluate(LArgValue, enableCache=enableCache)
        )
        RValue = (
            0
            if self.rson == None
            else self.rson.discrete_evaluate(RArgValue, enableCache=enableCache)
        )
        Value = ArithExpTree.compute_value(LValue, RValue, self.op)

        if enableCache:
            self.Cache[evaltuple] = Value
        return Value

    def cvec(self):
        arglist = []
        for i in range(TreeArgCount):
            if self.ArgSet & (1 << i):
                arglist.append(i)
        results = []
        for testcase in testcases[len(arglist)]:
            ArgDict = dict()
            for i, val in enumerate(testcase):
                ArgDict[f"s{arglist[i]}"] = val
            results.append(self.discrete_evaluate(ArgDict, True))
        return tuple(results)


class TreeDatabase:
    def __init__(self, from_DB: str, maxHeight: int, argCount: int):
        self.maxHeight = maxHeight
        self.argCount = argCount
        if from_DB != "":
            import pickle

            AUTOINF_LOG.info(f"Loading trees from database {from_DB}")
            timestamp = time.time()
            with open(from_DB, "rb") as f:
                self.DB = pickle.load(f)
            AUTOINF_LOG.info(
                f"Finish loading trees, time elapsed: {round(time.time() - timestamp, 1)}s"
            )
        else:
            from collections import defaultdict

            self.DB = []
            for _ in range(maxHeight + 1):
                self.DB.append(defaultdict(list))

    def Add(self, depth: int, ArgSet: int, Tree: ArithExpTree):
        self.DB[depth][ArgSet].append(Tree)

    def strip(self):
        for depth in range(0, self.maxHeight + 1):
            for ArgSet in range(0, 1 << self.argCount):
                for tree in self.DB[depth][ArgSet]:
                    tree.strip()

    def getTree(self, depth, ArgSet, i) -> ArithExpTree:
        return deepcopy(self.DB[depth][ArgSet][i])
        # return self.DB[depth][ArgSet][i]

    def getTreeSet(self, depth: int, ArgSet):
        return self.DB[depth][ArgSet]

    @staticmethod
    def judge(val1, val2, mode) -> bool:
        if mode == "equal":
            return val1 == val2
        else:
            return val1 != val2

    def genTreeList(self, maxHeight, argCount):
        for depth in range(0, maxHeight + 1):
            for args in range(0, argCount + 1):
                ArgSet = (1 << args) - 1
                for i in range(len(self.DB[depth][ArgSet])):
                    yield (depth, ArgSet, i)

    def filterTreeSet(
        self,
        ArgValue,
        expectedValue: int,
        maxdepth: int,
        TreeSet,
        ArgCount,
        mode="equal",
    ):
        if TreeSet == None:
            res = []
            for depth in range(0, maxdepth + 1):
                for ArgSet in range(0, (1 << ArgCount)):
                    for i, tree in enumerate(self.DB[depth][ArgSet]):
                        if TreeDatabase.judge(
                            tree.evaluate(ArgValue), expectedValue, mode
                        ):
                            res.append((depth, ArgSet, i))
        else:
            res = []
            for (depth, ArgSet, i) in TreeSet:
                tree = self.DB[depth][ArgSet][i]
                if TreeDatabase.judge(tree.evaluate(ArgValue), expectedValue, mode):
                    res.append((depth, ArgSet, i))
        return res

    def filterTreeSetTopOperator(self, TreeSet, op):
        res = []
        for (depth, ArgSet, i) in TreeSet:
            tree = self.DB[depth][ArgSet][i]
            if tree.op == op:
                res.append((depth, ArgSet, i))
        return res

    def filterTreeSetArg(self, TreeSet, argNum):
        res = []
        for (depth, ArgSet, i) in TreeSet:
            tree = self.DB[depth][ArgSet][i]
            if (tree.ArgSet & (1 << argNum)) == 0:
                res.append((depth, ArgSet, i))
        return res

    def Count(self) -> int:
        res = 0
        for depth in range(1, self.maxHeight + 1):
            for ArgSet in range(1 << self.argCount):
                if ArgSet in self.DB[depth]:
                    res += len(self.DB[depth][ArgSet])
        return res

    def Count_depth(self, depth) -> int:
        res = 0
        for ArgSet in range(1 << self.argCount):
            if ArgSet in self.DB[depth]:
                res += len(self.DB[depth][ArgSet])
        return res

    def display_tree(self, depth, ArgSet, i):
        return self.DB[depth][ArgSet][i].display()

    def display_tree_nnsmith(self, depth, ArgSet, i):
        return self.DB[depth][ArgSet][i].display_nnsmith()

    def display(self, depth: int):
        for _ in range(1, depth + 2):
            for tree in self.DB[depth][(1 << _) - 1]:
                print(tree.display())

    def dump(self, DB_Name=None):
        import pickle

        treeMode = ""
        if DB_Name == None:
            DB_Name = f"{GEN_DIR}/tree-{self.argCount}{self.maxHeight}{treeMode}.pkl"
        with open(DB_Name, "wb") as f:
            pickle.dump(self.DB, f)


class CvecDatabase:
    def __init__(self):
        self.DB = defaultdict(list)

    def Add(self, ArgSet, cvec, expr) -> bool:
        for e in self.DB[(ArgSet, cvec)]:
            if str_equivalent(e, expr):
                return False
        if len(self.DB[(ArgSet, cvec)]) == 0:
            self.DB[(ArgSet, cvec)].append(expr)
        return True


cvecDB = CvecDatabase()
TreeDB, TreeDB_simple = None, None


def gen_tree_from_string(expr: str) -> ArithExpTree:
    if expr[0] == "(" and expr[-1] == ")":
        expr = expr[1:-1]
    bracket_count = 0
    # leaf
    if (
        expr.count("+")
        + expr.count("-")
        + expr.count("*")
        + expr.count("/")
        + expr.count("%")
        + expr.count("min")
        + expr.count("max")
        == 0
    ):
        if expr[0] == "s":
            leaf = ArithExpNode(type="ARGUMENT", value=int(expr[1:]))
        else:
            leaf = ArithExpNode(type="CONSTANT", value=int(expr))
        return ArithExpTree(op=None, lson=leaf, rson=None, ArgSet=0)
    # recursion
    for i, ch in enumerate(expr):
        if ch == "(":
            bracket_count += 1
        if ch == ")":
            bracket_count -= 1
        if bracket_count == 0 and ch in ["+", "-", "*", "/", "%"]:
            lexpr, rexpr = expr[:i], expr[i + 1 :]
            return ArithExpTree(
                op=ch,
                lson=gen_tree_from_string(lexpr),
                rson=gen_tree_from_string(rexpr),
                ArgSet=0,
            )
    if expr[:3] == "min" or expr[:3] == "max":
        content_expr = expr[4:-1]
        for i, ch in enumerate(content_expr):
            if ch == "(":
                bracket_count += 1
            if ch == ")":
                bracket_count -= 1
            if bracket_count == 0 and ch == ",":
                lexpr, rexpr = content_expr[:i], content_expr[i + 1 :]
                return ArithExpTree(
                    op=expr[:3],
                    lson=gen_tree_from_string(lexpr),
                    rson=gen_tree_from_string(rexpr),
                    ArgSet=0,
                )
    raise Exception("invalid expr")


def Unique(
    op,
    Ltree: ArithExpTree,
    Rtree: ArithExpTree,
    equiv: bool,
    rarity: bool
    # Empirical: bool = EmpiricalPruning,
    # Equality: bool = EqualityPruning,
    # AggresiveEmpirical: bool = AggresiveEmpiricalPruning,
) -> bool:
    if rarity:
        # The ratio of CONSTANT should < 0.75
        # Example:
        #   arg_0  <=>  (2 * arg_0 + 1) // 2
        opNum = Ltree.opNum + Rtree.opNum + 1
        cntNode = opNum + 1
        cntConst = Ltree.cntConst + Rtree.cntConst
        if 1.0 * cntConst / cntNode >= 0.75:
            return False
        # deep division is forbidden
        # if op == '/' and (Ltree.cntDiv > 0 or Rtree.cntDiv > 0):
        # return False
        # multiple division is forbidden
        # if Ltree.cntDiv + Rtree.cntDiv + (op == '/') > 1:
        # return False
        # CONSTANT shouldn't be connected by operators
        if Ltree.maxarg < 0 and Rtree.maxarg < 0:
            return False
        # Only consider 2* /2 1+ -1
        if op in ["-", "/", "%"] and Ltree.maxarg < 0:
            return False
        if op in ["+", "-"] and (Ltree.maxarg == -2 or Rtree.maxarg == -2):
            return False
        if op in ["*", "/"] and (Ltree.maxarg == -1 or Rtree.maxarg == -1):
            return False
        # %1, %2 are useless
        if op == "%" and Rtree.maxarg < 0:
            return False
        # no multiple min/max
        if Ltree.cntMinMax + Rtree.cntMinMax + (op == "min") + (op == "max") > 1:
            return False
    if equiv:
        # Associativity
        if op in ["+", "-"] and Rtree.op in ["+", "-"]:
            return False
        if op in ["*", "/"] and Rtree.op in ["*", "/"]:
            return False
        # Commutativity
        # level 1
        if op in ["+", "*", "min", "max"] and Ltree.maxarg >= Rtree.maxarg:
            return False
        # level 2
        # it's useful for filtering out:
        #  (arg_1 + arg_3) + arg_2
        #  (arg_1 // 2) * 2
        #  (1 + arg_1) - 1
        #  ...
        if (
            op in ["+", "-"]
            and Ltree.op in ["+", "-"]
            and Ltree.rson.maxarg >= Rtree.maxarg
        ):
            return False
        if (
            op in ["*", "/"]
            and Ltree.op in ["*", "/"]
            and Ltree.rson.maxarg >= Rtree.maxarg
        ):
            return False
        """
        # (1 + _) - 1, (2 * _) // 2
        if op == '-' and Ltree.op == '+' and Ltree.lson.maxarg < 0 and Rtree.maxarg < 0: return False
        if op == '/' and Ltree.op == '*' and Ltree.lson.maxarg < 0 and Rtree.maxarg < 0: return False
        """
    return True


DUMMY_CONST = [1, 2]


def NodeGeneration(ArgCount):
    for const_val in DUMMY_CONST:
        TreeDB.Add(
            0,
            0,
            ArithExpTree(lson=ArithExpNode(type="CONSTANT", value=const_val), ArgSet=0),
        )
    for arg in range(0, ArgCount):
        new_tree = ArithExpTree(
            lson=ArithExpNode(type="SYMBOL", value=arg), ArgSet=(1 << arg)
        )
        cvecDB.Add((1 << arg), new_tree.cvec(), new_tree.display())
        TreeDB.Add(0, (1 << arg), new_tree)


def TreeGeneration(ArgSet, depth: int, equiv: bool, rarity: bool):
    """
    Generate all the arithmetic expression tree and put them into TreeDB.
    Trees with only CONSTANT/SYMBOL are regarded as 0-depth.
    """
    # at least one of the left/right subtree should be with height depth-1
    # each symbol should appear only once in the expression
    """
    # uop
    for subtree in TreeDB.getTreeSet(depth-1, ArgSet):
        if subtree.op == '/':
            TreeDB.Add(depth, ArgSet,
                ArithExpTree(op='CEIL',
                             lson=subtree,
                             ArgSet=ArgSet))
            TreeDB.Add(depth, ArgSet,
                ArithExpTree(op='FLOOR',
                             lson='subtree,
                             ArgSet=ArgSet))
    """
    # bop
    for LArgSet in range(0, ArgSet + 1):
        if (ArgSet & LArgSet) == LArgSet:
            RArgSet = LArgSet ^ ArgSet
            for ldepth in range(0, depth):
                rdepth = depth - 1 - ldepth
                for Ltree in TreeDB.getTreeSet(ldepth, LArgSet):
                    for Rtree in TreeDB.getTreeSet(rdepth, RArgSet):
                        valid_op_set = (
                            ["+", "-", "*", "/", "%", "min", "max"]
                            # if EnableDiv
                            # else ["+", "-", "*"]
                        )
                        for op in valid_op_set:
                            if Unique(op, Ltree, Rtree, equiv, rarity):
                                new_tree = ArithExpTree(
                                    op=op, lson=Ltree, rson=Rtree, ArgSet=ArgSet
                                )
                                # if ValuePruning and (
                                #     not cvecDB.Add(
                                #         ArgSet, new_tree.cvec(), new_tree.display()
                                #     )
                                # ):
                                #     continue
                                TreeDB.Add(depth, ArgSet, new_tree)


def build_tree(depth: int, ArgCount: int, dump_dir: str, equiv: bool, rarity: bool):
    global TreeDB
    TreeDB = TreeDatabase("", depth, ArgCount)
    generate_testcase(5)
    NodeGeneration(ArgCount)
    for dep in range(1, depth + 1):
        for ArgSet in range(0, (1 << ArgCount)):
            TreeGeneration(ArgSet, dep, equiv, rarity)
    TreeDB.strip()
    os.makedirs(dump_dir, exist_ok=True)
    TreeDB.dump(os.path.join(dump_dir, "tree.pkl"))


def replicate_simple_tree(dump_dir: str):
    global TreeDB, TreeDB_simple
    depth = ArgCount = 5
    TreeDB_simple = TreeDatabase("", 5, 5)
    for dep in range(0, depth + 1):
        for ArgSet in range(0, (1 << ArgCount)):
            for tree in TreeDB.DB[dep][ArgSet]:
                if tree.is_simple():
                    TreeDB_simple.Add(dep, ArgSet, deepcopy(tree))
    TreeDB_simple.dump(os.path.join(dump_dir, "tree_simple.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", default=GEN_DIR, type=str)
    parser.add_argument("--equiv", default=True, type=bool)
    parser.add_argument("--rarity", default=True, type=bool)
    args = parser.parse_args()
    build_tree(5, 5, args.dump_dir, args.equiv, args.rarity)
    replicate_simple_tree(args.dump_dir)
