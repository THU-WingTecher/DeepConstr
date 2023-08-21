import os
from copy import deepcopy

from neuri.autoinf.inference.const import GEN_DIR
from neuri.autoinf.inference.tree import ArithExpNode, ArithExpTree, TreeDatabase

TreeDB = TreeDatabase(os.path.join(GEN_DIR, "tree.pkl"), 5, 5)
TreeDB_simple = TreeDatabase("", 5, 5)

for dep in range(6):
    for ArgSet in range(1 << 5):
        for tree in TreeDB.getTreeSet(dep, ArgSet):
            # print(tree.display(), tree.is_simple())
            if tree.is_simple():
                TreeDB_simple.Add(dep, ArgSet, deepcopy(tree))

TreeDB_simple.dump(os.path.join(GEN_DIR, "tree_simple.pkl"))
