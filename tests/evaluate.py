# class Evaluator() :
#     def __init__(self, 
#                  target : ErrorMessage,
#                  constr : Constraint,
#                  record : Dict[str, Any],
#                  executor : Executor,
#                  cfg : Dict[str, Any] = None
#                  ) :
        

# evaluator = Evaluator(self.target, constraint, self.record, self.executor, self.cfg)
# evaluator.get_f1(num_of_check=self.cfg['simple_eval_asset'] if skim else self.cfg['eval_asset'])
# scores["precision"] = evaluator.precision
# scores["recall"] = evaluator.recall
# scores["f1_score"] = evaluator.f1_score

# @hydra.main(version_base=None, config_path="../neuri/config/", config_name="main")
# def main(cfg: DictConfig):
#     from neuri.constrinf.smt_funcs import load_z3_const
#     from neuri.abstract.dtype import AbsDType
#     from neuri.abstract.dtype import AbsTensor
#     ## whole constraints testing ##
#     # test_whole_constraints()

#     ## target constr testing ##
#     """
#     Example Usage : test_with_given_constraints
#     arg_names = ['a', 'b','c']
#     dtypes = [
#         [AbsDType.int.to_iter()],
#         [AbsDType.int.to_iter()],
#         [AbsTensor.to_iter()],
#         ]
#     test_constraints = [
#         "all((a[i]>1 and a[i]<4) for i in a[2:])",
#         "c[0].shape[0] == b[0]",
#         'a[-1] > b[-2]'
#     ]
#     test_with_given_constraints(test_constraints, arg_names, dtypes)
#     test_smt(arg_names, [d[0] for d in dtypes], constrs, noise_prob=0.3)
#     """
#     arg_names = ['a', 'b','c']
#     dtypes = [
#         [AbsDType.int.to_iter()],
#         [AbsDType.int.to_iter()],
#         [AbsTensor.to_iter()],
#         ]
#     test_constraints = [
#         "d < len(self.shape) for d in c"
#         # "all(i > len(c) and i < len(c) for i in a)",
#         # "alld > len(c)",
#         # "all((a[i]>1 and a[i]<4) for i in a[2:])",
#         # "c[0].shape[0] == b[0]",
#         # 'a[-1] > b[-2]'
#     ]
#     constrs = test_with_given_constraints(test_constraints, arg_names, dtypes)
#     test_smt(arg_names, [d[0] for d in dtypes], constrs, noise_prob=0.3)


# if __name__ == "__main__" :
#      main()
#     #  (len(mat2.shape) == 2) or ((out.shape == [mat1.shape[0], mat2.shape[1]]) or (len(input) == len(mat1)))
