class UnsolverableError(Exception):
    """
    tensor element constraint
    """

    pass
class WrongInferenceError(Exception):
    """The error is raised when the inference is not correct."""

    pass

class InternalError(Exception):
    """Fatal unexpected internal errors in NNSmith that should shut down the program immediately."""

    pass

class ConstraintError(Exception):
    """Expected possible constraint unsat error used in shape transfer function."""

    pass

class IncorrectConstrError(Exception):
    """
    The error is raised when the constraint is not structurally correct.
    """

    pass