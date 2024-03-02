import logging
from abc import ABC, abstractmethod


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

class IncompatiableConstrError(Exception):
    """The error is raised when the constraint is not compatiable with the current dtype.
    For example, the constraint is X[0] a float but the dtype is int.    
    """

    pass

class IncorrectConstrError(Exception):
    """
    The error is raised when the constraint is not structurally correct.
    """

    pass

class BaseChecker(ABC):
    @classmethod
    @abstractmethod
    def handler(cls, msg):
        pass

    @classmethod
    def eq(cls, lhs, rhs, msg=""):
        if lhs != rhs:
            cls.handler(f"Failed asertion :: {msg} | {lhs} != {rhs}")

    @classmethod
    def gt(cls, lhs, rhs, msg=""):
        if lhs <= rhs:
            cls.handler(f"Failed asertion :: {msg} | {lhs} <= {rhs}")

    @classmethod
    def ge(cls, lhs, rhs, msg=""):
        if lhs < rhs:
            cls.handler(f"Failed asertion :: {msg} | {lhs} < {rhs}")

    @classmethod
    def lt(cls, lhs, rhs, msg=""):
        if lhs >= rhs:
            cls.handler(f"Failed asertion :: {msg} | {lhs} >= {rhs}")

    @classmethod
    def le(cls, lhs, rhs, msg=""):
        if lhs > rhs:
            cls.handler(f"Failed asertion :: {msg} | {lhs} > {rhs}")

    @classmethod
    def none(cls, obj, msg=""):
        if obj is not None:
            cls.handler(f"Failed asertion :: {msg} | expr is not None")

    @classmethod
    def not_none(cls, obj, msg=""):
        if obj is None:
            cls.handler(f"Failed asertion :: {msg} | expr is None")

    @classmethod
    def true(cls, cond, msg=""):
        if not cond:
            cls.handler(f"Failed asertion :: {msg} | condition is not True")

    @classmethod
    def false(cls, cond, msg=""):
        if cond:
            cls.handler(f"Failed asertion :: {msg} | condition is not False")


class SanityCheck(BaseChecker):
    @classmethod
    def handler(cls, msg):
        logging.critical(msg)
        raise InternalError(
            msg + " | Reporting bugs @ https://github.com/ise-uiuc/nnsmith/issues"
        )


class ConstraintCheck(BaseChecker):
    @classmethod
    def handler(cls, msg):
        raise ConstraintError(msg)
