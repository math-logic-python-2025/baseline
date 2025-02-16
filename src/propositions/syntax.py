# File: propositions/syntax.py
"""Syntactic handling of propositional formulas."""

from __future__ import annotations
from collections.abc import Callable
from functools import lru_cache
from typing import Mapping, Optional, Set, Tuple, Union

from src.logic_utils import frozen, memoized_parameterless_method


@lru_cache(maxsize=100)
def is_variable(string: str) -> bool:
    return string[0] >= "p" and string[0] <= "z" and (len(string) == 1 or string[1:].isdecimal())


@lru_cache(maxsize=100)
def is_constant(string: str) -> bool:
    return string == "T" or string == "F"


@lru_cache(maxsize=100)
def is_unary(string: str) -> bool:
    return string == "~"


@lru_cache(maxsize=100)
def is_binary(string: str) -> bool:
    return string in {"&", "|", "->", "+", "<->", "-&", "-|"}


@frozen
class Formula:
    """An immutable propositional formula in tree representation, composed from
    variable names, and operators applied to them.

    Attributes:
        root (str): the constant, variable name, or operator at the root of the formula tree.
        first (Optional[Formula]): the first operand of the root, if the root is a unary or binary operator.
        second (Optional[Formula]): the second operand of the root, if the root is a binary operator.
    """

    root: str
    first: Optional[Formula]
    second: Optional[Formula]

    def __init__(self, root: str, first: Optional[Formula] = None, second: Optional[Formula] = None):
        if is_variable(root) or is_constant(root):
            assert first is None and second is None
            self.root = root
        elif is_unary(root):
            assert first is not None and second is None
            self.root, self.first = root, first
        else:
            assert is_binary(root)
            assert first is not None and second is not None
            self.root, self.first, self.second = root, first, second

    @memoized_parameterless_method
    def __repr__(self) -> str:
        if is_variable(self.root):
            return self.root
        elif is_constant(self.root):
            return self.root
        elif is_unary(self.root):
            return self.root + str(self.first)
        return "(" + str(self.first) + self.root + str(self.second) + ")"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @memoized_parameterless_method
    def variables(self) -> Set[str]:
        if is_variable(self.root):
            return {self.root}
        elif is_constant(self.root):
            return set()
        vars_set = self.first.variables()
        if is_unary(self.root):
            return vars_set
        return vars_set.union(self.second.variables())

    @memoized_parameterless_method
    def operators(self) -> Set[str]:
        if is_variable(self.root):
            return set()
        elif is_constant(self.root):
            return {self.root}
        if is_unary(self.root):
            return {self.root}.union(self.first.operators())
        return {self.root}.union(self.first.operators()).union(self.second.operators())

    @staticmethod
    def __is_char_like(string: str, index: int, func: Callable[[str], bool]) -> bool:
        if len(string) <= index:
            return False
        return func(string[index])

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Union[Formula, None], str]:
        if not string:
            return None, "Empty string"
        if is_variable(string[0]):
            i = 1
            while i < len(string) and string[i].isdigit():
                i += 1
            var = string[:i]
            return Formula(var), string[i:]
        if is_constant(string[0]):
            return Formula(string[0]), string[1:]
        if is_unary(string[0]):
            operand, remaining = Formula._parse_prefix(string[1:])
            if operand is None:
                return None, "Expected operand after unary operator '~'"
            return Formula(string[0], operand), remaining
        if string[0] == "(":
            first, remaining = Formula._parse_prefix(string[1:])
            if first is None:
                return None, "Expected first operand after '('"
            bin_ops = ["<->", "->", "-&", "-|", "+", "&", "|"]
            operator = None
            for op in bin_ops:
                if remaining.startswith(op):
                    operator = op
                    remaining = remaining[len(op) :]
                    break
            if operator is None:
                return None, f"Expected binary operator after first operand, found: '{remaining[:3]}'"
            second, remaining = Formula._parse_prefix(remaining)
            if second is None:
                return None, "Expected second operand after operator"
            if not remaining.startswith(")"):
                return None, "Expected ')' after second operand"
            remaining = remaining[1:]
            return Formula(operator, first, second), remaining
        return None, f"Unexpected character: '{string[0]}'"

    @staticmethod
    def is_formula(string: str) -> bool:
        formula, remaining = Formula._parse_prefix(string)
        return formula is not None and remaining == ""

    @staticmethod
    def parse(string: str) -> Formula:
        assert Formula.is_formula(string)
        formula, _ = Formula._parse_prefix(string)
        return formula

    def polish(self) -> str:
        pass

    @staticmethod
    def parse_polish(string: str) -> Formula:
        pass

    def substitute_variables(self, substitution_map: Mapping[str, Formula]) -> Formula:
        for variable in substitution_map:
            assert is_variable(variable)
        if is_variable(self.root):
            return substitution_map[self.root] if self.root in substitution_map else self
        elif is_constant(self.root):
            return self
        elif is_unary(self.root):
            return Formula(self.root, self.first.substitute_variables(substitution_map))
        else:
            return Formula(
                self.root,
                self.first.substitute_variables(substitution_map),
                self.second.substitute_variables(substitution_map),
            )

    def substitute_operators(self, substitution_map: Mapping[str, Formula]) -> Formula:
        for operator in substitution_map:
            assert is_constant(operator) or is_unary(operator) or is_binary(operator)
            assert substitution_map[operator].variables().issubset({"p", "q"})
        if is_variable(self.root) or is_constant(self.root):
            new_self = self
        elif is_unary(self.root):
            new_self = Formula(self.root, self.first.substitute_operators(substitution_map))
        else:
            new_self = Formula(
                self.root,
                self.first.substitute_operators(substitution_map),
                self.second.substitute_operators(substitution_map),
            )
        if new_self.root in substitution_map:
            sub_map = {}
            if hasattr(new_self, "first") and new_self.first is not None:
                sub_map["p"] = new_self.first
            if hasattr(new_self, "second") and new_self.second is not None:
                sub_map["q"] = new_self.second
            return substitution_map[new_self.root].substitute_variables(sub_map)
        else:
            return new_self
