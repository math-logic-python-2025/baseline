# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: predicates/syntax.py

"""Syntactic handling of predicate-logic expressions."""

from __future__ import annotations

from functools import lru_cache
from typing import AbstractSet, Mapping, Optional, Sequence, Set, Tuple, Union

from src.logic_utils import (
    frozen,
    memoized_parameterless_method,
    fresh_variable_name_generator,
)
from src.propositions.syntax import (
    Formula as PropositionalFormula,
)


@lru_cache(maxsize=100)  # Cache the return value of is_binary_prefix
def is_binary_prefix(string: str) -> bool:
    """Checks if the given string is a valid binary-operator prefix.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is one of "&", "|", "->", "-", ``False`` otherwise.
    """
    return string in ("&", "|", "->", "-")


class ForbiddenVariableError(Exception):
    """Raised by `Term.substitute` and `Formula.substitute` when a substituted
    term contains a variable name that is forbidden in that context.

    Attributes:
        variable_name (`str`): the variable name that was forbidden in the
            context in which a term containing it was to be substituted.
    """

    variable_name: str

    def __init__(self, variable_name: str):
        """Initializes a `ForbiddenVariableError` from the offending variable
        name.

        Parameters:
            variable_name: variable name that is forbidden in the context in
                which a term containing it is to be substituted.
        """
        assert is_variable(variable_name)
        self.variable_name = variable_name


@lru_cache(maxsize=100)  # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant name, ``False`` otherwise.
    """
    return (
            ((string[0] >= "0" and string[0] <= "9") or (string[0] >= "a" and string[0] <= "e")) and string.isalnum()
    ) or string == "_"


@lru_cache(maxsize=100)  # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string and string[0] >= "u" and string[0] <= "z" and string.isalnum()


@lru_cache(maxsize=100)  # Cache the return value of is_function
def is_function(string: str) -> bool:
    """Checks if the given string is a function name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a function name, ``False`` otherwise.
    """
    return string and string[0] >= "f" and string[0] <= "t" and string.isalnum()


@frozen
class Term:
    """An immutable predicate-logic term in tree representation, composed from
    variable names and constant names, and function names applied to them.

    Attributes:
        root (`str`): the constant name, variable name, or function name at the
            root of the term tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments of the root, if the root is a function name.
    """

    root: str
    arguments: Optional[Tuple[Term, ...]]

    def __init__(self, root: str, arguments: Optional[Sequence[Term]] = None):
        """Initializes a `Term` from its root and root arguments.

        Parameters:
            root: the root for the formula tree.
            arguments: the arguments for the root, if the root is a function
                name.
        """
        if is_constant(root) or is_variable(root):
            assert arguments is None, \
                "Constants and variables cannot have arguments."
            self.root = root
            self.arguments = None
        else:
            assert is_function(root), f"Invalid function name: {root}"
            assert arguments is not None and len(arguments) > 0, \
                "Function must have non-empty arguments list."
            self.root = root
            self.arguments = tuple(arguments)

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current term.

        Returns:
            The standard string representation of the current term.
        """
        # Task 7.1
        if is_constant(self.root) or is_variable(self.root):
            return self.root

        if is_function(self.root) and self.arguments:
            return f"{self.root}({','.join(map(str, self.arguments))})"

        raise ValueError(f"Invalid expression node: {self.root}")

    def __eq__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Term` object that equals the
            current term, ``False`` otherwise.
        """
        return isinstance(other, Term) and repr(self) == repr(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Term` object or does not
            equal the current term, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(repr(self))

    @staticmethod
    def _read_name_prefix(text: str) -> Tuple[str, str]:
        """Extracts the longest prefix of `text` that is a valid name
        (constant, variable, or function), returning (name, remainder)."""
        end = 0
        while end < len(text) and (
                is_function(text[: end + 1])
                or is_constant(text[: end + 1])
                or is_variable(text[: end + 1])
        ):
            end += 1
        assert end > 0, f"No valid name at prefix of: {text}"
        return text[:end], text[end:]

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Term, str]:
        """Parses a prefix of the given string into a term.

        Parameters:
            string: string to parse, which has a prefix that is a valid
                representation of a term.

        Returns:
            A pair of the parsed term and the unparsed suffix of the string. If
            the given string has as a prefix a constant name (e.g., ``'c12'``)
            or a variable name (e.g., ``'x12'``), then the parsed prefix will be
            that entire name (and not just a part of it, such as ``'x1'``).
        """
        # Task 7.3a
        name, rest = Term._read_name_prefix(string)

        if is_constant(name) or is_variable(name):
            return Term(name), rest

        # At this point, name must be a function
        assert is_function(name) and rest and rest[0] == "(", "Malformed function application."
        rest = rest[1:]  # skip '('
        arguments: list[Term] = []
        while True:
            arg, rest = Term._parse_prefix(rest)
            arguments.append(arg)
            if rest[0] == ")":
                break
            assert rest[0] == ",", "Expected ',' between function arguments."
            rest = rest[1:]  # skip ','
        # Skip closing ')'
        return Term(name, arguments), rest[1:]

    @staticmethod
    def parse(string: str) -> Term:
        """Parses the given valid string representation into a term.

        Parameters:
            string: string to parse.

        Returns:
            A term whose standard string representation is the given string.
        """
        # Task 7.3b
        term, remainder = Term._parse_prefix(string)
        assert remainder == "", f"Unexpected trailing characters in term: {remainder}"
        return term

    def constants(self) -> Set[str]:
        """Finds all constant names in the current term.

        Returns:
            A set of all constant names used in the current term.
        """
        # Task 7.5a
        if is_constant(self.root):
            return {self.root}

        if is_variable(self.root):
            return set()

        # Function application
        return {c for arg in self.arguments for c in arg.constants()}

    def variables(self) -> Set[str]:
        """Finds all variable names in the current term.

        Returns:
            A set of all variable names used in the current term.
        """
        # Task 7.5b
        if is_constant(self.root):
            return set()

        if is_variable(self.root):
            return {self.root}

        # Function application
        return {v for arg in self.arguments for v in arg.variables()}

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current term, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current term.
        """
        # Task 7.5c
        if is_constant(self.root) or is_variable(self.root):
            return set()

        # Function application
        func_set = {(self.root, len(self.arguments))}
        for arg in self.arguments:
            func_set |= arg.functions()
        return func_set

    def substitute(
            self,
            substitution_map: Mapping[str, Term],
            forbidden_variables: AbstractSet[str] = frozenset(),
    ) -> Term:
        """Substitutes in the current term, each constant name `construct` or
        variable name `construct` that is a key in `substitution_map` with the
        term `substitution_map`\ ``[``\ `construct`\ ``]``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variable names not allowed in substitution
                terms.

        Returns:
            The term resulting from performing all substitutions. Only
            constant name and variable name occurrences originating in the
            current term are substituted (i.e., those originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable name from
                `forbidden_variables`.

        Examples:
            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'y'})
            f(c,plus(d,x))

            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,y)')}, {'y'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)

        # Task 9.1
        if is_function(self.root):
            new_args = [arg.substitute(substitution_map, forbidden_variables)
                        for arg in self.arguments]
            return Term(self.root, new_args)
        elif self.root in substitution_map:
            new_term = substitution_map[self.root]
            bad_variables = new_term.variables() & forbidden_variables
            if bad_variables:
                raise ForbiddenVariableError(next(iter(bad_variables)))
            return new_term
        else:
            return self


@lru_cache(maxsize=100)  # Cache the return value of is_equality
def is_equality(string: str) -> bool:
    """Checks if the given string is the equality relation.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is the equality relation, ``False``
        otherwise.
    """
    return string == "="


@lru_cache(maxsize=100)  # Cache the return value of is_relation
def is_relation(string: str) -> bool:
    """Checks if the given string is a relation name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a relation name, ``False`` otherwise.
    """
    return string[0] >= "F" and string[0] <= "T" and string.isalnum()


@lru_cache(maxsize=100)  # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == "~"


@lru_cache(maxsize=100)  # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == "&" or string == "|" or string == "->"


@lru_cache(maxsize=100)  # Cache the return value of is_quantifier
def is_quantifier(string: str) -> bool:
    """Checks if the given string is a quantifier.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a quantifier, ``False`` otherwise.
    """
    return string == "A" or string == "E"


@frozen
class Formula:
    """An immutable predicate-logic formula in tree representation, composed
    from relation names applied to predicate-logic terms, and operators and
    quantifications applied to them.

    Attributes:
        root (`str`): the relation name, equality relation, operator, or
            quantifier at the root of the formula tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments of the root, if the root is a relation name or the
            equality relation.
        first (`~typing.Optional`\\[`Formula`]): the first operand of the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand of the
            root, if the root is a binary operator.
        variable (`~typing.Optional`\\[`str`]): the variable name quantified by
            the root, if the root is a quantification.
        statement (`~typing.Optional`\\[`Formula`]): the statement quantified by
            the root, if the root is a quantification.
    """

    root: str
    arguments: Optional[Tuple[Term, ...]]
    first: Optional[Formula]
    second: Optional[Formula]
    variable: Optional[str]
    statement: Optional[Formula]

    def __init__(
            self,
            root: str,
            arguments_or_first_or_variable: Union[Sequence[Term], Formula, str],
            second_or_statement: Optional[Formula] = None,
    ):
        """Initializes a `Formula` from its root and root arguments, root
        operands, or root quantified variable name and statement.

        Parameters:
            root: the root for the formula tree.
            arguments_or_first_or_variable: the arguments for the root, if the
                root is a relation name or the equality relation; the first
                operand for the root, if the root is a unary or binary operator;
                the variable name to be quantified by the root, if the root is a
                quantification.
            second_or_statement: the second operand for the root, if the root is
                a binary operator; the statement to be quantified by the root,
                if the root is a quantification.
        """
        if is_equality(root) or is_relation(root):
            # Populate self.root and self.arguments
            assert isinstance(arguments_or_first_or_variable, Sequence) and not isinstance(
                arguments_or_first_or_variable, str
            )
            if is_equality(root):
                assert len(arguments_or_first_or_variable) == 2
            assert second_or_statement is None
            self.root, self.arguments = root, tuple(arguments_or_first_or_variable)
            self.first = None
            self.second = None
            self.variable = None
            self.statement = None
        elif is_unary(root):
            # Populate self.first
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is None
            self.root, self.first = root, arguments_or_first_or_variable
            self.arguments = None
            self.second = None
            self.variable = None
            self.statement = None
        elif is_binary(root):
            # Populate self.first and self.second
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is not None
            self.root, self.first, self.second = (
                root,
                arguments_or_first_or_variable,
                second_or_statement,
            )
            self.arguments = None
            self.variable = None
            self.statement = None
        else:
            # Quantifier
            assert is_quantifier(root)
            # Populate self.variable and self.statement
            assert isinstance(arguments_or_first_or_variable, str) and is_variable(arguments_or_first_or_variable)
            assert second_or_statement is not None
            self.root, self.variable, self.statement = (
                root,
                arguments_or_first_or_variable,
                second_or_statement,
            )
            self.arguments = None
            self.first = None
            self.second = None

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        # Task 7.2
        if is_equality(self.root):
            return f"{self.arguments[0]}={self.arguments[1]}"

        if is_relation(self.root):
            return f"{self.root}({','.join(map(str, self.arguments))})"

        if is_unary(self.root):
            return f"{self.root}{self.first}"

        if is_binary(self.root):
            return f"({self.first}{self.root}{self.second})"

        if is_quantifier(self.root):
            return f"{self.root}{self.variable}[{self.statement}]"

        raise ValueError(f"Invalid expression node: {self.root}")

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and repr(self) == repr(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(repr(self))

    @staticmethod
    def _read_relation_prefix(text: str) -> Tuple[str, str]:
        """Extracts the longest prefix of `text` that is a relation, unary,
        or quantifier symbol, returning (prefix, remainder)."""
        end = 0
        while end < len(text) and (
                is_relation(text[: end + 1])
                or is_unary(text[: end + 1])
                or is_quantifier(text[: end + 1])
        ):
            end += 1
        return text[:end], text[end:]

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Formula, str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse, which has a prefix that is a valid
                representation of a formula.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a term followed by an equality
            followed by a constant name (e.g., ``'f(y)=c12'``) or by a variable
            name (e.g., ``'f(y)=x12'``), then the parsed prefix will include
            that entire name (and not just a part of it, such as ``'f(y)=x1'``).
        """
        # Task 7.4a

        # Attempt to read a relation, unary operator, or quantifier prefix
        prefix_symbol, rem = Formula._read_relation_prefix(string)
        if prefix_symbol:
            if is_relation(prefix_symbol):
                assert rem and rem[0] == "(", "Relation must be followed by '('."
                rem = rem[1:]  # skip '('
                args: list[Term] = []
                while rem[0] != ")":
                    arg, rem = Term._parse_prefix(rem)
                    args.append(arg)
                    if rem[0] == ",":
                        rem = rem[1:]
                    else:
                        assert rem[0] == ")", "Expected ')' after relation arguments."
                return Formula(prefix_symbol, args), rem[1:]

            if is_unary(prefix_symbol):
                first_subformula, rem_after = Formula._parse_prefix(rem)
                return Formula(prefix_symbol, first_subformula), rem_after

            # Quantifier
            assert is_quantifier(prefix_symbol)
            quant = prefix_symbol
            var_len = 0
            while var_len < len(rem) and is_variable(rem[: var_len + 1]):
                var_len += 1
            assert var_len > 0 and rem[var_len] == "[", "Quantifier must be followed by a variable and '['."
            variable = rem[:var_len]
            rem_after = rem[var_len + 1:]  # skip variable and '['
            statement_subformula, rem_after = Formula._parse_prefix(rem_after)
            assert rem_after and rem_after[0] == "]", "Quantifier statement must end with ']'."
            return Formula(quant, variable, statement_subformula), rem_after[1:]

        # If not relation/unary/quantifier, must be binary or equality
        if string[0] == "(":
            rem = string[1:]  # skip '('
            first_subformula, rem_after = Formula._parse_prefix(rem)
            op_end = 0
            while op_end < len(rem_after) and is_binary_prefix(rem_after[: op_end + 1]):
                op_end += 1
            assert op_end > 0, "Missing binary operator in parentheses."
            operator = rem_after[:op_end]
            rem_after2 = rem_after[op_end:]
            second_subformula, rem_final = Formula._parse_prefix(rem_after2)
            assert rem_final and rem_final[0] == ")", "Binary formula must end with ')'."
            return Formula(operator, first_subformula, second_subformula), rem_final[1:]

        # Otherwise, it must be an equality without surrounding parentheses
        first_term, rem_term = Term._parse_prefix(string)
        assert rem_term and rem_term[0] == "=", "Equality must contain '='."
        second_term, rem_after = Term._parse_prefix(rem_term[1:])
        return Formula("=", (first_term, second_term)), rem_after

    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        # Task 7.4b
        formula, remainder = Formula._parse_prefix(string)
        assert remainder == "", f"Unexpected trailing characters in formula: {remainder}"
        return formula

    def constants(self) -> Set[str]:
        """Finds all constant names in the current formula.

        Returns:
            A set of all constant names used in the current formula.
        """
        # Task 7.6a
        if is_equality(self.root):
            return self.arguments[0].constants() | self.arguments[1].constants()

        if is_relation(self.root):
            assert self.arguments is not None
            return {c for arg in self.arguments for c in arg.constants()}

        if is_unary(self.root):
            assert self.first is not None
            return self.first.constants()

        if is_binary(self.root):
            assert self.first is not None and self.second is not None
            return self.first.constants() | self.second.constants()

        # Quantifier case
        assert is_quantifier(self.root)
        assert self.statement is not None
        return self.statement.constants()

    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        # Task 7.6b
        if is_equality(self.root):
            return self.arguments[0].variables() | self.arguments[1].variables()

        if is_relation(self.root):
            assert self.arguments is not None
            return {v for arg in self.arguments for v in arg.variables()}

        if is_unary(self.root):
            assert self.first is not None
            return self.first.variables()

        if is_binary(self.root):
            assert self.first is not None and self.second is not None
            return self.first.variables() | self.second.variables()

        assert is_quantifier(self.root) and self.statement is not None
        return {self.variable} | self.statement.variables()

    def free_variables(self) -> Set[str]:
        """Finds all variable names that are free in the current formula.

        Returns:
            A set of every variable name that is used in the current formula not
            only within a scope of a quantification on that variable name.
        """
        # Task 7.6c
        if is_equality(self.root):
            return self.arguments[0].variables() | self.arguments[1].variables()

        if is_relation(self.root):
            assert self.arguments is not None
            return {v for arg in self.arguments for v in arg.variables()}

        if is_unary(self.root):
            assert self.first is not None
            return self.first.free_variables()

        if is_binary(self.root):
            assert self.first is not None and self.second is not None
            return self.first.free_variables() | self.second.free_variables()

        assert is_quantifier(self.root) and self.statement is not None
        return self.statement.free_variables() - {self.variable}

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current formula.
        """
        # Task 7.6d
        if is_equality(self.root):
            return self.arguments[0].functions() | self.arguments[1].functions()

        if is_relation(self.root):
            assert self.arguments is not None
            return {f for arg in self.arguments for f in arg.functions()}

        if is_unary(self.root):
            assert self.first is not None
            return self.first.functions()

        if is_binary(self.root):
            assert self.first is not None and self.second is not None
            return self.first.functions() | self.second.functions()

        assert is_quantifier(self.root) and self.statement is not None
        return self.statement.functions()

    def relations(self) -> Set[Tuple[str, int]]:
        """Finds all relation names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of relation name and arity (number of arguments) for
            all relation names used in the current formula.
        """
        # Task 7.6e
        if is_equality(self.root):
            return set()

        if is_relation(self.root):
            assert self.arguments is not None
            return {(self.root, len(self.arguments))}

        if is_unary(self.root):
            assert self.first is not None
            return self.first.relations()

        if is_binary(self.root):
            assert self.first is not None and self.second is not None
            return self.first.relations() | self.second.relations()

        assert is_quantifier(self.root) and self.statement is not None
        return self.statement.relations()

    def substitute(
            self,
            substitution_map: Mapping[str, Term],
            forbidden_variables: AbstractSet[str] = frozenset(),
    ) -> Formula:
        """Substitutes in the current formula, each constant name `construct` or
        free occurrence of variable name `construct` that is a key in
        `substitution_map` with the term
        `substitution_map`\ ``[``\ `construct`\ ``]``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variable names not allowed in substitution
                terms.

        Returns:
            The formula resulting from performing all substitutions. Only
            constant name and variable name occurrences originating in the
            current formula are substituted (i.e., those originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable name from `forbidden_variables`
                or a variable name occurrence that becomes bound when that term
                is substituted into the current formula.

        Examples:
            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'z'})
            Ay[c=plus(d,x)]

            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,z)')}, {'z'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: z

            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,y)')})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)

        # Task 9.2
        if is_equality(self.root) or is_relation(self.root):
            f = lambda t: t.substitute(substitution_map, forbidden_variables)
            return Formula(self.root, [*map(f, self.arguments)])  # type: ignore
        elif is_unary(self.root):
            return Formula(
                self.root,
                self.first.substitute(substitution_map, forbidden_variables)  # type: ignore
            )
        elif is_binary(self.root):
            return Formula(
                self.root,
                self.first.substitute(substitution_map, forbidden_variables),  # type: ignore
                self.second.substitute(substitution_map, forbidden_variables)  # type: ignore
            )
        else:
            # Quantifier case
            assert is_quantifier(self.root)
            substitution_map_copy = dict(substitution_map)
            if self.variable in substitution_map_copy:
                del substitution_map_copy[self.variable]
            return Formula(
                self.root, self.variable,
                self.statement.substitute(substitution_map_copy, forbidden_variables | {self.variable})  # type: ignore
            )

    def _propositional_skeleton(
            self,
            mappings: dict[str, Formula],
            reverse_mappings: dict[Formula, str]
    ) -> PropositionalFormula:
        if is_unary(self.root):
            new_first = self.first._propositional_skeleton(mappings, reverse_mappings)  # type: ignore
            return PropositionalFormula(self.root, new_first)
        elif is_binary(self.root):
            new_first = self.first._propositional_skeleton(mappings, reverse_mappings)  # type: ignore
            new_second = self.second._propositional_skeleton(mappings, reverse_mappings)  # type: ignore
            return PropositionalFormula(self.root, new_first, new_second)

        # Relation, equality, or quantifier => propositional variable
        if self not in reverse_mappings:
            new_var = next(fresh_variable_name_generator)
            mappings[new_var] = self
            reverse_mappings[self] = new_var
        return PropositionalFormula(reverse_mappings[self])

    def propositional_skeleton(
            self,
    ) -> Tuple[PropositionalFormula, Mapping[str, Formula]]:
        """Computes a propositional skeleton of the current formula.

        Returns:
            A pair. The first element of the pair is a propositional formula
            obtained from the current formula by substituting every (outermost)
            subformula that has a relation name, equality, or quantifier at its
            root with a propositional variable name, consistently such that
            multiple identical such (outermost) subformulas are substituted with
            the same propositional variable name. The propositional variable
            names used for substitution are obtained, from left to right
            (considering their first occurrence), by calling
            `next`\ ``(``\ `~logic_utils.fresh_variable_name_generator`\ ``)``.
            The second element of the pair is a mapping from each propositional
            variable name to the subformula for which it was substituted.

        Examples:
            >>> formula = Formula.parse('((Ax[x=7]&x=7)|(~Q(y)->x=7))')
            >>> formula.propositional_skeleton()
            (((z1&z2)|(~z3->z2)), {'z1': Ax[x=7], 'z2': x=7, 'z3': Q(y)})
            >>> formula.propositional_skeleton()
            (((z4&z5)|(~z6->z5)), {'z4': Ax[x=7], 'z5': x=7, 'z6': Q(y)})
        """
        mappings: dict[str, Formula] = {}
        reverse_mappings: dict[Formula, str] = {}
        skeleton = self._propositional_skeleton(mappings, reverse_mappings)
        return skeleton, mappings

    @staticmethod
    def from_propositional_skeleton(
            skeleton: PropositionalFormula,
            substitution_map: Mapping[str, Formula]
    ) -> Formula:
        """Computes a predicate-logic formula from a propositional skeleton and
        a substitution map.

        Arguments:
            skeleton: propositional skeleton for the formula to compute,
                containing no constants or operators beyond ``'~'``, ``'->'``,
                ``'|'``, and ``'&'``.
            substitution_map: mapping from each propositional variable name of
                the given propositional skeleton to a predicate-logic formula.

        Returns:
            A predicate-logic formula obtained from the given propositional
            skeleton by substituting each propositional variable name with the
            formula mapped to it by the given map.

        Examples:
            >>> Formula.from_propositional_skeleton(
            ...     PropositionalFormula.parse('((z1&z2)|(~z3->z2))'),
            ...     {'z1': Formula.parse('Ax[x=7]'), 'z2': Formula.parse('x=7'),
            ...      'z3': Formula.parse('Q(y)')})
            ((Ax[x=7]&x=7)|(~Q(y)->x=7))

            >>> Formula.from_propositional_skeleton(
            ...     PropositionalFormula.parse('((z9&z2)|(~z3->z2))'),
            ...     {'z2': Formula.parse('x=7'), 'z3': Formula.parse('Q(y)'),
            ...      'z9': Formula.parse('Ax[x=7]')})
            ((Ax[x=7]&x=7)|(~Q(y)->x=7))
        """
        for operator in skeleton.operators():
            assert is_unary(operator) or is_binary(operator)
        for variable in skeleton.variables():
            assert variable in substitution_map

        # Task 9.10
        if is_unary(skeleton.root):
            new_first = Formula.from_propositional_skeleton(skeleton.first, substitution_map)  # type: ignore
            return Formula(skeleton.root, new_first)
        elif is_binary(skeleton.root):
            new_first = Formula.from_propositional_skeleton(skeleton.first, substitution_map)  # type: ignore
            new_second = Formula.from_propositional_skeleton(skeleton.second, substitution_map)  # type: ignore
            return Formula(skeleton.root, new_first, new_second)
        return substitution_map[skeleton.root]
