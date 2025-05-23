# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: predicates/functions.py

"""Syntactic conversion of predicate-logic formulas to not use functions and
equality."""

from typing import List, Literal, Callable

from src.logic_utils import is_z_and_number, fresh_variable_name_generator
from src.predicates.semantics import *


def function_name_to_relation_name(function: str) -> str:
    """Converts the given function name to a canonically corresponding relation
    name.

    Parameters:
        function: function name to convert.

    Returns:
        A relation name that is the same as the given function name, except that
        its first letter is capitalized.
    """
    assert is_function(function)
    return function[0].upper() + function[1:]


def relation_name_to_function_name(relation: str) -> str:
    """Converts the given relation name to a canonically corresponding function
    name.

    Parameters:
        relation: relation name to convert.

    Returns:
        A function name `function` such that
        `function_name_to_relation_name`\ ``(``\ `function`\ ``)`` is the given
        relation name.
    """
    assert is_relation(relation)
    return relation[0].lower() + relation[1:]


def replace_functions_with_relations_in_model(model: Model[T]) -> Model[T]:
    """Converts the given model to a canonically corresponding model without any
    function interpretations, replacing each function interpretation with a
    canonically corresponding relation interpretation.

    Parameters:
        model: model to convert, such that there exist no canonically
            corresponding function name and relation name that both have
            interpretations in this model.

    Returns:
        A model obtained from the given model by replacing every function
        interpretation of a function name with a relation interpretation of the
        canonically corresponding relation name, such that the relation
        interpretation contains any tuple
        ``(``\ `x1`\ ``,``...\ ``,``\ `xn`\ ``)``  if and only if `x1` is the
        output of the function interpretation for the arguments
        ``(``\ `x2`\ ``,``...\ ``,``\ `xn`\ ``)``.
    """
    for function in model.function_interpretations:
        assert function_name_to_relation_name(function) not in model.relation_interpretations

    # Task 8.1
    new_relations = dict(model.relation_interpretations)
    for name, function in model.function_interpretations.items():
        new_name = function_name_to_relation_name(name)
        new_relations[new_name] = set()
        for args, value in function.items():
            new_relations[new_name].add((value, *args))

    return Model(
        model.universe,
        model.constant_interpretations,
        new_relations,
        {}
    )


def replace_relations_with_functions_in_model(
        model: Model[T], original_functions: AbstractSet[str]
) -> Union[Model[T], None]:
    """Converts the given model with no function interpretations to a
    canonically corresponding model with interpretations for the given function
    names, having each new function interpretation replace a canonically
    corresponding relation interpretation.

    Parameters:
        model: model to convert, that contains no function interpretations.
        original_functions: function names for the model to convert to,
            such that no relation name that canonically corresponds to any of
            these function names has an interpretation in the given model.

    Returns:
        A model `model` with the given function names such that
        `replace_functions_with_relations_in_model`\ ``(``\ `model`\ ``)``
        is the given model, or ``None`` if no such model exists.
    """
    assert len(model.function_interpretations) == 0
    for function in original_functions:
        assert is_function(function)
        assert function not in model.function_interpretations
        assert function_name_to_relation_name(function) in model.relation_interpretations

    # Task 8.2
    new_functions = {}
    new_relations = {}
    for name, relation in model.relation_interpretations.items():
        new_name = relation_name_to_function_name(name)
        if new_name not in original_functions:
            new_relations[name] = relation
            continue

        ar = model.relation_arities[name]
        all_args = list(values[1:] for values in relation)
        if len(all_args) != len(model.universe) ** (ar - 1) or len(all_args) != len(set(all_args)):
            return

        new_functions[new_name] = {}
        for value, *args in relation:
            new_functions[new_name][tuple(args)] = value

    return Model(
        model.universe,
        model.constant_interpretations,
        new_relations,
        new_functions
    )


def _compile_term(term: Term) -> List[Formula]:
    """Syntactically compiles the given term into a list of single-function
    invocation steps.

    Parameters:
        term: term to compile, whose root is a function invocation, and which
            contains no variable names that are ``z`` followed by a number.

    Returns:
        A list of steps, each of which is a formula of the form
        ``'``\ `y`\ ``=``\ `f`\ ``(``\ `x1`\ ``,``...\ ``,``\ `xn`\ ``)'``,
        where `y` is a new variable name obtained by calling
        `next`\ ``(``\ `~logic_utils.fresh_variable_name_generator`\ ``)``, `f`
        is a function name, and each of the `x`\ `i` is either a constant name
        or a variable name. If `x`\ `i` is a new variable name, then it is also
        the left-hand side of a previous step, where all of the steps "leading
        up to" `x1` precede those "leading up" to `x2`, etc. If all the returned
        steps hold in any model, then the left-hand-side variable name of the
        last returned step evaluates in that model to the value of the given
        term.
    """
    assert is_function(term.root)
    for variable in term.variables():
        assert not is_z_and_number(variable)

    # Task 8.3
    def dfs(t: Term) -> Term:
        if not is_function(t.root):
            return t

        new_args = []
        for arg in t.arguments:
            new_args.append(dfs(arg))

        new_var = Term(next(fresh_variable_name_generator))
        result.append(Formula('=', [
            new_var, Term(t.root, new_args)
        ]))
        return new_var

    result: List[Formula] = []
    dfs(term)
    return result


def _add_condition(prev: Optional[Formula], new: Formula):
    if prev is None:
        return new
    return Formula('&', prev, new)


def _process_term(term: Term, special_variables: List[str], formula: Optional[Formula] = None) -> Tuple[
    Term, Optional[Formula]]:
    if not is_function(term.root):
        return term, formula

    steps: List[Formula] = _compile_term(term)
    for step in steps:
        new_var, function = step.arguments
        special_variables.append(new_var.root)
        relation_name = function_name_to_relation_name(function.root)
        formula = _add_condition(formula, Formula(
            relation_name, [new_var, *function.arguments]
        ))

    return steps[-1].arguments[0], formula


def _add_special_variables_quantifiers(formula: Formula, special_variables: List[str]):
    for var in special_variables:
        formula = Formula('E', var, formula)
    return formula


def replace_functions_with_relations_in_formula(formula: Formula) -> Formula:
    """Syntactically converts the given formula to a formula that does not
    contain any function invocations, and is "one-way equivalent" in the sense
    that the former holds in a model if and only if the latter holds in the
    canonically corresponding model with no function interpretations.

    Parameters:
        formula: formula to convert, which contains no variable names that are
            ``z`` followed by a number, and such that there exist no canonically
            corresponding function name and relation name that are both invoked
            in this formula.

    Returns:
        A formula such that the given formula holds in any model `model` if and
        only if the returned formula holds in
        `replace_function_with_relations_in_model`\ ``(``\ `model`\ ``)``.
    """
    assert (
            len(
                {function_name_to_relation_name(function) for function, arity in formula.functions()}.intersection(
                    {relation for relation, arity in formula.relations()}
                )
            )
            == 0
    )
    for variable in formula.variables():
        assert not is_z_and_number(variable)

    # Task 8.4
    if is_equality(formula.root) or is_relation(formula.root):
        special_variables = []
        terms, result = [], None
        for arg in formula.arguments:
            term, result = _process_term(arg, special_variables, result)
            terms.append(term)
        result = _add_condition(result, Formula(formula.root, terms))
        return _add_special_variables_quantifiers(result, special_variables)
    elif is_unary(formula.root):
        first = replace_functions_with_relations_in_formula(formula.first)
        return Formula(formula.root, first)
    elif is_binary(formula.root):
        first = replace_functions_with_relations_in_formula(formula.first)
        second = replace_functions_with_relations_in_formula(formula.second)
        return Formula(formula.root, first, second)
    elif is_quantifier(formula.root):
        inner = replace_functions_with_relations_in_formula(formula.statement)
        return Formula(formula.root, formula.variable, inner)

    assert False, "Unexpected formula type"


def _for_arg(n: int, inner_func: Callable, type_: Literal['A', 'E'] = 'A') -> Formula:
    arg_names = [next(fresh_variable_name_generator) for _ in range(n)]
    args = tuple(Term(name) for name in arg_names)

    formula = inner_func(*args)
    for name in arg_names:
        formula = Formula(type_, name, formula)
    return formula


def _make_check_relation_is_function_formula(relation_name):
    def check_function(*args: Term):
        def has_value(y: Term) -> Formula:
            return Formula(relation_name, [y, *args])

        def only_one_value(y1: Term, y2: Term) -> Formula:
            return Formula(
                '->',
                _add_condition(
                    Formula(relation_name, [y1, *args]),
                    Formula(relation_name, [y2, *args])
                ),
                Formula('=', [y1, y2])
            )

        return _add_condition(
            _for_arg(1, has_value, 'E'),
            _for_arg(2, only_one_value)
        )

    return check_function


def replace_functions_with_relations_in_formulas(
        formulas: AbstractSet[Formula],
) -> Set[Formula]:
    """Syntactically converts the given set of formulas to a set of formulas
    that do not contain any function invocations, and is "two-way
    equivalent" in the sense that:

    1. The former holds in a model if and only if the latter holds in the
       canonically corresponding model with no function interpretations.
    2. The latter holds in a model if and only if that model has a
       canonically corresponding model with interpretations for the functions
       names of the former, and the former holds in that model.

    Parameters:
        formulas: formulas to convert, which contain no variable names that are
            ``z`` followed by a number, and such that there exist no canonically
            corresponding function name and relation name that are both invoked
            in these formulas.

    Returns:
        A set of formulas, one for each given formula as well as one additional
        formula for each relation name that replaces a function name from the
        given formulas, such that:

        1. The given formulas hold in a model `model` if and only if the
           returned formulas hold in
           `replace_functions_with_relations_in_model`\ ``(``\ `model`\ ``)``.
        2. The returned formulas hold in a model `model` if and only if
           `replace_relations_with_functions_in_model`\ ``(``\ `model`\ ``,``\ `original_functions`\ ``)``,
           where `original_functions` are all the function names in the given
           formulas, is a model and the given formulas hold in it.
    """
    assert (
            len(
                set.union(
                    *[
                        {function_name_to_relation_name(function) for function, arity in formula.functions()}
                        for formula in formulas
                    ]
                ).intersection(
                    set.union(*[{relation for relation, arity in formula.relations()} for formula in formulas]))
            )
            == 0
    )
    for formula in formulas:
        for variable in formula.variables():
            assert not is_z_and_number(variable)

    # Task 8.5
    result = set()
    for formula in formulas:
        new_formula = replace_functions_with_relations_in_formula(formula)
        for function_name, function_arity in formula.functions():
            relation_name = function_name_to_relation_name(function_name)
            new_formula = _add_condition(new_formula, _for_arg(
                function_arity,
                _make_check_relation_is_function_formula(relation_name)
            ))
        result.add(new_formula)

    return result


def _replace_equality_with_SAME_in_formula(formula: Formula) -> Formula:
    if is_equality(formula.root) or is_relation(formula.root):
        new_root = formula.root if formula.root != '=' else 'SAME'
        return Formula(new_root, formula.arguments)
    elif is_unary(formula.root):
        return Formula(formula.root, _replace_equality_with_SAME_in_formula(formula.first))
    elif is_binary(formula.root):
        return Formula(formula.root,
                       _replace_equality_with_SAME_in_formula(formula.first),
                       _replace_equality_with_SAME_in_formula(formula.second))
    elif is_quantifier(formula.root):
        return Formula(formula.root, formula.variable, _replace_equality_with_SAME_in_formula(formula.statement))

    assert False, "Unexpected formula type"


def _add_SAME_requirements() -> Formula:
    formula = _for_arg(1, lambda x: Formula('SAME', [x, x]))
    formula = _add_condition(formula, _for_arg(2, lambda x, y: Formula(
        '->',
        Formula('SAME', [x, y]),
        Formula('SAME', [y, x])
    )))
    formula = _add_condition(formula, _for_arg(3, lambda x, y, z: Formula(
        '->',
        Formula('&', Formula('SAME', [x, y]), Formula('SAME', [y, z])),
        Formula('SAME', [x, z])
    )))
    return formula


def _make_check_relation_with_SAME(relation_name):
    def check_relation(*args: Term):
        n = len(args) // 2
        formula = None
        for i in range(n):
            formula = _add_condition(formula, Formula(
                'SAME', [args[i], args[n + i]]
            ))

        first = Formula(relation_name, args[:n])
        second = Formula(relation_name, args[n:])
        return Formula(
            '->',
            formula,
            Formula(
                '&',
                Formula('->', first, second),
                Formula('->', second, first)
            )
        )

    return check_relation


def replace_equality_with_SAME_in_formulas(
        formulas: AbstractSet[Formula],
) -> Set[Formula]:
    """Syntactically converts the given set of formulas to a canonically
    corresponding set of formulas that do not contain any equalities, consisting
    of the following formulas:

    1. A formula for each of the given formulas, where each equality is
       replaced with a matching invocation of the relation name ``'SAME'``.
    2. Formula(s) that ensure that in any model of the returned formulas, the
       interpretation of the relation name ``'SAME'`` is reflexive,
       symmetric, and transitive.
    3. For each relation name from the given formulas, formula(s) that ensure
       that in any model of the returned formulas, the interpretation of this
       relation name respects the interpretation of the relation name
       ``'SAME'``.

    Parameters:
        formulas: formulas to convert, that contain no function names and do not
            contain the relation name ``'SAME'``.

    Returns:
        The converted set of formulas.
    """
    for formula in formulas:
        assert len(formula.functions()) == 0
        assert "SAME" not in {relation for relation, arity in formula.relations()}

    # Task 8.6
    result = set()
    for formula in formulas:
        formula = _replace_equality_with_SAME_in_formula(formula)
        result.add(_add_SAME_requirements())

        for relation_name, relation_arity in formula.relations():
            if relation_name == 'SAME':
                continue

            result.add(_for_arg(
                2 * relation_arity,
                _make_check_relation_with_SAME(relation_name)
            ))

        result.add(formula)
    return result


def add_SAME_as_equality_in_model(model: Model[T]) -> Model[T]:
    """Adds an interpretation of the relation name ``'SAME'`` in the given
    model, that canonically corresponds to equality in the given model.

    Parameters:
        model: model that has no interpretation of the relation name
            ``'SAME'``, to add the interpretation to.

    Returns:
        A model obtained from the given model by adding an interpretation of the
        relation name ``'SAME'``, that contains precisely all pairs
        ``(``\ `x`\ ``,``\ `x`\ ``)`` for every element `x` of the universe of
        the given model.
    """
    assert "SAME" not in model.relation_interpretations

    # Task 8.7
    relations = dict(model.relation_interpretations)
    relations['SAME'] = set()
    for x in model.universe:
        relations['SAME'].add((x, x))

    return Model(
        model.universe, model.constant_interpretations, relations, model.function_interpretations
    )


def make_equality_as_SAME_in_model(model: Model[T]) -> Model[T]:
    """Converts the given model to a model where equality coincides with the
    interpretation of ``'SAME'`` in the given model, in the sense that any set
    of formulas holds in the returned model if and only if its canonically
    corresponding set of formulas that do not contain equality holds in the
    given model.

    Parameters:
        model: model to convert, that contains no function interpretations, and
            contains an interpretation of the relation name ``'SAME'`` that is
            reflexive, symmetric, transitive, and respected by the
            interpretations of all other relation names.

    Returns:
        A model that is a model of any set `formulas` if and only if the given
        model is a model of
        `replace_equality_with_SAME`\ ``(``\ `formulas`\ ``)``. The universe of
        the returned model corresponds to the equivalence classes of the
        interpretation of ``'SAME'`` in the given model.
    """
    assert "SAME" in model.relation_interpretations and model.relation_arities["SAME"] == 2
    assert len(model.function_interpretations) == 0
    # Task 8.8

    new_universe = set()
    new_values = {}
    same = model.relation_interpretations['SAME']
    for x in model.universe:
        for y in new_universe:
            if (x, y) in same:
                new_values[x] = y
                break
        else:
            new_values[x] = x
            new_universe.add(x)

    new_relations = {
        key: set(tuple(new_values[value] for value in values) for values in relation)
        for key, relation in model.relation_interpretations.items() if key != 'SAME'
    }
    return Model(
        new_universe,
        {key: new_values[value] for key, value in model.constant_interpretations.items()},
        new_relations,
        {}
    )
