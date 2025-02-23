# File: propositions/semantics.py
"""Semantic analysis of propositional-logic constructs."""

from itertools import product
from typing import Iterable
from src.propositions.proofs import *
from src.propositions.syntax import *

Model = Mapping[str, bool]


def is_model(model: Model) -> bool:
    for key in model:
        if not is_variable(key):
            return False
    return True


def variables(model: Model) -> AbstractSet[str]:
    assert is_model(model)
    return model.keys()


def evaluate(formula: Formula, model: Model) -> bool:
    assert is_model(model)
    assert formula.variables().issubset(variables(model))
    if is_variable(formula.root):
        return model[formula.root]
    elif is_constant(formula.root):
        return True if formula.root == "T" else False
    elif is_unary(formula.root):
        return not evaluate(formula.first, model)
    else:
        left_val = evaluate(formula.first, model)
        right_val = evaluate(formula.second, model)
        if formula.root == "&":
            return left_val and right_val
        elif formula.root == "|":
            return left_val or right_val
        elif formula.root == "->":
            return (not left_val) or right_val
        elif formula.root == "+":
            return (left_val and not right_val) or (not left_val and right_val)
        elif formula.root == "<->":
            return left_val == right_val
        elif formula.root == "-&":
            return not (left_val and right_val)
        elif formula.root == "-|":
            return not (left_val or right_val)
        else:
            raise Exception("Неизвестный оператор: " + formula.root)


def all_models(variables: Sequence[str]) -> Iterable[Model]:
    for v in variables:
        assert is_variable(v)
    for values in product([False, True], repeat=len(variables)):
        yield {v: val for v, val in zip(variables, values)}


def truth_values(formula: Formula, models: Iterable[Model]) -> Iterable[bool]:
    return (evaluate(formula, model) for model in models)


def print_truth_table(formula: Formula) -> None:
    vars_sorted = sorted(formula.variables())
    headers = list(vars_sorted) + [str(formula)]
    widths = [max(len(h), 1) for h in headers]

    def format_cell(text, width):
        return f" {text:^{width}} "

    header_line = "|" + "|".join(format_cell(h, w) for h, w in zip(headers, widths)) + "|"
    separator_line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    print(header_line)
    print(separator_line)
    models_list = list(all_models(vars_sorted))
    for model in models_list:
        row_cells = [format_cell("T" if model[v] else "F", w) for v, w in zip(vars_sorted, widths[:-1])]
        formula_val = "T" if evaluate(formula, model) else "F"
        row_cells.append(format_cell(formula_val, widths[-1]))
        row_line = "|" + "|".join(row_cells) + "|"
        print(row_line)


def is_tautology(formula: Formula) -> bool:
    vars_list = sorted(formula.variables())
    return all(evaluate(formula, model) for model in all_models(vars_list))


def is_contradiction(formula: Formula) -> bool:
    vars_list = sorted(formula.variables())
    return all(not evaluate(formula, model) for model in all_models(vars_list))


def is_satisfiable(formula: Formula) -> bool:
    vars_list = sorted(formula.variables())
    return any(evaluate(formula, model) for model in all_models(vars_list))


def _synthesize_for_model(model: Model) -> Formula:
    assert is_model(model)
    assert len(model.keys()) > 0
    literals = []
    for v in sorted(model.keys()):
        if model[v]:
            literals.append(Formula(v))
        else:
            literals.append(Formula("~", Formula(v)))
    if len(literals) == 1:
        return literals[0]
    conj = Formula("&", literals[0], literals[1])
    for lit in literals[2:]:
        conj = Formula("&", conj, lit)
    return conj


def synthesize(variables: Sequence[str], values: Iterable[bool]) -> Formula:
    assert len(variables) > 0
    models_list = list(all_models(variables))
    dnf_clauses = []
    for model, truth in zip(models_list, values):
        if truth:
            dnf_clauses.append(_synthesize_for_model(model))
    if not dnf_clauses:
        p = variables[0]
        return Formula("&", Formula("~", Formula(p)), Formula(p))
    if len(dnf_clauses) == 1:
        return dnf_clauses[0]
    disj = Formula("|", dnf_clauses[0], dnf_clauses[1])
    for clause in dnf_clauses[2:]:
        disj = Formula("|", disj, clause)
    return disj


def _synthesize_for_all_except_model(model: Model) -> Formula:
    assert is_model(model)
    assert len(model.keys()) > 0
    literals = []
    for v in sorted(model.keys()):
        if model[v]:
            literals.append(Formula("~", Formula(v)))
        else:
            literals.append(Formula(v))
    if len(literals) == 1:
        return literals[0]
    disj = Formula("|", literals[0], literals[1])
    for lit in literals[2:]:
        disj = Formula("|", disj, lit)
    return disj


def synthesize_cnf(variables: Sequence[str], values: Iterable[bool]) -> Formula:
    assert len(variables) > 0
    models_list = list(all_models(variables))
    cnf_clauses = []
    for model, truth in zip(models_list, values):
        if not truth:
            cnf_clauses.append(_synthesize_for_all_except_model(model))
    if not cnf_clauses:
        p = variables[0]
        return Formula("|", Formula(p), Formula("~", Formula(p)))
    if len(cnf_clauses) == 1:
        return cnf_clauses[0]
    conj = Formula("&", cnf_clauses[0], cnf_clauses[1])
    for clause in cnf_clauses[2:]:
        conj = Formula("&", conj, clause)
    return conj


def evaluate_inference(rule: InferenceRule, model: Model) -> bool:
    assert is_model(model)
    for assumption in rule.assumptions:
        if not evaluate(assumption, model):
            return True
    return evaluate(rule.conclusion, model)


def is_sound_inference(rule: InferenceRule) -> bool:
    vars_set = set(rule.conclusion.variables())
    for assumption in rule.assumptions:
        vars_set |= assumption.variables()
    for model in all_models(sorted(vars_set)):
        if all(evaluate(a, model) for a in rule.assumptions) and not evaluate(rule.conclusion, model):
            return False
    return True
