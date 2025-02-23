# File: propositions/operators.py
"""Syntactic conversion of propositional formulas to use only specific sets of operators."""

from src.propositions.syntax import Formula, is_variable, is_constant, is_unary


def to_not_and_or(formula: Formula) -> Formula:
    if is_variable(formula.root):
        return formula
    if is_constant(formula.root):
        if formula.root == "T":
            return Formula("|", Formula("p"), Formula("~", Formula("p")))
        else:
            return Formula("&", Formula("p"), Formula("~", Formula("p")))
    if is_unary(formula.root):
        return Formula("~", to_not_and_or(formula.first))
    left = to_not_and_or(formula.first)
    right = to_not_and_or(formula.second)
    op = formula.root
    if op in {"&", "|"}:
        return Formula(op, left, right)
    elif op == "->":
        return Formula("|", Formula("~", left), right)
    elif op == "+":
        return Formula("&", Formula("|", left, right), Formula("~", Formula("&", left, right)))
    elif op == "<->":
        return Formula("|", Formula("&", left, right), Formula("&", Formula("~", left), Formula("~", right)))
    elif op == "-&":
        return Formula("~", Formula("&", left, right))
    elif op == "-|":
        return Formula("~", Formula("|", left, right))
    else:
        raise Exception("Unknown operator in to_not_and_or: " + op)


def to_not_and(formula: Formula) -> Formula:
    f = to_not_and_or(formula)
    if is_variable(f.root) or is_constant(f.root):
        return f
    if is_unary(f.root):
        return Formula("~", to_not_and(f.first))
    left = to_not_and(f.first)
    right = to_not_and(f.second)
    if f.root == "&":
        return Formula("&", left, right)
    elif f.root == "|":
        return Formula("~", Formula("&", Formula("~", left), Formula("~", right)))
    else:
        raise Exception("Unexpected operator in to_not_and: " + f.root)


def to_nand(formula: Formula) -> Formula:
    f = to_not_and(formula)
    if is_variable(f.root) or is_constant(f.root):
        return f
    if is_unary(f.root):
        sub = to_nand(f.first)
        return Formula("-&", sub, sub)
    left = to_nand(f.first)
    right = to_nand(f.second)
    nand_expr = Formula("-&", left, right)
    return Formula("-&", nand_expr, nand_expr)


def to_implies_not(formula: Formula) -> Formula:
    f = to_not_and_or(formula)
    if is_variable(f.root) or is_constant(f.root):
        if is_constant(f.root):
            if f.root == "T":
                return Formula("->", Formula("p"), Formula("p"))
            else:
                return Formula("->", Formula("p"), Formula("~", Formula("p")))
        return f
    if is_unary(f.root):
        return Formula("~", to_implies_not(f.first))
    left = to_implies_not(f.first)
    right = to_implies_not(f.second)
    if f.root == "&":
        return Formula("~", Formula("->", left, Formula("~", right)))
    elif f.root == "|":
        return Formula("->", Formula("~", left), right)
    elif f.root == "->":
        return Formula("->", left, right)
    else:
        raise Exception("Unexpected operator in to_implies_not: " + f.root)


def to_implies_false(formula: Formula) -> Formula:
    f = to_not_and_or(formula)
    if is_variable(f.root) or is_constant(f.root):
        if is_constant(f.root):
            if f.root == "T":
                return Formula("->", Formula("p"), Formula("p"))
            else:
                return Formula("->", Formula("p"), Formula("~", Formula("p")))
        return f
    if is_unary(f.root):
        sub = to_implies_false(f.first)
        return Formula("->", sub, Formula("F"))
    left = to_implies_false(f.first)
    right = to_implies_false(f.second)
    if f.root == "&":
        return Formula("->", Formula("->", left, Formula("->", right, Formula("F"))), Formula("F"))
    elif f.root == "|":
        return Formula("->", Formula("->", left, Formula("F")), right)
    elif f.root == "->":
        return Formula("->", left, right)
    else:
        raise Exception("Unexpected operator in to_implies_false: " + f.root)
