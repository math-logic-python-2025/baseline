# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: test_chapter03.py

"""Tests all Chapter 3 tasks."""

from tests.propositions import semantics_test, operators_test, syntax_test


def test_before_tasks(debug=False):
    assert operators_test.is_binary("+"), "Change is_binary() before testing Chapter 3 tasks."
    operators_test.test_operators_defined(debug)


def test_task1(debug=False):
    syntax_test.test_repr(debug)
    syntax_test.test_repr_all_operators(debug)
    syntax_test.test_variables(debug)
    syntax_test.test_variables_all_operators(debug)
    syntax_test.test_operators(debug)
    syntax_test.test_operators_all_operators(debug)
    syntax_test.test_parse_prefix(debug)
    syntax_test.test_parse_prefix_all_operators(debug)
    syntax_test.test_is_formula(debug)
    syntax_test.test_is_formula_all_operators(debug)
    syntax_test.test_parse(debug)
    syntax_test.test_parse_all_operators(debug)


def test_task2(debug=False):
    semantics_test.test_evaluate(debug)
    semantics_test.test_evaluate_all_operators(debug)
    semantics_test.test_truth_values(debug)
    semantics_test.test_is_tautology(debug)
    semantics_test.test_is_contradiction(debug)
    semantics_test.test_is_satisfiable(debug)
    semantics_test.test_is_tautology_all_operators(debug)
    semantics_test.test_print_truth_table(debug)


def test_task3(debug=False):
    syntax_test.test_substitute_variables(debug)


def test_task4(debug=False):
    syntax_test.test_substitute_operators(debug)


def test_task5(debug=False):
    operators_test.test_to_not_and_or(debug)


def test_task6a(debug=False):
    operators_test.test_to_not_and(debug)


def test_task6b(debug=False):
    operators_test.test_to_nand(debug)


def test_task6c(debug=False):
    operators_test.test_to_implies_not(debug)


def test_task6d(debug=False):
    operators_test.test_to_implies_false(debug)
