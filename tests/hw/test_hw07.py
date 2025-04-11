# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: test_chapter07.py

"""Tests all Chapter 7 tasks."""

from tests.predicates import semantics_test
from tests.predicates import syntax_test


def test_task1(debug=False):
    syntax_test.test_term_repr(debug)


def test_task2(debug=False):
    syntax_test.test_formula_repr(debug)


def test_task3(debug=False):
    syntax_test.test_term_parse_prefix(debug)
    syntax_test.test_term_parse(debug)


def test_task4(debug=False):
    syntax_test.test_formula_parse_prefix(debug)
    syntax_test.test_formula_parse(debug)


def test_task5(debug=False):
    syntax_test.test_term_constants(debug)
    syntax_test.test_term_variables(debug)
    syntax_test.test_term_functions(debug)


def test_task6(debug=False):
    syntax_test.test_formula_constants(debug)
    syntax_test.test_formula_variables(debug)
    syntax_test.test_free_variables(debug)
    syntax_test.test_formula_functions(debug)
    syntax_test.test_relations(debug)


def test_task7(debug=False):
    semantics_test.test_evaluate_term(debug)


def test_task8(debug=False):
    semantics_test.test_evaluate_formula(debug)


def test_task9(debug=False):
    semantics_test.test_is_model_of(debug)
