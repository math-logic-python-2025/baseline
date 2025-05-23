# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: test_chapter09.py

"""Tests all Chapter 9 tasks."""

from tests.predicates import proofs_test
from tests.predicates import syntax_test


def test_task1(debug=False):
    syntax_test.test_term_substitute(debug)


def test_task2(debug=False):
    syntax_test.test_formula_substitute(debug)


def test_task3(debug=False):
    proofs_test.test_instantiate_helper(debug)


def test_task4(debug=False):
    proofs_test.test_instantiate(debug)


def test_task5(debug=False):
    proofs_test.test_assumption_line_is_valid(debug)


def test_task6(debug=False):
    proofs_test.test_mp_line_is_valid(debug)


def test_task7(debug=False):
    proofs_test.test_ug_line_is_valid(debug)


def test_task8(debug=False):
    syntax_test.test_propositional_skeleton(debug)


def test_task9(debug=False):
    proofs_test.test_tautology_line_is_valid(debug)


def test_task10(debug=False):
    syntax_test.test_from_propositional_skeleton(debug)


def test_task11(debug=False):
    proofs_test.test_axiom_specialization_map_to_schema_instantiation_map(debug)
    proofs_test.test_prove_from_skeleton_proof(debug)


def test_task12(debug=False):
    proofs_test.test_prove_tautology(debug)
