# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: test_chapter10.py

"""Tests all Chapter 10 tasks."""

from tests.predicates import prover_test
from tests.predicates import some_proofs_test


def test_skeleton(debug=False):
    prover_test.test_prover_basic(debug)


def test_task1(debug=False):
    prover_test.test_add_universal_instantiation(debug)


def test_task2(debug=False):
    prover_test.test_add_tautological_implication(debug)


def test_task3(debug=False):
    prover_test.test_add_existential_derivation(debug)


def test_task4(debug=False):
    some_proofs_test.test_prove_lovers(debug)


def test_task5(debug=False):
    some_proofs_test.test_prove_homework(debug)


def test_task6(debug=False):
    prover_test.test_add_flipped_equality(debug)


def test_task7(debug=False):
    prover_test.test_add_free_instantiation(debug)


def test_task8(debug=False):
    prover_test.test_add_substituted_equality(debug)


def test_task9(debug=False):
    prover_test.test_add_chained_equality(debug)


def test_task10(debug=False):
    some_proofs_test.test_prove_group_unique_zero(debug)


def test_task11(debug=False):
    some_proofs_test.test_prove_field_zero_multiplication(debug)


def test_task12(debug=False):
    some_proofs_test.test_prove_peano_left_neutral(debug)


def test_task13(debug=False):
    some_proofs_test.test_prove_russell_paradox(debug)
