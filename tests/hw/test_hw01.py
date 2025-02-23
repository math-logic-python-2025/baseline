# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: test_chapter01.py

"""Tests all Chapter 1 tasks."""

from tests.propositions import syntax_test


def test_task1(debug=False):
    syntax_test.test_repr(debug)


def test_task2(debug=False):
    syntax_test.test_variables(debug)


def test_task3(debug=False):
    syntax_test.test_operators(debug)


def test_task4(debug=False):
    syntax_test.test_parse_prefix(debug)


def test_task5(debug=False):
    syntax_test.test_is_formula(debug)


def test_task6(debug=False):
    syntax_test.test_parse(debug)
