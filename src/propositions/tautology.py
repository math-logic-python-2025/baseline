# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: propositions/tautology.py

"""The Tautology Theorem and its implications."""

from typing import List, Tuple, Union, Sequence

from src.logic_utils import frozendict

from src.propositions.semantics import *
from src.propositions.deduction import *


def formulas_capturing_model(model: Model) -> List[Formula]:
    """Computes the formulas that capture the given model: ``'``\ `x`\ ``'``
    for each variable name `x` that is assigned the value ``True`` in the given
    model, and ``'~``\ `x`\ ``'`` for each variable name `x` that is assigned
    the value ``False``.

    Parameters:
        model: model to construct the formulas for.

    Returns:
        A list of the constructed formulas, ordered alphabetically by variable
        name.

    Examples:
        >>> formulas_capturing_model({'p2': False, 'p1': True, 'q': True})
        [p1, ~p2, q]
    """
    assert is_model(model)

    # Task 6.1a
    capturing_formulas = []
    for var in sorted(model.keys()):
        if model[var]:
            capturing_formulas.append(Formula.parse(var))
        else:
            capturing_formulas.append(Formula.parse(f"~{var}"))
    return capturing_formulas


def _prove_in_formula(formula: Formula, model: Model, proof_lines: list[Proof.Line]) -> Tuple[int, bool]:
    """
    Recursively adds proof lines for the given formula according to the model.

    Returns a tuple containing:
      - The index of the last added proof line.
      - The truth value (according to `model`) of the given formula.
    """
    if is_variable(formula.root):
        # Base case: formula is a single variable.
        if model[formula.root]:
            proof_lines.append(Proof.Line(Formula.parse(formula.root)))
        else:
            proof_lines.append(Proof.Line(Formula.parse(f"~{formula.root}")))
        return len(proof_lines) - 1, model[formula.root]

    if formula.root == '->':
        # Recursively prove the antecedent and consequent.
        left_index, left_truth = _prove_in_formula(formula.first, model, proof_lines)
        right_index, right_truth = _prove_in_formula(formula.second, model, proof_lines)

        if not left_truth:
            # Using inference I2.
            proof_lines.append(Proof.Line(
                I2.conclusion.substitute_variables({"p": formula.first, "q": formula.second}),
                I2, []
            ))
            proof_lines.append(Proof.Line(
                formula,
                MP, [left_index, len(proof_lines) - 1]
            ))
        elif right_truth:
            # Using inference I1.
            proof_lines.append(Proof.Line(
                I1.conclusion.substitute_variables({"p": formula.first, "q": formula.second}),
                I1, []
            ))
            proof_lines.append(Proof.Line(
                formula,
                MP, [right_index, len(proof_lines) - 1]
            ))
        else:
            # Using inference NI.
            proof_lines.append(Proof.Line(
                NI.conclusion.substitute_variables({"p": formula.first, "q": formula.second}),
                NI, []
            ))
            proof_lines.append(Proof.Line(
                NI.conclusion.substitute_variables({"p": formula.first, "q": formula.second}).second,
                MP, [left_index, len(proof_lines) - 1]
            ))
            proof_lines.append(Proof.Line(
                Formula("~", formula),
                MP, [right_index, len(proof_lines) - 1]
            ))

        return len(proof_lines) - 1, left_truth <= right_truth

    # Case: formula.root == '~'
    inner_index, inner_truth = _prove_in_formula(formula.first, model, proof_lines)
    if inner_truth:
        proof_lines.append(Proof.Line(
            NN.conclusion.substitute_variables({"p": formula.first}),
            NN, []
        ))
        proof_lines.append(Proof.Line(
            Formula("~", formula),
            MP, [inner_index, len(proof_lines) - 1]
        ))
    return len(proof_lines) - 1, not inner_truth


def prove_in_model(formula: Formula, model: Model) -> Proof:
    """Either proves the given formula or proves its negation, from the formulas
    that capture the given model.

    Parameters:
        formula: formula that contains no constants or operators beyond ``'->'``
            and ``'~'``, whose affirmation or negation is to prove.
        model: model from whose formulas to prove.

    Returns:
        If `formula` evaluates to ``True`` in the given model, then a valid
        proof of `formula`; otherwise a valid proof of ``'~``\ `formula`\ ``'``.
        The returned proof is from the formulas that capture the given model, in
        the order returned by `formulas_capturing_model`\ ``(``\ `model`\ ``)``,
        via `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM`.

    Examples:
        >>> proof = prove_in_model(Formula.parse('(p->q7)'),
        ...                        {'q7': False, 'p': False})
        >>> proof.is_valid()
        True
        >>> proof.statement.conclusion
        (p->q7)
        >>> proof.statement.assumptions
        (~p, ~q7)
        >>> proof.rules == AXIOMATIC_SYSTEM
        True

        >>> proof = prove_in_model(Formula.parse('(p->q7)'),
        ...                        {'q7': False, 'p': True})
        >>> proof.is_valid()
        True
        >>> proof.statement.conclusion
        ~(p->q7)
        >>> proof.statement.assumptions
        (p, ~q7)
        >>> proof.rules == AXIOMATIC_SYSTEM
        True
    """
    assert formula.operators().issubset({"->", "~"})
    assert is_model(model)

    # Task 6.1b
    assumptions = formulas_capturing_model(model)
    proof_lines: list[Proof.Line] = []
    _prove_in_formula(formula, model, proof_lines)
    return Proof(InferenceRule(assumptions, proof_lines[-1].formula), AXIOMATIC_SYSTEM, proof_lines)


def reduce_assumption(proof_from_affirmation: Proof, proof_from_negation: Proof) -> Proof:
    """Combines the two given proofs, both of the same formula `conclusion` and
    from the same assumptions except that the last assumption of the latter is
    the negation of that of the former, into a single proof of `conclusion` from
    only the common assumptions.

    Parameters:
        proof_from_affirmation: valid proof of `conclusion` from one or more
            assumptions, the last of which is an assumption `assumption`.
        proof_from_negation: valid proof of `conclusion` from the same
            assumptions and inference rules of `proof_from_affirmation`, but
            with the last assumption being ``'~``\ `assumption`\ ``'`` instead
            of `assumption`.

    Returns:
        A valid proof of `conclusion` from only the assumptions common to the
        given proofs (i.e., without the last assumption of each), via the same
        inference rules of the given proofs and in addition
        `~propositions.axiomatic_systems.MP`,
        `~propositions.axiomatic_systems.I0`,
        `~propositions.axiomatic_systems.I1`,
        `~propositions.axiomatic_systems.D`, and
        `~propositions.axiomatic_systems.R`.

    Examples:
        If `proof_from_affirmation` is of ``['p', '~q', 'r'] ==> '(p&(r|~r))'``,
        then `proof_from_negation` must be of
        ``['p', '~q', '~r'] ==> '(p&(r|~r))'`` and the returned proof is of
        ``['p', '~q'] ==> '(p&(r|~r))'``.
    """
    assert proof_from_affirmation.is_valid()
    assert proof_from_negation.is_valid()
    assert proof_from_affirmation.statement.conclusion == proof_from_negation.statement.conclusion
    assert len(proof_from_affirmation.statement.assumptions) > 0
    assert len(proof_from_negation.statement.assumptions) > 0
    assert proof_from_affirmation.statement.assumptions[:-1] == proof_from_negation.statement.assumptions[:-1]
    assert (
            Formula("~", proof_from_affirmation.statement.assumptions[-1]) ==
            proof_from_negation.statement.assumptions[-1]
    )
    assert proof_from_affirmation.rules == proof_from_negation.rules

    # Task 6.2
    common_proof_affirm = remove_assumption(proof_from_affirmation)
    common_proof_negate = remove_assumption(proof_from_negation)
    common_conclusion = proof_from_affirmation.statement.conclusion
    return combine_proofs(common_proof_affirm, common_proof_negate, common_conclusion, R)


def prove_tautology(tautology: Formula, model: Model = frozendict()) -> Proof:
    """Proves the given tautology from the formulas that capture the given
    model.

    Parameters:
        tautology: tautology that contains no constants or operators beyond
            ``'->'`` and ``'~'``, to prove.
        model: model over a (possibly empty) prefix (with respect to the
            alphabetical order) of the variable names of `tautology`, from whose
            formulas to prove.

    Returns:
        A valid proof of the given tautology from the formulas that capture the
        given model, in the order returned by
        `formulas_capturing_model`\ ``(``\ `model`\ ``)``, via
        `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM`.

    Examples:
        >>> proof = prove_tautology(Formula.parse('(~(p->p)->q)'),
        ...                         {'p': True, 'q': False})
        >>> proof.is_valid()
        True
        >>> proof.statement.conclusion
        (~(p->p)->q)
        >>> proof.statement.assumptions
        (p, ~q)
        >>> proof.rules == AXIOMATIC_SYSTEM
        True

        >>> proof = prove_tautology(Formula.parse('(~(p->p)->q)'))
        >>> proof.is_valid()
        True
        >>> proof.statement.conclusion
        (~(p->p)->q)
        >>> proof.statement.assumptions
        ()
        >>> proof.rules == AXIOMATIC_SYSTEM
        True
    """
    assert is_tautology(tautology)
    assert tautology.operators().issubset({"->", "~"})
    assert is_model(model)
    sorted_vars = sorted(tautology.variables())
    assert sorted_vars[: len(model)] == sorted(model.keys())

    # Task 6.3a
    if len(model) == len(sorted_vars):
        return prove_in_model(tautology, model)

    proof_affirm = prove_tautology(tautology, model | {sorted_vars[len(model)]: True})
    proof_negate = prove_tautology(tautology, model | {sorted_vars[len(model)]: False})
    return reduce_assumption(proof_affirm, proof_negate)


def proof_or_counterexample(formula: Formula) -> Union[Proof, Model]:
    """Either proves the given formula or finds a model in which it does not
    hold.

    Parameters:
        formula: formula that contains no constants or operators beyond ``'->'``
            and ``'~'``, to either prove or find a counterexample for.

    Returns:
        If the given formula is a tautology, then an assumptionless proof of the
        formula via `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM`,
        otherwise a model in which the given formula does not hold.
    """
    assert formula.operators().issubset({"->", "~"})

    # Task 6.3b
    for candidate_model in all_models(formula.variables()):
        if not evaluate(formula, candidate_model):
            return candidate_model

    return prove_tautology(formula)


def encode_as_formula(rule: InferenceRule) -> Formula:
    """Encodes the given inference rule as a formula consisting of a chain of
    implications.

    Parameters:
        rule: inference rule to encode.

    Returns:
        The formula encoding the given rule.

    Examples:
        >>> encode_as_formula(InferenceRule([Formula('p1'), Formula('p2'),
        ...                                  Formula('p3'), Formula('p4')],
        ...                                 Formula('q')))
        (p1->(p2->(p3->(p4->q))))

        >>> encode_as_formula(InferenceRule([], Formula('q')))
        q
    """
    # Task 6.4a
    encoded_formula = rule.conclusion
    for assumption in rule.assumptions[::-1]:
        encoded_formula = Formula('->', assumption, encoded_formula)
    return encoded_formula


def prove_sound_inference(rule: InferenceRule) -> Proof:
    """Proves the given sound inference rule.

    Parameters:
        rule: sound inference rule whose assumptions and conclusion contain no
            constants or operators beyond ``'->'`` and ``'~'``, to prove.

    Returns:
        A valid proof of the given sound inference rule via
        `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM`.
    """
    assert is_sound_inference(rule)
    for formula in {rule.conclusion}.union(rule.assumptions):
        assert formula.operators().issubset({"->", "~"})

    # Task 6.4b
    encoded_rule_formula = encode_as_formula(rule)
    tautology_proof = prove_tautology(encoded_rule_formula)
    proof_lines = list(tautology_proof.lines)
    base_index = len(proof_lines) - 1

    for i, assumption in enumerate(rule.assumptions):
        proof_lines.append(Proof.Line(assumption))
        # Using MP with the new assumption line and corresponding earlier line.
        proof_lines.append(Proof.Line(
            proof_lines[base_index + 2 * i].formula.second,
            MP, [base_index + 2 * i + 1, base_index + 2 * i]
        ))

    return Proof(rule, tautology_proof.rules, proof_lines)


def model_or_inconsistency(formulas: Sequence[Formula]) -> Union[Model, Proof]:
    """Either finds a model in which all the given formulas hold, or proves
    ``'~(p->p)'`` from these formulas.

    Parameters:
        formulas: formulas that use only the operators ``'->'`` and ``'~'``, to
            either find a model of, or prove ``'~(p->p)'`` from.

    Returns:
        A model in which all of the given formulas hold if such exists,
        otherwise a valid proof of ``'~(p->p)'`` from the given formulas via
        `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM`.
    """
    for formula in formulas:
        assert formula.operators().issubset({"->", "~"})

    # Task 6.5
    all_vars = set()
    for formula in formulas:
        all_vars |= formula.variables()

    for candidate_model in all_models(all_vars):
        if all(evaluate(formula, candidate_model) for formula in formulas):
            return candidate_model

    return prove_sound_inference(InferenceRule(formulas, Formula.parse("~(p->p)")))


def prove_in_model_full(formula: Formula, model: Model) -> Proof:
    """Either proves the given formula or proves its negation, from the formulas
    that capture the given model.

    Parameters:
        formula: formula that contains no operators beyond ``'->'``, ``'~'``,
            ``'&'``, and ``'|'`` (and may contain constants), whose affirmation
            or negation is to prove.
        model: model from whose formulas to prove.

    Returns:
        If `formula` evaluates to ``True`` in the given model, then a valid
        proof of `formula`; otherwise a valid proof of ``'~``\ `formula`\ ``'``.
        The returned proof is from the formulas that capture the given model, in
        the order returned by `formulas_capturing_model`\ ``(``\ `model`\ ``)``,
        via `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM_FULL`.

    Examples:
        >>> proof = prove_in_model_full(Formula.parse('(p&q7)'),
        ...                             {'q7': True, 'p': True})
        >>> proof.is_valid()
        True
        >>> proof.statement.conclusion
        (p&q7)
        >>> proof.statement.assumptions
        (p, q7)
        >>> proof.rules == AXIOMATIC_SYSTEM_FULL
        True

        >>> proof = prove_in_model_full(Formula.parse('(p&q7)'),
        ...                             {'q7': False, 'p': True})
        >>> proof.is_valid()
        True
        >>> proof.statement.conclusion
        ~(p&q7)
        >>> proof.statement.assumptions
        (p, ~q7)
        >>> proof.rules == AXIOMATIC_SYSTEM_FULL
        True
    """
    assert formula.operators().issubset({"T", "F", "->", "~", "&", "|"})
    assert is_model(model)
    # Optional Task 6.6
