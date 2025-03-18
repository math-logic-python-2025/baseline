# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: propositions/deduction.py

"""Useful proof manipulation maneuvers in Propositional Logic."""

from src.propositions.axiomatic_systems import *


def add_offset(lines: Tuple[Proof.Line, ...], offset: int) -> Tuple[Proof.Line, ...]:
    return tuple(
        line if line.is_assumption() else Proof.Line(
            line.formula,
            line.rule,
            [i + offset for i in line.assumptions]
        )
        for line in lines
    )


def prove_corollary(antecedent_proof: Proof, consequent: Formula, conditional: InferenceRule) -> Proof:
    """Converts the given proof of a formula `antecedent` to a proof of the
    given formula `consequent` by using the given assumptionless inference rule
    of which ``'(``\ `antecedent`\ ``->``\ `consequent`\ ``)'`` is a
    specialization.

    Parameters:
        antecedent_proof: valid proof of `antecedent`.
        consequent: formula to prove.
        conditional: assumptionless inference rule of which the assumptionless
            inference rule with conclusion
            ``'(``\ `antecedent`\ ``->``\ `consequent`\ ``)'`` is a
            specialization.

    Returns:
        A valid proof of `consequent` from the same assumptions as the given
        proof, via the same inference rules as the given proof and in addition
        `~propositions.axiomatic_systems.MP` and `conditional`.
    """
    assert antecedent_proof.is_valid()
    assert InferenceRule([], Formula("->", antecedent_proof.statement.conclusion, consequent)).is_specialization_of(
        conditional
    )
    # Task 5.3a
    lines_count = len(antecedent_proof.lines)

    return Proof(
        InferenceRule(
            antecedent_proof.statement.assumptions, consequent
        ),
        antecedent_proof.rules | {conditional, MP},
        (
            *antecedent_proof.lines,
            Proof.Line(Formula("->", antecedent_proof.statement.conclusion, consequent), conditional, []),
            Proof.Line(consequent, MP, [lines_count - 1, lines_count])
        )
    )


def combine_proofs(
        antecedent1_proof: Proof,
        antecedent2_proof: Proof,
        consequent: Formula,
        double_conditional: InferenceRule,
) -> Proof:
    """Combines the given proofs of two formulas `antecedent1` and `antecedent2`
    into a proof of the given formula `consequent` by using the given
    assumptionless inference rule of which
    ``'(``\ `antecedent1`\ ``->(``\ `antecedent2`\ ``->``\ `consequent`\ ``))'``
    is a specialization.

    Parameters:
        antecedent1_proof: valid proof of `antecedent1`.
        antecedent2_proof: valid proof of `antecedent2` from the same
            assumptions and inference rules as `antecedent1_proof`.
        consequent: formula to prove.
        double_conditional: assumptionless inference rule of which the
            assumptionless inference rule with conclusion
            ``'(``\ `antecedent1`\ ``->(``\ `antecedent2`\ ``->``\ `consequent`\ ``))'``
            is a specialization.

    Returns:
        A valid proof of `consequent` from the same assumptions as the given
        proofs, via the same inference rules as the given proofs and in addition
        `~propositions.axiomatic_systems.MP` and `double_conditional`.
    """
    assert antecedent1_proof.is_valid()
    assert antecedent2_proof.is_valid()
    assert antecedent1_proof.statement.assumptions == antecedent2_proof.statement.assumptions
    assert antecedent1_proof.rules == antecedent2_proof.rules
    assert InferenceRule(
        [],
        Formula(
            "->",
            antecedent1_proof.statement.conclusion,
            Formula("->", antecedent2_proof.statement.conclusion, consequent),
        ),
    ).is_specialization_of(double_conditional)
    # Task 5.3b
    antecedent1_lines = antecedent1_proof.lines
    antecedent2_lines = antecedent2_proof.lines

    offset = len(antecedent1_lines)
    lines_count_1 = offset
    lines_count_2 = lines_count_1 + len(antecedent2_lines)

    conclusion1 = antecedent1_proof.statement.conclusion
    conclusion2 = antecedent2_proof.statement.conclusion

    double_conditional_formula = Formula("->", conclusion1, Formula("->", conclusion2, consequent))
    mp_formula = Formula("->", conclusion2, consequent)

    new_line_double_conditional = Proof.Line(double_conditional_formula, double_conditional, [])
    new_line_mp1 = Proof.Line(mp_formula, MP, [lines_count_1 - 1, lines_count_2])
    new_line_mp2 = Proof.Line(consequent, MP, [lines_count_2 - 1, lines_count_2 + 1])

    offset_lines = add_offset(antecedent2_lines, offset)

    new_lines = (
        *antecedent1_lines,
        *offset_lines,
        new_line_double_conditional,
        new_line_mp1,
        new_line_mp2
    )

    return Proof(
        InferenceRule(antecedent1_proof.statement.assumptions, consequent),
        antecedent1_proof.rules | {double_conditional, MP},
        new_lines
    )


def remove_assumption(proof: Proof) -> Proof:
    """Converts the given proof of some `conclusion` formula, the last
    assumption of which is an assumption `assumption`, to a proof of
    ``'(``\ `assumption`\ ``->``\ `conclusion`\ ``)'`` from the same assumptions
    except `assumption`.

    Parameters:
        proof: valid proof to convert, with at least one assumption, via some
            set of inference rules all of which have no assumptions except
            perhaps `~propositions.axiomatic_systems.MP`.

    Returns:
        A valid proof of ``'(``\ `assumption`\ ``->``\ `conclusion`\ ``)'``
        from the same assumptions as the given proof except the last one, via
        the same inference rules as the given proof and in addition
        `~propositions.axiomatic_systems.MP`,
        `~propositions.axiomatic_systems.I0`,
        `~propositions.axiomatic_systems.I1`, and
        `~propositions.axiomatic_systems.D`.
    """
    assert proof.is_valid()
    assert len(proof.statement.assumptions) > 0
    for rule in proof.rules:
        assert rule == MP or len(rule.assumptions) == 0
    # Task 5.4
    last_assumption = proof.statement.assumptions[-1]
    conclusion = proof.statement.conclusion

    new_positions = {}
    new_lines = []

    for i, line in enumerate(proof.lines):
        current_index = len(new_lines)

        if line.is_assumption():
            if line.formula == last_assumption:
                self_implication = Formula('->', last_assumption, last_assumption)
                new_line = Proof.Line(self_implication, I0, [])
                new_lines.append(new_line)
            else:
                first_line = line
                inner_implication = Formula('->', last_assumption, line.formula)
                second_formula = Formula('->', line.formula, inner_implication)
                second_line = Proof.Line(second_formula, I1, [])
                third_line = Proof.Line(inner_implication, MP, [current_index, current_index + 1])
                new_lines.extend((first_line, second_line, third_line))

        elif line.rule == MP:
            antecedent_index = line.assumptions[0]
            secondary_index = line.assumptions[1]
            antecedent_line = proof.lines[antecedent_index]

            substitution = {
                'p': last_assumption,
                'q': antecedent_line.formula,
                'r': line.formula
            }
            d_line = Proof.Line(D.conclusion.substitute_variables(substitution), D, [])

            antecedent_implication = Formula('->', last_assumption, antecedent_line.formula)
            consequent_implication = Formula('->', last_assumption, line.formula)
            mp_formula = Formula('->', antecedent_implication, consequent_implication)

            mp_line1 = Proof.Line(mp_formula, MP, [new_positions[secondary_index], current_index])
            mp_line2 = Proof.Line(consequent_implication, MP, [new_positions[antecedent_index], current_index + 1])
            new_lines.extend((d_line, mp_line1, mp_line2))

        else:
            first_line = line
            inner_implication = Formula('->', last_assumption, line.formula)
            second_formula = Formula('->', line.formula, inner_implication)
            second_line = Proof.Line(second_formula, I1, [])
            third_line = Proof.Line(inner_implication, MP, [current_index, current_index + 1])
            new_lines.extend((first_line, second_line, third_line))

        new_positions[i] = len(new_lines) - 1

    new_inference_rule = InferenceRule(
        proof.statement.assumptions[:-1],
        Formula('->', last_assumption, conclusion)
    )
    new_rules = proof.rules | {MP, I0, I1, D}

    return Proof(new_inference_rule, new_rules, new_lines)


def prove_from_opposites(proof_of_affirmation: Proof, proof_of_negation: Proof, conclusion: Formula) -> Proof:
    """Combines the given proofs of a formula `affirmation` and its negation
    ``'~``\ `affirmation`\ ``'`` into a proof of the given formula.

    Parameters:
        proof_of_affirmation: valid proof of `affirmation`.
        proof_of_negation: valid proof of ``'~``\ `affirmation`\ ``'`` from the
            same assumptions and inference rules of `proof_of_affirmation`.
        conclusion: formula to prove.

    Returns:
        A valid proof of `conclusion` from the same assumptions as the given
        proofs, via the same inference rules as the given proofs and in addition
        `~propositions.axiomatic_systems.MP` and
        `~propositions.axiomatic_systems.I2`.
    """
    assert proof_of_affirmation.is_valid()
    assert proof_of_negation.is_valid()
    assert proof_of_affirmation.statement.assumptions == proof_of_negation.statement.assumptions
    assert Formula("~", proof_of_affirmation.statement.conclusion) == proof_of_negation.statement.conclusion
    assert proof_of_affirmation.rules == proof_of_negation.rules
    # Task 5.6
    return combine_proofs(proof_of_negation, proof_of_affirmation, conclusion, I2)


def prove_by_way_of_contradiction(proof: Proof) -> Proof:
    """Converts the given proof of ``'~(p->p)'``, the last assumption of which
    is an assumption ``'~``\ `formula`\ ``'``, to a proof of `formula` from the
    same assumptions except ``'~``\ `formula`\ ``'``.

    Parameters:
        proof: valid proof of ``'~(p->p)'`` to convert, the last assumption of
            which is of the form ``'~``\ `formula`\ ``'``, via some set of
            inference rules all of which have no assumptions except perhaps
            `~propositions.axiomatic_systems.MP`.

    Returns:
        A valid proof of `formula` from the same assumptions as the given proof
        except the last one, via the same inference rules as the given proof and
        in addition `~propositions.axiomatic_systems.MP`,
        `~propositions.axiomatic_systems.I0`,
        `~propositions.axiomatic_systems.I1`,
        `~propositions.axiomatic_systems.D`, and
        `~propositions.axiomatic_systems.N`.
    """
    assert proof.is_valid()
    assert proof.statement.conclusion == Formula.parse("~(p->p)")
    assert len(proof.statement.assumptions) > 0
    assert proof.statement.assumptions[-1].root == "~"
    for rule in proof.rules:
        assert rule == MP or len(rule.assumptions) == 0
    # Task 5.7
    new_proof = remove_assumption(proof)
    old_assumption = proof.statement.assumptions[-1]
    new_conclusion = old_assumption.first
    n = len(new_proof.lines)

    implication_pp = Formula.parse('(p->p)')
    implication_pq = Formula.parse('(p->q)').substitute_variables({
        'p': implication_pp,
        'q': new_conclusion
    })
    n_conclusion = N.conclusion.substitute_variables({
        'p': implication_pp,
        'q': new_conclusion
    })

    line_N = Proof.Line(n_conclusion, N, [])
    line_MP1 = Proof.Line(implication_pq, MP, [n - 1, n])
    line_I0 = Proof.Line(implication_pp, I0, [])
    line_MP2 = Proof.Line(new_conclusion, MP, [n + 2, n + 1])

    new_inference_rule = InferenceRule(new_proof.statement.assumptions, new_conclusion)
    new_rules = new_proof.rules | {N}

    return Proof(
        new_inference_rule,
        new_rules,
        (
            *new_proof.lines,
            line_N,
            line_MP1,
            line_I0,
            line_MP2,
        )
    )

