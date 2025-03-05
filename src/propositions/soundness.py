# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: propositions/soundness.py

"""Programmatic proof of the soundness of Propositional Logic."""

from src.propositions.semantics import *


def rule_nonsoundness_from_specialization_nonsoundness(
        general: InferenceRule, specialization: InferenceRule, model: Model) \
        -> Model:
    """Demonstrates the non-soundness of the given general inference rule given
    an example of the non-soundness of the given specialization of this rule.

    Parameters:
        general: inference rule to the soundness of which to find a
            counterexample.
        specialization: non-sound specialization of `general`.
        model: model in which `specialization` does not hold.

    Returns:
        A model in which `general` does not hold.
    """
    assert specialization.is_specialization_of(general)
    assert not evaluate_inference(specialization, model)
    # Task 4.9
    mp = general.specialization_map(specialization)
    newmodel = {}
    for p in model.keys():
        newmodel[p] = model[p]
    for p in mp.keys():
        newmodel[p] = evaluate(mp[p], model)
    return newmodel


def nonsound_rule_of_nonsound_proof(proof: Proof, model: Model) -> \
        Tuple[InferenceRule, Model]:
    """Finds a non-sound inference rule used by the given valid proof of a
    non-sound inference rule, and demonstrates the non-soundness of the former
    rule.

    Parameters:
        proof: valid proof of a non-sound inference rule.
        model: model in which the inference rule proved by the given proof does
            not hold.

    Returns:
        A pair of a non-sound inference rule used in the given proof and a model
        in which this rule does not hold.
    """
    assert proof.is_valid()
    assert not evaluate_inference(proof.statement, model)
    # Task 4.10
    for line in range(len(proof.lines) - 1, -1, -1):
        proof_line = proof.lines[line]
        rule = proof.rule_for_line(line)
        assert not evaluate(rule.conclusion, model), 'model is bad'
        bad = True
        for assum in rule.assumptions:
            if not evaluate(assum, model):
                bad = False
                break
        if bad:
            newmodel = rule_nonsoundness_from_specialization_nonsoundness(proof_line.rule, rule, model)
            return proof_line.rule, newmodel
    return None
