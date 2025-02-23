from tests.propositions import proofs_test, semantics_test, some_proofs_test, soundness_test


def test_task1(debug=False):
    proofs_test.test_variables(debug)


def test_task2(debug=False):
    semantics_test.test_evaluate_inference(debug)


def test_task3(debug=False):
    semantics_test.test_is_sound_inference(debug)


def test_task4(debug=False):
    proofs_test.test_specialize(debug)


def test_task5(debug=False):
    proofs_test.test_merge_specialization_maps(debug)
    proofs_test.test_formula_specialization_map(debug)
    proofs_test.test_specialization_map(debug)
    proofs_test.test_is_specialization_of(debug)


def test_task6(debug=False):
    proofs_test.test_rule_for_line(debug)
    proofs_test.test_is_line_valid(debug)
    proofs_test.test_is_valid(debug)


def test_task7(debug=False):
    some_proofs_test.test_prove_and_commutativity(debug)


def test_task8(debug=False):
    some_proofs_test.test_prove_I0(debug)


def test_task9(debug=False):
    soundness_test.test_rule_nonsoundness_from_specialization_nonsoundness(debug)


def test_task10(debug=False):
    soundness_test.test_nonsound_rule_of_nonsound_proof(debug)
