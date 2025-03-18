import tests.propositions.tautology_test as tautology_test


def test_task1(debug=False):
    tautology_test.test_formulas_capturing_model(debug)
    tautology_test.test_prove_in_model(debug)


def test_task2(debug=False):
    tautology_test.test_reduce_assumption(debug)


def test_task3(debug=False):
    tautology_test.test_prove_tautology(debug)
    tautology_test.test_proof_or_counterexample(debug)


def test_task4(debug=False):
    tautology_test.test_encode_as_formula(debug)
    tautology_test.test_prove_sound_inference(debug)


def test_task5(debug=False):
    tautology_test.test_model_or_inconsistency(debug)
