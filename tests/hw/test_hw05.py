from tests.propositions import proofs_test, some_proofs_test, deduction_test


def test_task1(debug=False):
    proofs_test.test_prove_specialization(debug)


def test_task2(debug=False):
    proofs_test.test_inline_proof_once(debug)
    proofs_test.test_inline_proof(debug)


def test_task3(debug=False):
    deduction_test.test_prove_corollary(debug)
    deduction_test.test_combine_proofs(debug)


def test_task4(debug=False):
    deduction_test.test_remove_assumption(debug)


def test_task5(debug=False):
    some_proofs_test.test_prove_hypothetical_syllogism(debug)


def test_task6(debug=False):
    deduction_test.test_prove_from_opposites(debug)


def test_task7(debug=False):
    deduction_test.test_prove_by_way_of_contradiction(debug)
