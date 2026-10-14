from headrouter.classifier.embedding import Router
from headrouter.evaluation.metrics import extract_number, math_correct, rouge_l


def test_gsm8k_answer_extraction():
    assert extract_number("so 3 + 4 = 7. The answer is 7.") == "7"
    assert extract_number("The answer is $1,250.") == "1250"
    assert extract_number("first 10, then 12 apples") == "12"
    assert math_correct("The answer is 18.", "18")
    assert not math_correct("The answer is 17.", "18")
    assert extract_number("no numbers here") is None


def test_rouge_l_bounds():
    assert rouge_l("the cat sat", "the cat sat") == 1.0
    assert rouge_l("dogs", "the cat sat") == 0.0


def test_router_fallbacks():
    masks = {"sentiment": {"sparsity": 0.3}, "math": {"sparsity": 0.0}}
    r = Router(None, masks, 0.8)
    assert r.route("x")["variant"] == "dense"
    assert r.route("x", force_task="sentiment")["variant"] == "sentiment"
    assert r.route("x", force_task="math")["variant"] == "dense"        # no level met the budget
    assert r.route("x", force_task="general")["variant"] == "dense"
