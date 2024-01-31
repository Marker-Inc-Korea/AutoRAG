import pytest

from autorag.evaluate.metric.generation import bleu, meteor, rouge

generation_gts = [
    ['The dog had bit the man.', 'The man had bitten the dog.'],
    ['I want to be a artist, but I end up to be a programmer.'],
    ['To be a artist these days, you can overcome by AI.',
     'To be a programmer these days, you can overcome by AI.',
     'To be a lawyer these days, you can overcome by AI.'],
]
generations = [
    'The dog bit the man.',
    'It really like to be a programmer, but I think artist is my passion.',
    'To be a artist these days, you can overcome by AI.',
]


def base_test_generation_metrics(func, solution):
    scores = func(generation_gt=generation_gts, generations=generations)
    assert len(scores) == len(generation_gts)
    assert all(isinstance(score, float) for score in scores)
    assert all(list(map(lambda x: x[0] == pytest.approx(x[1], 0.001),
                        zip(scores, solution))))


def test_bleu():
    base_test_generation_metrics(bleu,  [51.1507, 23.5783, 100.0])


def test_meteor():
    base_test_generation_metrics(meteor, [0.853462, 0.5859375, 1.0])


def test_rouge():
    base_test_generation_metrics(rouge, [0.909, 0.35714, 1.0])
