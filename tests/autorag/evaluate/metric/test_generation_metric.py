import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.evaluate.metric import bleu, meteor, rouge, sem_score, g_eval, bert_score
from tests.delete_tests import is_github_action

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

ko_generation_gts = [
    ['개가 남자를 물었다.', '남자가 개를 물었다.'],
    ['나는 예술가가 되고 싶었지만, 결국 프로그래머가 되었다.'],
    ['요즘 예술가가 되려면, AI를 이겨야 한다.',
     '요즘 프로그래머가 되려면, AI를 이겨야 한다.',
     '요즘 변호사가 되려면, AI를 이겨야 한다.'],
]

ko_generations = [
    '개가 남자를 물었다.',
    '나는 정말이지 예술가가 되고 싶었지만, 결국 프로그래머가 되었다.',
    '요즘 세상에서는 예술가가 되려면, AI를 이겨야 한다.',
]


def base_test_generation_metrics(func, solution, **kwargs):
    scores = func(generation_gt=generation_gts, generations=generations, **kwargs)
    assert len(scores) == len(generation_gts)
    assert all(isinstance(score, float) for score in scores)
    assert all(list(map(lambda x: x[0] == pytest.approx(x[1], 0.001),
                        zip(scores, solution))))


def ko_base_test_generation_metrics(func, solution, **kwargs):
    scores = func(generation_gt=ko_generation_gts, generations=ko_generations, **kwargs)
    assert len(scores) == len(ko_generation_gts)
    assert all(isinstance(score, float) for score in scores)
    assert all(list(map(lambda x: x[0] == pytest.approx(x[1], 0.001),
                        zip(scores, solution))))


def test_bleu():
    base_test_generation_metrics(bleu, [51.1507, 23.5783, 100.0], lowercase=True)


def test_meteor():
    base_test_generation_metrics(meteor, [0.454033, 0.2985435, 0.64077828], alpha=0.85, beta=0.2, gamma=0.6)


def test_rouge():
    base_test_generation_metrics(rouge, [0.909, 0.35714, 1.0])


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions. It use local model")
def test_sem_score():
    base_test_generation_metrics(sem_score, [0.9005998, 0.7952, 1.0])


def test_sem_score_other_model():
    base_test_generation_metrics(sem_score, [0.9888, 0.9394, 1.0], embedding_model=OpenAIEmbedding())


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_g_eval_fluency():
    base_test_generation_metrics(g_eval, [3.0, 2.0, 2.0], metrics=['fluency'], model='gpt-3.5-turbo')


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_g_eval_full():
    base_test_generation_metrics(g_eval, [4.25, 2.5, 4.25], model='gpt-3.5-turbo')


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_bert_score_en():
    base_test_generation_metrics(bert_score, [0.981902, 0.93164, 1.0], n_threads=8)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_bert_score_ko():
    ko_base_test_generation_metrics(bert_score, [1.0, 0.965312, 0.96309], lang='ko')
