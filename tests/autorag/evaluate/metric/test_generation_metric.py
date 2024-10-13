from unittest.mock import patch

import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.evaluation.metric import (
	bleu,
	meteor,
	rouge,
	sem_score,
	g_eval,
	bert_score,
	deepeval_faithfulness,
)
from autorag.schema.metricinput import MetricInput
from tests.delete_tests import is_github_action
from tests.mock import mock_get_text_embedding_batch

generation_gts = [
	["The dog had bit the man.", "The man had bitten the dog."],
	["I want to be a artist, but I end up to be a programmer."],
	[
		"To be a artist these days, you can overcome by AI.",
		"To be a programmer these days, you can overcome by AI.",
		"To be a lawyer these days, you can overcome by AI.",
	],
]

retrieval_gt_contents = [
	[
		[
			"The dog bite something easily. Actually the dog can bite a human. When you see a dog, you should be careful."
		]
	],
	[
		[
			"The artist is a person who creates art. The artist can be a painter, a sculptor, or a musician."
		]
	],
	[
		[
			"AI is a technology that can simulate human intelligence. AI can be used in various fields such as healthcare, finance, and transportation. So its potential is huge."
		]
	],
]

generations = [
	"The dog bit the man.",
	"It really like to be a programmer, but I think artist is my passion.",
	"To be a artist these days, you can overcome by AI.",
]

ko_generation_gts = [
	["개가 남자를 물었다.", "남자가 개를 물었다."],
	["나는 예술가가 되고 싶었지만, 결국 프로그래머가 되었다."],
	[
		"요즘 예술가가 되려면, AI를 이겨야 한다.",
		"요즘 프로그래머가 되려면, AI를 이겨야 한다.",
		"요즘 변호사가 되려면, AI를 이겨야 한다.",
	],
]

ko_generations = [
	"개가 남자를 물었다.",
	"나는 정말이지 예술가가 되고 싶었지만, 결국 프로그래머가 되었다.",
	"요즘 세상에서는 예술가가 되려면, AI를 이겨야 한다.",
]

ja_generation_gts = [
	["犬が男を噛んだ。", "男が犬を噛んだ。"],
	["私は芸術家になりたかったが、結局プログラマーになった。"],
	[
		"最近では、芸術家になるためにはAIに打ち勝つ必要がある。",
		"最近では、プログラマーになるためにはAIに打ち勝つ必要がある。",
		"最近では、弁護士になるためにはAIに打ち勝つ必要がある。",
	],
]

ja_generations = [
	"犬が男を噛んだ。",
	"本当にプログラマーになることになったが、芸術家になるのが自分の情熱だ。",
	"最近では、芸術家になるためにはAIに打ち勝つ必要がある。",
]

summarization_query_list = [
	"""
The 'coverage score' is calculated as the percentage of assessment questions
for which both the summary and the original document provide a 'yes' answer. This
method ensures that the summary not only includes key information from the original
text but also accurately represents it. A higher coverage score indicates a
more comprehensive and faithful summary, signifying that the summary effectively
encapsulates the crucial points and details from the original content.
""",
	"""The 'coverage score' is calculated as the percentage of assessment questions
for which both the summary and the original document provide a 'yes' answer. This
method ensures that the summary not only includes key information from the original
text but also accurately represents it. A higher coverage score indicates a
more comprehensive and faithful summary, signifying that the summary effectively
encapsulates the crucial points and details from the original content.""",
]
summarization_generated_texts_list = [
	"""
The coverage score quantifies how well a summary captures and
accurately represents key information from the original text,
with a higher score indicating greater comprehensiveness.
""",
	"""In the latest One Piece chapters, the story shifts focus to two key developments:

The Straw Hat Crew's Separation: As they head toward Elbaf, the crew gets separated. Nami wakes up in a strange, Lego-like kingdom where she faces a dangerous, transforming creature. Luffy, Zoro, and Sanji come to her rescue, while other crew members’ fates, like Chopper, remain unknown​(
Dexerto
).

Kuma’s Storyline: Bartholomew Kuma’s past continues to unfold. He’s shown making tough choices regarding his daughter, Bonney, and a deal with Vegapunk involving clone technology. His deepening ties with the Revolutionary Army and the threat from the Marines add further tension​(
OtakuKart
).""",
]

summarization_metric_inputs = [
	MetricInput(generated_texts=gen, query=q)
	for gen, q in zip(summarization_generated_texts_list, summarization_query_list)
]
similarity_generation_metric_inputs = [
	MetricInput(
		generated_texts=gen,
		generation_gt=gen_gt,
		retrieval_gt_contents=retrieval_gt_content,
	)
	for gen, gen_gt, retrieval_gt_content in zip(
		generations, generation_gts, retrieval_gt_contents
	)
]
ko_similarity_generation_metric_inputs = [
	MetricInput(generated_texts=gen, generation_gt=gen_gt)
	for gen, gen_gt in zip(ko_generations, ko_generation_gts)
]
ja_similarity_generation_metric_inputs = [
	MetricInput(generated_texts=gen, generation_gt=gen_gt)
	for gen, gen_gt in zip(ja_generations, ja_generation_gts)
]
general_metric_inputs_with_gt = [
	MetricInput(
		query="What are the benefits of space exploration?",
		retrieval_gt_contents=[
			[
				"Space exploration has led to technological advancements such as satellite communication, GPS, "
				"and weather forecasting."
			],
			[
				"It also contributes to scientific research, expanding our understanding of the universe, and fosters international cooperation in space missions."
			],
		],
		retrieved_contents=[
			"Space exploration has resulted in numerous technological advancements, including satellite technology, which has revolutionized communication and weather prediction.",
			"It has also expanded our understanding of the cosmos and encouraged international collaboration in scientific research.",
		],
		generated_texts="The benefits of space exploration include technological innovations like satellite communications and GPS, which have improved life on Earth. Additionally, space exploration contributes to scientific knowledge and fosters international cooperation.",
		generation_gt=[
			"Space exploration brings technological advancements, such as satellites and GPS, that improve daily life. It also enhances our scientific understanding of the universe and encourages cooperation between nations."
		],
	),
	MetricInput(
		query="What are the major causes of climate change?",
		retrieval_gt_contents=[
			[
				"The major causes of climate change include the burning of fossil fuels such as coal, oil, and gas, deforestation, and industrial activities."
			],
			[
				"Human activities like agriculture and waste management also contribute to the increase in greenhouse gases, leading to climate change."
			],
		],
		retrieved_contents=[
			"Climate change is primarily driven by human activities like the burning of fossil fuels, which release carbon dioxide (CO2) and other greenhouse gases.",
			"Deforestation and certain industrial activities also play a significant role in global warming.",
		],
		generated_texts="Climate change is caused by human activities such as the burning of fossil fuels, deforestation, and industrial production. These activities release large amounts of greenhouse gases into the atmosphere, leading to global warming and other climate-related changes.",
		generation_gt=[
			"The main causes of climate change are the burning of fossil fuels, deforestation, and industrial activities that emit greenhouse gases. These gases trap heat in the atmosphere, causing global warming."
		],
	),
]


def base_test_metrics(func, solution, metric_inputs, **kwargs):
	scores = func(metric_inputs, **kwargs)
	assert len(scores) == len(metric_inputs)
	assert all(isinstance(score, float) for score in scores)
	assert all(
		list(map(lambda x: x[0] == pytest.approx(x[1], 0.001), zip(scores, solution)))
	)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it uses OpenAI API but hard to mock.",
)
def test_deepeval_faithfulness():
	base_test_metrics(
		deepeval_faithfulness,
		[1.0, 1.0, 1.0],
		similarity_generation_metric_inputs,
		generator_module_type="openai_llm",
		lang="en",
		llm="gpt-4o-mini-2024-07-18",
	)


def test_bleu():
	base_test_metrics(
		bleu,
		[51.1507, 23.5783, 100.0],
		similarity_generation_metric_inputs,
		lowercase=True,
	)


def test_meteor():
	base_test_metrics(
		meteor,
		[0.454033, 0.2985435, 0.64077828],
		similarity_generation_metric_inputs,
		alpha=0.85,
		beta=0.2,
		gamma=0.6,
	)


def test_rouge():
	base_test_metrics(rouge, [0.909, 0.35714, 1.0], similarity_generation_metric_inputs)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions. It use local model",
)
def test_sem_score():
	base_test_metrics(
		sem_score, [0.9005998, 0.7952, 1.0], similarity_generation_metric_inputs
	)


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_sem_score_other_model():
	scores = sem_score(
		metric_inputs=similarity_generation_metric_inputs,
		embedding_model=OpenAIEmbedding(),
	)
	assert len(scores) == len(generation_gts)
	assert all(isinstance(score, float) for score in scores)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_g_eval_fluency():
	base_test_metrics(
		g_eval,
		[3.0, 2.0, 2.0],
		similarity_generation_metric_inputs,
		metrics=["fluency"],
		model="gpt-3.5-turbo",
	)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_g_eval_full():
	base_test_metrics(
		g_eval,
		[3.5, 2.75, 2.0],
		similarity_generation_metric_inputs,
		model="gpt-4o-mini",
	)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_bert_score_en():
	base_test_metrics(
		bert_score,
		[0.981902, 0.93164, 1.0],
		similarity_generation_metric_inputs,
		n_threads=8,
	)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_bert_score_ko():
	base_test_metrics(
		bert_score,
		[1.0, 0.965312, 0.96309],
		ko_similarity_generation_metric_inputs,
		lang="ko",
	)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_bert_score_ja():
	base_test_metrics(
		bert_score,
		[1.0, 0.82659, 1.0],
		ja_similarity_generation_metric_inputs,
		lang="ja",
	)
