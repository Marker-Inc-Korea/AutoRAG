from unittest.mock import patch

from ragas.metrics import ContextPrecision

from autorag.evaluate.metric.retrieval_no_gt import ragas_context_precision

queries_example = ["What is the capital of France?",
                   "How many members are in Newjeans?"]
contents_example = [["NomaDamas is Great Team", "Paris is the capital of France.", "havertz is suck at soccer",
                     "Paris is one of the capital from France. Isn't it?"],
                    ["i am hungry", "LA is a country in the United States.", "Newjeans has 5 members.",
                     "Danielle is one of the members of Newjeans."]]
generation_gt_example = [["Paris is the capital of France."], ["New jeans has total five members, including Danielle."]]


async def mock_context_precision_ascore(self, row, callbacks, is_async) -> float:
    return 0.3


@patch.object(ContextPrecision, "_ascore", mock_context_precision_ascore)
def test_ragas_context_precision():
    scores = ragas_context_precision(queries_example, contents_example, generation_gt_example)
    assert isinstance(scores, list)
    assert len(scores) == len(queries_example)
    assert isinstance(scores[0], float)
    assert scores == [0.3, 0.3]
