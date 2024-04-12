from typing import List, Tuple

from transformers import pipeline

from autorag.nodes.passagefilter.base import passage_filter_node


@passage_filter_node
def ner_pii_masking(contents_list: List[List[str]],
                    scores_list: List[List[float]], ids_list: List[List[str]],
                    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Mask PII in the contents using NER.
    Uses HF transformers model.

    :param contents_list: The list of lists of contents to filter
    :param scores_list: The list of lists of scores retrieved
    :param ids_list: The list of lists of ids retrieved
    :return: Tuple of lists containing the filtered contents, ids, and scores
    """
    model = pipeline("ner", grouped_entities=True)

    masked_contents_list = list(
        map(lambda contents: list(map(lambda content: mask_pii(model, content), contents)), contents_list))

    return masked_contents_list, ids_list, scores_list


def mask_pii(model, text: str) -> str:
    new_text = text
    response = model(text)
    for entry in response:
        entity_group_tag = f"[{entry['entity_group']}_{entry['start']}]"
        new_text = new_text.replace(entry["word"], entity_group_tag).strip()
    return new_text
