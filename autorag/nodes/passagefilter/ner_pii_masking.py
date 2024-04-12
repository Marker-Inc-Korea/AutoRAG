import asyncio
from typing import List, Tuple

from transformers import pipeline

from autorag.nodes.passagefilter.base import passage_filter_node


@passage_filter_node
def ner_pii_masking(contents_list: List[List[str]],
                    scores_list: List[List[float]], ids_list: List[List[str]],
                    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Mask PII in the contents using NER.
    Use a Hugging Face NER model for PII Masking

    :param contents_list: The list of lists of contents to filter
    :param scores_list: The list of lists of scores retrieved
    :param ids_list: The list of lists of ids retrieved
    :return: Tuple of lists containing the filtered contents, ids, and scores
    """
    model = pipeline("ner", grouped_entities=True)

    tasks = [mask_pii(model, contents) for contents in contents_list]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    masked_contents_list = list(results)

    return masked_contents_list, ids_list, scores_list


async def mask_pii(model, contents: List[str]) -> List[str]:
    new_contents_list = []
    for content in contents:
        new_contents = content
        response = model(content)
        for entry in response:
            entity_group_tag = f"[{entry['entity_group']}_{entry['start']}]"
            new_contents = new_contents.replace(entry["word"], entity_group_tag).strip()
        new_contents_list.append(new_contents)
    return new_contents_list
