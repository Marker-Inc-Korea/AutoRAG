from typing import List

from autorag.nodes.promptmaker.base import prompt_maker_node


@prompt_maker_node
def fstring(prompt: str,
            queries: List[str], retrieved_contents: List[List[str]]) -> List[str]:
    """
    Make a prompt using f-string from a query and retrieved_contents.
    You must type a prompt or prompt list at config yaml file like this:

    .. Code:: yaml
    nodes:
    - node_type: prompt_maker
      modules:
      - module_type: fstring
        prompt: [Answer this question: {query} \n\n {retrieved_contents},
        Read the passages carefully and answer this question: {query} \n\n Passages: {retrieved_contents}]

    :param prompt: A prompt string.
    :param queries: List of query strings.
    :param retrieved_contents: List of retrieved contents.
    :return: Prompts that made by f-string.
    """
    def fstring_row(_prompt: str, _query: str, _retrieved_contents: List[str]) -> str:
        contents_str = "\n\n".join(_retrieved_contents)
        return _prompt.format(query=_query, retrieved_contents=contents_str)

    return list(map(lambda x: fstring_row(prompt, x[0], x[1]), zip(queries, retrieved_contents)))
