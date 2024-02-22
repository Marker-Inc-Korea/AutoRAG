from typing import List

from autorag.nodes.queryexpansion.base import query_expansion_node


@query_expansion_node
def pass_query_expansion(queries: List[str]):
    """
    Do not perform query expansion.
    Return with the same queries.
    The dimension will be 2-d list, and the column name will be 'queries'.
    """
    return list(map(lambda x: [x], queries))
