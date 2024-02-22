from typing import List

from autorag.nodes.passagecompressor.base import passage_compressor_node


@passage_compressor_node
def pass_compressor(contents: List[List[str]]):
    """Do not perform any passage compression"""
    return contents
