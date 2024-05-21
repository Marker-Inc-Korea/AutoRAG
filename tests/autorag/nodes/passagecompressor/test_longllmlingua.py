from autorag.nodes.passagecompressor import longllmlingua
from tests.autorag.nodes.passagecompressor.test_base_passage_compressor import (queries, retrieved_contents,
                                                                                check_result)


def test_longllmlingua_default():
    result = longllmlingua.__wrapped__(queries, retrieved_contents, [], [])
    print(result)
    check_result(result)
