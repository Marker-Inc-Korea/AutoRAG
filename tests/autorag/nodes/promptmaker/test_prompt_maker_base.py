import pandas as pd

prompt = "Answer this question: {query} \n\n {retrieved_contents}"
queries = ["What is the capital of Japan?", "What is the capital of China?"]
retrieved_contents = [
    ["Tokyo is the capital of Japan.", "Tokyo, the capital of Japan, is a huge metropolitan city."],
    ["Beijing is the capital of China.", "Beijing, the capital of China, is a huge metropolitan city."]]
retrieve_scores = [[0.9, 0.8], [0.9, 0.8]]
previous_result = pd.DataFrame({
    "query": queries,
    "retrieved_contents": retrieved_contents,
    "retrieve_scores": retrieve_scores
})
