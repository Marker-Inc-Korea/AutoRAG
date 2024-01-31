import pandas as pd
import pytest
from transformers import AutoTokenizer

from autorag.evaluate.generation import evaluate_generation

generation_gts = [
    ['The dog had bit the man.', 'The man had bitten the dog.'],
    ['I want to be a artist, but I end up to be a programmer.'],
    ['To be a artist these days, you can overcome by AI.',
     'To be a programmer these days, you can overcome by AI.',
     'To be a lawyer these days, you can overcome by AI.'],
]
pseudo_generations = [
    'The dog bit the man.',
    'It really like to be a programmer, but I think artist is my passion.',
    'To be a artist these days, you can overcome by AI.',
]

tokenizer = AutoTokenizer.from_pretrained('gpt2')
pseudo_tokens = list(map(lambda x: tokenizer.tokenize(x), pseudo_generations))
pseudo_log_probs = list(map(lambda x: [0.1] * len(x), pseudo_tokens))


@evaluate_generation(generation_gt=generation_gts, metrics=['bleu', 'meteor', 'rouge'])
def pseudo_generation():
    return pseudo_generations


@evaluate_generation(generation_gt=generation_gts, metrics=['bleu', 'meteor', 'donggeon_metric'])
def pseudo_generation_with_log_probs():
    return pseudo_generations, pseudo_tokens, pseudo_log_probs


def test_evaluate_generation():
    result_df = pseudo_generation()
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 3
    assert len(result_df.columns) == 4
    assert set(result_df.columns) == {'generated_texts', 'bleu', 'meteor', 'rouge'}

    with pytest.warns():
        result_df_log_probs = pseudo_generation_with_log_probs()
    assert isinstance(result_df_log_probs, pd.DataFrame)
    assert len(result_df_log_probs) == 3
    assert len(result_df_log_probs.columns) == 5
    assert set(result_df_log_probs.columns) == {'generated_texts', 'bleu', 'meteor', 'generated_tokens',
                                                'generated_log_probs'}

    assert result_df_log_probs['generated_texts'].tolist() == pseudo_generations
    assert result_df_log_probs['generated_tokens'].tolist() == pseudo_tokens
    assert result_df_log_probs['generated_log_probs'].tolist() == pseudo_log_probs
    assert all(list(map(lambda x: x[0] == pytest.approx(x[1], 0.001),
                        zip(result_df['bleu'].tolist(), [51.1507, 23.5783, 100.0]))))
    assert all(list(map(lambda x: x[0] == pytest.approx(x[1], 0.001),
                        zip(result_df['meteor'].tolist(), [0.853462, 0.5859375, 1.0]))))
    assert all(list(map(lambda x: x[0] == pytest.approx(x[1], 0.001),
                        zip(result_df['rouge'].tolist(), [0.909, 0.35714, 1.0]))))
