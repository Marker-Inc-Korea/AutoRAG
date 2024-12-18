import logging
from typing import List, Tuple
import time

import pandas as pd
import requests
from asyncio import to_thread

from autorag.nodes.generator.base import BaseGenerator
from autorag.utils.util import get_event_loop, process_batch, result_to_dataframe

logger = logging.getLogger("AutoRAG")

DEFAULT_MAX_TOKENS = 4096  # 기본 토큰 제한값

class VllmAPI(BaseGenerator):
    def __init__(
        self, project_dir, llm: str, uri: str, max_tokens: int = None, batch: int = 16, *args, **kwargs
    ):
        """
        VLLM API Wrapper for OpenAI-compatible chat/completions format.

        :param project_dir: 프로젝트 디렉토리.
        :param llm: 모델 이름 (예: LLaMA 모델).
        :param uri: VLLM API 서버 uri.
        :param max_tokens: 최대 토큰 제한값 (입력값을 우선 사용, 없으면 기본값 사용).
        :param batch: 요청 배치 크기.
        """
        super().__init__(project_dir, llm, *args, **kwargs)
        assert batch > 0, "Batch size must be greater than 0."
        self.uri = uri.rstrip("/")  # API uri 설정
        self.batch = batch
        # max_tokens가 입력으로 제공되면 이를 사용하고, 없으면 기본값 사용
        self.max_token_size = max_tokens if max_tokens else DEFAULT_MAX_TOKENS

    @result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
    def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
        prompts = self.cast_to_run(previous_result)
        return self._pure(prompts, **kwargs)

    def _pure(
        self, prompts: List[str], truncate: bool = True, **kwargs
    ) -> Tuple[List[str], List[List[int]], List[List[float]]]:
        """
        VLLM API를 호출하여 텍스트를 생성하는 메서드.

        :param prompts: 입력 프롬프트 리스트.
        :param truncate: 입력 프롬프트를 토큰 제한 내에서 자를지 여부.
        :param kwargs: 추가 옵션 (temperature, top_p 등).
        :return: 생성된 텍스트, 토큰 리스트, 로그 확률 리스트.
        """
        if kwargs.get("logprobs") is not None:
            kwargs.pop("logprobs")
            logger.warning(
                "parameter logprob does not effective. It always set to True."
            )
        if kwargs.get("n") is not None:
            kwargs.pop("n")
            logger.warning("parameter n does not effective. It always set to 1.")
   
        if truncate:
            prompts = list(map(lambda p: self.truncate_by_token(p), prompts))
        loop = get_event_loop()
        tasks = [ 
            to_thread(self.get_result, prompt, **kwargs) for prompt in prompts 
        ]
        results = loop.run_until_complete(process_batch(tasks, self.batch))
        
        answer_result = list(map(lambda x: x[0], results))
        token_result = list(map(lambda x: x[1], results))
        logprob_result = list(map(lambda x: x[2], results))
        return answer_result, token_result, logprob_result

    def truncate_by_token(self, prompt: str) -> str:
        """
        프롬프트를 최대 토큰 제한값 내에서 자르는 함수.
        """
        tokens = self.encoding_for_model(prompt)['tokens']  # 간단한 토큰화
        return self.decoding_for_model(tokens[:self.max_token_size])['prompt']
    def call_vllm_api(self, prompt: str, **kwargs) -> dict:
        """
        VLLM API를 호출하여 chat/completions 응답을 가져옵니다.

        :param prompt: 입력 프롬프트.
        :param kwargs: API 추가 옵션 (temperature, max_tokens 등).
        :return: API 응답 결과.
        """
        payload = {
            "model": self.llm,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.4),
            "max_tokens": min(kwargs.get("max_tokens", self.max_token_size), self.max_token_size),
            "logprobs": True,
            "n": 1
        }
        start_time = time.time()  # 요청 시작 시간 기록
        response = requests.post(f"{self.uri}/v1/chat/completions", json=payload)
        end_time = time.time()  # 요청 종료 시간 기록

        response.raise_for_status()
        elapsed_time = end_time - start_time  # 소요 시간 계산
        logger.info(f"Request chat completions to vllm server completed in {elapsed_time:.2f} seconds")
        return response.json()

    # 추가된 메서드: abstract 메서드 구현
    async def astream(self, prompt: str, **kwargs):
        """
        비동기 스트림 방식 미구현.
        """
        raise NotImplementedError("astream method is not implemented for VLLM API yet.")

    def stream(self, prompt: str, **kwargs):
        """
        동기 스트림 방식 미구현.
        """
        raise NotImplementedError("stream method is not implemented for VLLM API yet.")
    def get_result(self, prompt: str, **kwargs):
      response = self.call_vllm_api(prompt, **kwargs)
      choice = response['choices'][0]
      answer = choice['message']['content']

			# logprobs가 None인 경우를 처리 
      if choice.get('logprobs') and 'content' in choice['logprobs']:
        logprobs = list(map(lambda x: x['logprob'], choice['logprobs']['content']))
        tokens = list(
					map(
						lambda x: self.encoding_for_model(x['token'])['tokens'],
						choice['logprobs']['content'],
					)
				)
      else:
        logprobs = []
        tokens = []

      return answer, tokens, logprobs
    
    def encoding_for_model(self, answer_piece: str):
      payload = {
					"model": self.llm,
					"prompt": answer_piece,
					"add_special_tokens": True
			}
      response = requests.post(f"{self.uri}/tokenize", json=payload)
      response.raise_for_status()
      return response.json()
    def decoding_for_model(self, tokens: list[int]):
      payload = {
					"model": self.llm,
					"tokens": tokens,
			}
      response = requests.post(f"{self.uri}/detokenize", json=payload)
      response.raise_for_status()
      return response.json()