class FaithfulnessTemplate:
	@staticmethod
	def generate_claims(text, lang: str = "en"):
		if lang == "en":
			return f"""Based on the given text, please generate a comprehensive list of FACTUAL claims that can inferred from the provided text.

	Example:
	Example Text:
	"Einstein won the noble prize in 1968 for his discovery of the photoelectric effect."

	Example JSON:
	{{
	    "claims": [
	        "Einstein won the noble prize for his discovery of the photoelectric effect.",
	        "Einstein won the noble prize in 1968."
	    ]
	}}
	===== END OF EXAMPLE ======

	**
	IMPORTANT: Please make sure to only return in JSON format, with the "claims" key as a list of strings. No words or explanation is needed.
	Only include claims that are factual, and the claims you extract should include the full context it was presented in, NOT cherry picked facts.
	You should NOT include any prior knowledge, and take the text at face value when extracting claims.
	**

	Text:
	{text}

	JSON:
	"""
		elif lang == "ko":
			return f"""주어진 텍스트에서 찾을 수 있는 사실적 정보들의 목록을 생성하세요.

예시:
예시 텍스트:
“아인슈타인은 1968년에 광전 효과 발견으로 노벨상을 수상했다.”

예시 JSON:
{{
“claims”: [
“아인슈타인은 광전 효과 발견으로 노벨상을 수상했다.”,
“아인슈타인은 1968년에 노벨상을 수상했다.”
]
}}
===== 예시 끝 ======

**
중요: 오직 JSON 형식으로 “claims” 키가 문자열 목록으로 반환되도록 해야 합니다. 다른 단어나 설명은 필요하지 않습니다.
사실에 기반한 주장만 포함하며, 추출한 주장은 전체 맥락을 유지해야 하며, 부분적으로 선택된 사실을 포함하지 않아야 합니다.
사전 지식은 포함하지 말고, 텍스트에만 기초해 주장들을 추출해야 합니다.
**

텍스트:
{text}

JSON:
"""
		elif lang == "ja":
			return f"""与えられたテキストに基づいて、そこから推測できる事実に基づく主張のリストを生成してください。

例:
例のテキスト:
「アインシュタインは1968年に光電効果の発見でノーベル賞を受賞しました。」

例のJSON:
{{
    "claims": [
        "アインシュタインは光電効果の発見でノーベル賞を受賞しました。",
        "アインシュタインは1968年にノーベル賞を受賞しました。"
    ]
}}
===== 例の終わり ======

**
重要: 必ずJSON形式で"claims"キーが文字列のリストとして返されるようにしてください。説明や余計な言葉は不要です。
事実に基づく主張のみを含め、抽出された主張は提示された文脈全体を含むものでなければなりません。一部の事実のみを抜粋することは避けてください。
事前知識を使用せず、テキストに基づいて主張を抽出してください。
**

テキスト:
{text}

JSON:
"""
		else:
			raise ValueError(f"Language {lang} is not supported.")

	@staticmethod
	def generate_truths(text, lang: str = "en"):
		if lang == "en":
			return f"""Based on the given text, please generate a comprehensive list of FACTUAL, undisputed truths that can inferred from the provided text.

	Example:
	Example Text:
	"Einstein won the noble prize in 1968 for his discovery of the photoelectric effect."

	Example JSON:
	{{
	    "truths": [
	        "Einstein won the noble prize for his discovery of the photoelectric effect.",
	        "Einstein won the noble prize in 1968."
	    ]
	}}
	===== END OF EXAMPLE ======

	**
	IMPORTANT: Please make sure to only return in JSON format, with the "truths" key as a list of strings. No words or explanation is needed.
	Only include truths that are factual.
	**

	Text:
	{text}

	JSON:
	"""
		elif lang == "ko":
			return f"""주어진 텍스트에서 추출할 수 있는 사실적이고 논란이 없는 진실들의 목록을 생성하세요.

예시:
예시 텍스트:
"아인슈타인은 1968년에 광전 효과 발견으로 노벨상을 수상했다."

예시 JSON:
{{
    "truths": [
        "아인슈타인은 광전 효과 발견으로 노벨상을 수상했다.",
        "아인슈타인은 1968년에 노벨상을 수상했다."
    ]
}}
===== 예시 끝 ======

**
중요: 오직 JSON 형식으로 "truths" 키가 문자열 목록으로 반환되도록 해야 합니다. 다른 단어나 설명은 필요하지 않습니다.
사실에 기반한 진실만 포함해야 합니다.
**

텍스트:
{text}

JSON:
"""
		elif lang == "ja":
			return f"""与えられたテキストに基づいて、そこから推測できる事実で議論の余地のない真実のリストを生成してください。

例:
例のテキスト:
「アインシュタインは1968年に光電効果の発見でノーベル賞を受賞しました。」

例のJSON:
{{
    "truths": [
        "アインシュタインは光電効果の発見でノーベル賞を受賞しました。",
        "アインシュタインは1968年にノーベル賞を受賞しました。"
    ]
}}
===== 例の終わり ======

**
重要: 必ずJSON形式で"truths"キーが文字列のリストとして返されるようにしてください。説明や余計な言葉は不要です。
事実に基づく真実のみを含めてください。
**

テキスト:
{text}

JSON:
"""
		else:
			raise ValueError(f"Language {lang} is not supported.")

	@staticmethod
	def generate_verdicts(claims, retrieval_context, lang: str = "en"):
		if lang == "en":
			return f"""Based on the given claims, which is a list of strings, generate a list of JSON objects to indicate whether EACH claim contradicts any facts in the retrieval context. The JSON will have 2 fields: 'verdict' and 'reason'.
	The 'verdict' key should STRICTLY be either 'yes', 'no', or 'idk', which states whether the given claim agrees with the context.
	Provide a 'reason' ONLY if the answer is 'no'.
	The provided claim is drawn from the actual output. Try to provide a correction in the reason using the facts in the retrieval context.

	**
	IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
	Example retrieval contexts: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
	Example claims: ["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a Germen chef."]

	Example:
	{{
	    "verdicts": [
	        {{
	            "verdict": "idk"
	        }},
	        {{
	            "verdict": "idk"
	        }},
	        {{
	            "verdict": "yes"
	        }},
	        {{
	            "verdict": "no",
	            "reason": "The actual output claims Einstein won the Nobel Prize in 1969, which is untrue as the retrieval context states it is 1968 instead."
	        }},
	        {{
	            "verdict": "no",
	            "reason": "The actual output claims Einstein is a Germen chef, which is not correct as the retrieval context states he was a German scientist instead."
	        }},
	    ]
	}}
	===== END OF EXAMPLE ======

	The length of 'verdicts' SHOULD BE STRICTLY EQUAL to that of claims.
	You DON'T have to provide a reason if the answer is 'yes' or 'idk'.
	ONLY provide a 'no' answer if the retrieval context DIRECTLY CONTRADICTS the claims. YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
	Claims made using vague, suggestive, speculative language such as 'may have', 'possibility due to', does NOT count as a contradiction.
	Claims that is not backed up due to a lack of information/is not mentioned in the retrieval contexts MUST be answered 'idk', otherwise I WILL DIE.
	**

	Retrieval Contexts:
	{retrieval_context}

	Claims:
	{claims}

	JSON:
	"""
		elif lang == "ko":
			return f"""주어진 주장에 대해, 각 주장이 주어진 문맥의 사실들과 모순되는지를 나타내는 JSON 객체 목록을 생성하세요. JSON은 두 개의 필드인 'verdict'와 'reason'으로 구성됩니다.
'verdict'는 'yes', 'no', 또는 'idk' 중 하나여야 하며, 주어진 주장이 문맥과 일치하는지를 나타냅니다.
'verdict'가 'no'인 경우에만 'reason'을 제공하세요. 'reason'에는 문맥에 따라 주장을 수정하는 내용이 포함되어야 합니다.

**
중요: 오직 JSON 형식으로 'verdicts' 키가 JSON 객체 목록으로 반환되도록 해야 합니다.
예시 문맥: "아인슈타인은 광전 효과 발견으로 노벨상을 수상했다. 아인슈타인은 1968년에 노벨상을 수상했다. 아인슈타인은 독일 과학자이다."
예시 주장: ["버락 오바마는 백인 남성이다.", "취리히는 런던에 있는 도시이다.", "아인슈타인은 광전 효과 발견으로 노벨상을 수상했으며, 이는 그의 명성에 기여했을 것이다.", "아인슈타인은 1969년에 광전 효과 발견으로 노벨상을 수상했다.", "아인슈타인은 독일 요리사였다."]

예시:
{{
    "verdicts": [
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "실제 출력은 아인슈타인이 1969년에 노벨상을 수상했다고 주장하지만, 문맥에서는 1968년이라고 명시되어 있습니다."
        }},
        {{
            "verdict": "no",
            "reason": "실제 출력은 아인슈타인이 독일 요리사라고 주장하지만, 문맥에서는 그가 독일 과학자라고 명시되어 있습니다."
        }},
    ]
}}
===== 예시 끝 ======

'verdicts' 리스트의 길이는 반드시 주장들의 길이와 같아야 합니다.
'yes' 또는 'idk'일 경우 'reason'을 제공할 필요가 없습니다.
검색된 문맥과 직접적으로 모순되는 경우에만 'no' 답변을 제공하세요. 절대로 선험적인 지식을 사용하지 마세요.
'~일 수 있다', '가능성이 있다'와 같은 모호한 표현은 모순으로 간주하지 마세요.
문맥에 대한 정보 부족으로 뒷받침되지 않거나 언급되지 않은 주장은 반드시 'idk'로 답변하세요, 그렇지 않으면 내가 죽습니다.
**

주어진 문맥:
{retrieval_context}

주장:
{claims}

JSON:
"""
		elif lang == "ja":
			return f"""与えられた主張について、それぞれの主張が取得された文脈の事実と矛盾しているかどうかを示すJSONオブジェクトのリストを生成してください。JSONには2つのフィールド、'verdict'と'reason'があります。
'verdict'フィールドは、主張が文脈に一致するかどうかを示すため、厳密に'yes', 'no', 'idk'のいずれかを使用します。
'verdict'が'no'の場合にのみ、'reason'を提供してください。'reason'には、文脈に基づいて主張を修正する内容が含まれている必要があります。

**
重要: 必ずJSON形式で'verdicts'キーがJSONオブジェクトのリストとして返されるようにしてください。
例の文脈:「アインシュタインは光電効果の発見でノーベル賞を受賞しました。アインシュタインは1968年にノーベル賞を受賞しました。アインシュタインはドイツの科学者です。」
例の主張: ["バラク・オバマは白人男性です。", "チューリッヒはロンドンにある都市です。", "アインシュタインは光電効果の発見でノーベル賞を受賞し、これが彼の名声に貢献したかもしれません。", "アインシュタインは1969年に光電効果の発見でノーベル賞を受賞しました。", "アインシュタインはドイツのシェフでした。"]

例のJSON:
{{
    "verdicts": [
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "実際の出力は、アインシュタインが1969年にノーベル賞を受賞したと主張していますが、文脈では1968年と述べられています。"
        }},
        {{
            "verdict": "no",
            "reason": "実際の出力は、アインシュタインがドイツのシェフだと主張していますが、文脈では彼がドイツの科学者であると述べられています。"
        }},
    ]
}}
===== 例の終わり ======

'verdicts'のリストの長さは、主張のリストの長さと必ず等しくなければなりません。
'yes'または'idk'の場合、'reason'を提供する必要はありません。
文脈と直接矛盾する場合にのみ、'no'を提供してください。決して事前知識を使用しないでください。
「〜かもしれない」や「〜の可能性がある」といった曖昧な表現は矛盾とは見なされません。
情報が不足している、または文脈で言及されていない主張には必ず'idk'で答えてください。さもないと私は死んでしまいます。
**

文脈:
{retrieval_context}

主張:
{claims}

JSON:
"""
		else:
			raise ValueError(f"Language {lang} is not supported.")
