import pandas as pd

passage1 = """NewJeans (뉴진스) is a 5-member girl group under ADOR and HYBE Labels.
The members consist of Minji, Hanni, Danielle, Haerin, and Hyein.
They released their debut single “Attention” on July 22, 2022,
followed by their debut extended play, New Jeans, which was released on August 1, 2022."""
passage2 = """The digital age has transformed the way we live, work, and interact, bringing both opportunities and challenges.
With the rise of artificial intelligence, automation, and data-driven technologies, industries are evolving at an unprecedented pace.
While these advancements enhance efficiency and innovation,
they also raise ethical concerns around privacy, employment, and the role of human decision-making."""

qa_df = pd.DataFrame(
	{
		"qid": ["jax1", "jax2"],
		"query": [
			"When is New Jeans debut day?",
			"What are the challenges of the digital age?",
		],
		"retrieval_gt": [[["havertz1"]], [["havertz2"]]],
		"retrieval_gt_contents": [
			[[passage1]],
			[[passage2]],
		],
	}
)
