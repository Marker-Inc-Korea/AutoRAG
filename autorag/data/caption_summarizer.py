import asyncio
import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional, Sequence


import click
from dotenv import load_dotenv
import pandas as pd


logger = logging.getLogger("AutoRAG")


def _read_image_b64(path: Path) -> tuple[str, str]:
	"""Read image bytes and return (media_type, base64_str)."""
	if not path.exists() or not path.is_file():
		raise FileNotFoundError(f"Image not found: {path}")
	media_type, _ = mimetypes.guess_type(str(path))
	if media_type is None:
		# Fallback to octet-stream if unknown
		media_type = "application/octet-stream"
	data = path.read_bytes()
	b64 = base64.b64encode(data).decode("utf-8")
	return media_type, b64


async def _asummarize_with_anthropic(
	*,
	image_path: Path,
	caption: str,
	model: str,
	max_tokens: int,
	temperature: float,
	max_retries: int = 3,
	retry_initial_delay: float = 1.0,
) -> str:
	try:
		from anthropic import AsyncAnthropic
	except Exception as exc:  # pragma: no cover
		raise ImportError(
			"anthropic package is required. Install with `uv pip install anthropic` or ensure it is in your env."
		) from exc

	api_key = os.getenv("ANTHROPIC_API_KEY")
	if not api_key:
		raise EnvironmentError(
			"ANTHROPIC_API_KEY is not set. Please export it to use this command."
		)

	# Local file IO; okay to do sync here
	media_type, b64 = _read_image_b64(image_path)

	client = AsyncAnthropic(api_key=api_key)

	system_prompt = (
		"You are an assistant that writes concise, faithful summaries of an image "
		"given the image itself and a human-provided caption. Maintain key details, "
		"remove redundancy.\n\n모든 문서와 이미지는 한국어이므로 한국어로 요약을 제공하세요."
	)
	user_text = (
		"Summarize the following image and caption.\n반드시 한국어로 요약문을 제공하세요.\n\n"
		f"Caption: {caption}"
	)

	delay = max(0.0, float(retry_initial_delay))
	for attempt in range(1, max(1, int(max_retries)) + 1):
		try:
			msg = await client.messages.create(
				model=model,
				max_tokens=max_tokens,
				temperature=temperature,
				system=system_prompt,
				messages=[
					{
						"role": "user",
						"content": [
							{"type": "text", "text": user_text},
							{
								"type": "image",
								"source": {
									"type": "base64",
									"media_type": media_type,
									"data": b64,
								},
							},
						],
					}
				],
			)

			parts: Sequence = getattr(msg, "content", []) or []
			for part in parts:
				if getattr(part, "type", None) == "text" and getattr(
					part, "text", None
				):
					text = str(part.text).strip()
					if text:
						return text
				if isinstance(part, dict) and part.get("type") == "text":
					text = str(part.get("text", "")).strip()
					if text:
						return text

			# Treat empty/none content as transient failure to trigger retry
			raise RuntimeError("Anthropic returned no text content")
		except Exception as e:  # pragma: no cover
			logger.warning(
				"[attempt %s/%s] Summarization API error for %s: %s",
				attempt,
				max_retries,
				image_path.name,
				e,
			)
			if attempt < max_retries:
				await asyncio.sleep(delay)
				delay = min(delay * 2.0, 30.0)
				continue
			# After final attempt, return explicit error marker (not empty)
			return "[ERROR: no content returned after retries]"


@click.command(name="summarize-captions")
@click.option(
	"--input-csv",
	type=click.Path(exists=True, dir_okay=False, path_type=Path),
	required=True,
	help="Path to input CSV containing 'image' and 'Human_Caption' columns.",
)
@click.option(
	"--output-csv",
	type=click.Path(dir_okay=False, path_type=Path),
	required=True,
	help="Path to write CSV with added 'Summarization' column.",
)
@click.option(
	"--images-dir",
	type=click.Path(file_okay=False, exists=False, path_type=Path),
	default=None,
	help="Directory containing image files. Defaults to input CSV's directory.",
)
@click.option(
	"--image-col",
	default="image",
	show_default=True,
	help="Image filename column name.",
)
@click.option(
	"--caption-col",
	default="Human_Caption",
	show_default=True,
	help="Caption column name.",
)
@click.option(
	"--model",
	default="claude-sonnet-4-5",
	show_default=True,
	help="Anthropic Claude model to use.",
)
@click.option("--max-tokens", default=256, show_default=True, type=int)
@click.option("--temperature", default=0.2, show_default=True, type=float)
@click.option("--limit", default=None, type=int, help="Limit rows processed (debug).")
@click.option(
	"--concurrency",
	default=8,
	show_default=True,
	type=int,
	help="Max concurrent requests to Anthropic API.",
)
def cli(
	input_csv: Path,
	output_csv: Path,
	images_dir: Optional[Path],
	image_col: str,
	caption_col: str,
	model: str,
	max_tokens: int,
	temperature: float,
	limit: Optional[int],
	concurrency: int,
):
	"""Summarize images with their human captions using Anthropic Claude.


	Reads CSV, loads images, calls Anthropic API, and writes a new CSV
	with a 'Summarization' column.
	"""
	api_key = os.getenv("ANTHROPIC_API_KEY")
	if not api_key:
		raise EnvironmentError(
			"ANTHROPIC_API_KEY is not set. Please export it to use this command."
		)

	df = pd.read_csv(input_csv)
	if image_col not in df.columns or caption_col not in df.columns:
		raise ValueError(
			f"CSV must contain columns '{image_col}' and '{caption_col}'. Found: {list(df.columns)}"
		)

	base_dir = images_dir or input_csv.parent

	async def _run_async():
		sem = asyncio.Semaphore(max(1, int(concurrency)))

		results: list[str] = [""] * len(df)

		async def _worker(idx: int, image_name: str, caption: str):
			image_path = base_dir / str(image_name)
			try:
				async with sem:
					summary = await _asummarize_with_anthropic(
						image_path=image_path,
						caption=str(caption),
						model=model,
						max_tokens=max_tokens,
						temperature=temperature,
						max_retries=3,
						retry_initial_delay=1.0,
					)
				results[idx] = summary or "[ERROR: empty content]"
			except Exception as e:  # pragma: no cover
				logger.warning("Summarization failed for %s: %s", image_name, e)
				results[idx] = "[ERROR: exception]"

		tasks = []
		rows = list(df.itertuples(index=False))
		if limit is not None:
			rows = rows[:limit]
		for i, row in enumerate(rows):
			tasks.append(
				asyncio.create_task(
					_worker(i, getattr(row, image_col), getattr(row, caption_col))
				)
			)
		for i in range(0, len(tasks), 100):
			# small chunks to yield control regularly
			await asyncio.gather(*tasks[i : i + 100])
			click.echo(f"Processed {min(i + 100, len(tasks))}/{len(tasks)} rows...")

		return results

	summarizations = asyncio.run(_run_async())
	# If limit used, pad out to dataframe size for safe assignment
	if limit is not None and len(summarizations) != len(df):
		summarizations = [""] * (len(df) - len(summarizations)) + summarizations
	df["Summarization"] = summarizations

	output_csv.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_csv, index=False)
	click.echo(f"Saved summarized CSV to {output_csv}")


if __name__ == "__main__":  # pragma: no cover
	load = load_dotenv()
	if not load:
		raise RuntimeError(".env file could not be loaded")
	cli()
