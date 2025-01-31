import ast
import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
import seaborn as sns
import yaml
from bokeh.models import NumberFormatter, BooleanFormatter

from autorag.utils.util import dict_to_markdown, dict_to_markdown_table

pn.extension(
	"terminal",
	"tabulator",
	"mathjax",
	"ipywidgets",
	console_output="disable",
	sizing_mode="stretch_width",
	css_files=[
		"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css"
	],
)
logger = logging.getLogger("AutoRAG")


def find_node_dir(trial_dir: str) -> List[str]:
	trial_summary_df = pd.read_csv(os.path.join(trial_dir, "summary.csv"))
	result_paths = []
	for idx, row in trial_summary_df.iterrows():
		node_line_name = row["node_line_name"]
		node_type = row["node_type"]
		result_paths.append(os.path.join(trial_dir, node_line_name, node_type))
	return result_paths


def get_metric_values(node_summary_df: pd.DataFrame) -> Dict:
	non_metric_column_names = [
		"filename",
		"module_name",
		"module_params",
		"execution_time",
		"average_output_token",
		"is_best",
	]
	best_row = node_summary_df.loc[node_summary_df["is_best"]].drop(
		columns=non_metric_column_names, errors="ignore"
	)
	assert len(best_row) == 1, "The best module must be only one."
	return best_row.iloc[0].to_dict()


def make_trial_summary_md(trial_dir):
	markdown_text = f"""# Trial Result Summary
- Trial Directory : {trial_dir}

"""
	node_dirs = find_node_dir(trial_dir)
	for node_dir in node_dirs:
		node_summary_filepath = os.path.join(node_dir, "summary.csv")
		node_type = os.path.basename(node_dir)
		node_summary_df = pd.read_csv(node_summary_filepath)
		best_row = node_summary_df.loc[node_summary_df["is_best"]].iloc[0]
		metric_dict = get_metric_values(node_summary_df)
		markdown_text += f"""---

## {node_type} best module

### Module Name

{best_row['module_name']}

### Module Params

{dict_to_markdown(ast.literal_eval(best_row['module_params']), level=3)}

### Metric Values

{dict_to_markdown_table(metric_dict, key_column_name='metric_name', value_column_name='metric_value')}

"""

	return markdown_text


def node_view(node_dir: str):
	non_metric_column_names = [
		"filename",
		"module_name",
		"module_params",
		"execution_time",
		"average_output_token",
		"is_best",
	]
	summary_df = pd.read_csv(os.path.join(node_dir, "summary.csv"))
	bokeh_formatters = {
		"float": NumberFormatter(format="0.000"),
		"bool": BooleanFormatter(),
	}
	first_df = pd.read_parquet(os.path.join(node_dir, "0.parquet"), engine="pyarrow")

	each_module_df_widget = pn.widgets.Tabulator(
		pd.DataFrame(columns=first_df.columns),
		name="Module DataFrame",
		formatters=bokeh_formatters,
		pagination="local",
		page_size=20,
		widths=150,
	)

	def change_module_widget(event):
		if event.column == "detail":
			filename = summary_df["filename"].iloc[event.row]
			filepath = os.path.join(node_dir, filename)
			each_module_df = pd.read_parquet(filepath, engine="pyarrow")
			each_module_df_widget.value = each_module_df

	df_widget = pn.widgets.Tabulator(
		summary_df,
		name="Summary DataFrame",
		formatters=bokeh_formatters,
		buttons={"detail": '<i class="fa fa-eye"></i>'},
		widths=150,
	)
	df_widget.on_click(change_module_widget)

	try:
		fig, ax = plt.subplots(figsize=(10, 5))
		metric_df = summary_df.drop(columns=non_metric_column_names, errors="ignore")
		sns.stripplot(data=metric_df, ax=ax)
		strip_plot_pane = pn.pane.Matplotlib(fig, tight=True)

		fig2, ax2 = plt.subplots(figsize=(10, 5))
		sns.boxplot(data=metric_df, ax=ax2)
		box_plot_pane = pn.pane.Matplotlib(fig2, tight=True)
		plot_pane = pn.Row(strip_plot_pane, box_plot_pane)

		layout = pn.Column(
			"## Summary distribution plot",
			plot_pane,
			"## Summary DataFrame",
			df_widget,
			"## Module Result DataFrame",
			each_module_df_widget,
		)
	except Exception as e:
		logger.error(f"Skipping make boxplot and stripplot with error {e}")
		layout = pn.Column("## Summary DataFrame", df_widget)
	layout.servable()
	return layout


CSS = """
div.card-margin:nth-child(1) {
    max-height: 300px;
}
div.card-margin:nth-child(2) {
    max-height: 400px;
}
"""


def yaml_to_markdown(yaml_filepath):
	markdown_content = ""
	with open(yaml_filepath, "r", encoding="utf-8") as file:
		try:
			content = yaml.safe_load(file)
			markdown_content += f"## {os.path.basename(yaml_filepath)}\n```yaml\n{yaml.safe_dump(content, allow_unicode=True)}\n```\n\n"
		except yaml.YAMLError as exc:
			print(f"Error in {yaml_filepath}: {exc}")
	return markdown_content


def run(trial_dir: str, port: int = 7690):
	trial_summary_md = make_trial_summary_md(trial_dir=trial_dir)
	trial_summary_tab = pn.pane.Markdown(trial_summary_md, sizing_mode="stretch_width")

	node_views = [
		(str(os.path.basename(node_dir)), node_view(node_dir))
		for node_dir in find_node_dir(trial_dir)
	]

	yaml_file_markdown = yaml_to_markdown(os.path.join(trial_dir, "config.yaml"))

	yaml_file = pn.pane.Markdown(yaml_file_markdown, sizing_mode="stretch_width")

	tabs = pn.Tabs(
		("Summary", trial_summary_tab),
		*node_views,
		("Used YAML file", yaml_file),
		dynamic=True,
	)

	template = pn.template.FastListTemplate(
		site="AutoRAG", title="Dashboard", main=[tabs], raw_css=[CSS]
	).servable()
	template.show(port=port)
