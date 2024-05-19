import ast
import logging
import os
import pathlib
from typing import Dict

import duckdb
import holoviews as hv
import pandas as pd
import panel as pn

from autorag.utils.util import dict_to_markdown, dict_to_markdown_table

logger = logging.getLogger("AutoRAG")

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent

pn.extension('terminal', 'tabulator', 'mathjax', 'vega', console_output='disable', sizing_mode="stretch_width")
hv.extension('bokeh')
# DuckDB를 사용하여 CSV 파일 읽기
con = duckdb.connect(database=':memory:')  # 메모리 데이터베이스 사용


def get_metric_values(node_summary_df: pd.DataFrame) -> Dict:
    non_metric_column_names = ['filename', 'module_name', 'module_params', 'execution_time', 'average_output_token',
                               'is_best']
    best_row = node_summary_df.loc[node_summary_df['is_best']].drop(columns=non_metric_column_names, errors='ignore')
    assert len(best_row) == 1, "The best module must be only one."
    return best_row.iloc[0].to_dict()


def make_trial_summary_md(trial_dir):
    trial_summary_csv = pd.read_csv(os.path.join(trial_dir, 'summary.csv'))
    markdown_text = f"""# Trial Result Summary
- Trial Directory : {trial_dir}

"""
    for idx, row in trial_summary_csv.iterrows():
        node_line_name = row['node_line_name']
        node_type = row['node_type']
        node_summary_filepath = os.path.join(trial_dir, node_line_name, node_type, 'summary.csv')
        node_summary_df = pd.read_csv(node_summary_filepath)
        metric_dict = get_metric_values(node_summary_df)

        markdown_text += f"""---

## {node_type} best module

### Module Name

{row['best_module_name']}

### Module Params

{dict_to_markdown(ast.literal_eval(row['best_module_params']), level=3)}

### Metric Values

{dict_to_markdown_table(metric_dict, key_column_name='metric_name', value_column_name='metric_value')}

"""

    return markdown_text


#
# df_0_summary = con.execute("SELECT * FROM read_csv_auto('./tests/resources/result_project/0/summary.csv')").df()
# df_0_summary_result_df = con.execute("SELECT node_line_name as name, best_execution_time FROM df_0_summary").df()
#
# df_0_post_summary = con.execute(
#     "SELECT * FROM read_csv_auto('./tests/resources/result_project/0/post_retrieve_node_line/summary.csv')").df()
# df_0_post_summary_result_df = con.execute(
#     "SELECT best_module_name as name, best_execution_time FROM df_0_post_summary").df()
#
# df_0_pre_summary = con.execute(
#     "SELECT * FROM read_csv_auto('./tests/resources/result_project/0/pre_retrieve_node_line/summary.csv')").df()
# df_0_pre_summary_result_df = con.execute(
#     "SELECT best_module_name as name, best_execution_time FROM df_0_pre_summary").df()
#
# # 차트 생성
# df_0_summary_plot = df_0_summary_result_df.hvplot.bar(x='name', y='best_execution_time', title="df_0_summary_plot",
#                                                       height=250, sizing_mode='stretch_width')
# df_0_post_summary_plot = df_0_post_summary_result_df.hvplot.bar(x='name', y='best_execution_time',
#                                                                 title="df_0_post_summary_plot", height=250,
#                                                                 sizing_mode='stretch_width')
# df_0_pre_summary_plot = df_0_pre_summary_result_df.hvplot.bar(x='name', y='best_execution_time',
#                                                               title="df_0_pre_summary_plot", height=250,
#                                                               sizing_mode='stretch_width')
#
# # Panel 대시보드 구성
# dashboard = pn.Column(
#     df_0_pre_summary_plot,
#     df_0_post_summary_plot,
#     df_0_summary_plot,
# )

#
CSS = """
div.card-margin:nth-child(1) {
    max-height: 300px;
}
div.card-margin:nth-child(2) {
    max-height: 400px;
}
"""


#
#
# layout1 = pn.Column(styles={"background": "green"}, sizing_mode="stretch_both")
# layout2 = dashboard
# from bokeh.plotting import figure
#
# p1 = figure(width=300, height=300, name='Scatter')
# p1.scatter([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 2, 1, 0])
#
# # ---- https://vega.github.io/
# pn.extension('vega')
# import altair as alt
#
# print(df_0_summary.head())
# chart = alt.Chart(df_0_summary).mark_circle(size=160).encode(
#     x='best_execution_time',
#     y='best_module_name',
#     color='node_line_name',
# ).interactive()
#
# altair_pane = pn.panel(chart)
# # --- data frame to altair
#
# daraframePanel = pn.pane.DataFrame(df_0_summary)
#
# # ----
# layout3 = pn.Column(
#     "# Result Project Summary Plot ",
#     altair_pane, daraframePanel,
#     dashboard,
#     scroll=True, height=620
# )
#
# xs = np.linspace(0, np.pi)
#
# freq = pn.widgets.FloatSlider(name="Frequency", start=0, end=10, value=2)
# phase = pn.widgets.FloatSlider(name="Phase", start=0, end=np.pi)
#
#
# def sine(freq, phase):
#     return pd.DataFrame(dict(y=np.sin(xs * freq + phase)), index=xs)
#
#
# def cosine(freq, phase):
#     return pd.DataFrame(dict(y=np.cos(xs * freq + phase)), index=xs)
#
#
# dfi_sine = hvplot.bind(sine, freq, phase).interactive()
# dfi_cosine = hvplot.bind(cosine, freq, phase).interactive()
#
# plot_opts = dict(
#     responsive=True, min_height=400,
#     # Align the curves' color with the template's color
#     color=pn.template.FastListTemplate.accent_base_color
# )
#
# # 파이썬  실행위치 변수
# print('os.getcwd()', os.getcwd())
# PWD = os.getcwd()
# import json
# import yaml
#
#
# def find_yaml_files(directory):
#     # 모든 YAML 파일의 경로를 찾기
#     yaml_files = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.yaml') or file.endswith('.yml'):
#                 yaml_files.append(os.path.join(root, file))
#     return yaml_files
#
#
# def yaml_to_markdown(yaml_files):
#     markdown_content = ""
#     for yaml_file in yaml_files:
#         with open(yaml_file, 'r', encoding='utf-8') as file:
#             try:
#                 # YAML 파일 로드
#                 content = yaml.safe_load(file)
#                 # Markdown 문법으로 포맷팅
#                 markdown_content += f"## {yaml_file.replace(f'{PROJECT_DIR}/', '')}\n```yaml\n{yaml.dump(content, allow_unicode=True)}\n```\n\n"
#             except yaml.YAMLError as exc:
#                 print(f"Error in {yaml_file}: {exc}")
#     return markdown_content
#
#
# # --TextInput--
#
# path_input = pn.widgets.TextInput(name='Text Input', value='/tests/resources/result_project',
#                                   placeholder='Enter a string here...')
#
# # --MultiSelect --
# PROJECT_DIR = f'{PWD}/tests/resources/result_project'
#
# with open(f'{PROJECT_DIR}/trial.json', 'r') as f:
#     json_data = json.load(f)
#
# print('getJsonLoad', json_data)
# formatted_list = [f"{trial['trial_name']} - {trial['start_time']}" for trial in json_data]
#
# fileListOptions = util.find_trial_dir(project_dir=PROJECT_DIR)
# # fileListOptions 의 내용에서 'result_project' 을 제거 하고 싶다.
# fileListOptions = [x.replace(f'{PROJECT_DIR}/', '') for x in fileListOptions]
# fileListOptions.sort()
# fileSelector = pn.widgets.MultiSelect(name='Test #', value=['Pear'], options=formatted_list, size=8)
# btn_info = pn.widgets.RadioButtonGroup(name='show info', options=['debug', 'info', 'warning'])
#
# yaml_files = find_yaml_files(PROJECT_DIR)
# yamlFileMarkdown = yaml_to_markdown(yaml_files)
#
# yamlFile = pn.pane.Markdown(yamlFileMarkdown, sizing_mode='stretch_width')
#
# tabs = pn.Tabs(('Chart', layout3), ('Config', yamlFile), dynamic=True)

def run(trial_dir: str):
    trial_summary_md = make_trial_summary_md(trial_dir=trial_dir)
    trial_summary_tab = pn.pane.Markdown(trial_summary_md, sizing_mode='stretch_width')

    tabs = pn.Tabs(('Summary', trial_summary_tab), dynamic=True)

    template = pn.template.FastListTemplate(site="AutoRAG", title="Dashboard",
                                            # sidebar=[path_input, fileSelector, ],
                                            main=[tabs], raw_css=[CSS]).servable()
    template.show(threaded=True)
