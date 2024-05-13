import duckdb
import pandas as pd
import panel as pn
import hvplot.pandas
import holoviews as hv

hv.extension('bokeh')
pn.extension("tabulator")

# DuckDB를 사용하여 CSV 파일 읽기
con = duckdb.connect(database=':memory:')  # 메모리 데이터베이스 사용

df_0_summary = con.execute("SELECT * FROM read_csv_auto('./tests/resources/result_project/0/summary.csv')").df()
df_0_summary_result_df = con.execute("SELECT node_line_name as name, best_execution_time FROM df_0_summary").df()

df_0_post_summary = con.execute("SELECT * FROM read_csv_auto('./tests/resources/result_project/0/post_retrieve_node_line/summary.csv')").df()
df_0_post_summary_result_df = con.execute("SELECT best_module_name as name, best_execution_time FROM df_0_post_summary").df()

df_0_pre_summary = con.execute("SELECT * FROM read_csv_auto('./tests/resources/result_project/0/pre_retrieve_node_line/summary.csv')").df()
df_0_pre_summary_result_df = con.execute("SELECT best_module_name as name, best_execution_time FROM df_0_pre_summary").df()

# 차트 생성
df_0_summary_plot = df_0_summary_result_df.hvplot.bar(x='name', y='best_execution_time', title="df_0_summary_plot")
df_0_post_summary_plot = df_0_post_summary_result_df.hvplot.bar(x='name', y='best_execution_time', title="df_0_post_summary_plot")
df_0_pre_summary_plot = df_0_pre_summary_result_df.hvplot.bar(x='name', y='best_execution_time', title="df_0_pre_summary_plot")



# Panel 대시보드 구성
dashboard = pn.Column(
    "# Result Project Summary Plot Dashboard",
    pn.Row(df_0_pre_summary_plot, df_0_post_summary_plot),
    df_0_summary_plot
)

# 서빙 
dashboard.servable()

