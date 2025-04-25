import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
from dateutil.relativedelta import relativedelta

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, lit, lag, last, coalesce, expr,
    unix_timestamp, year, quarter, round
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType,
    DoubleType, LongType
)

# --- 1) Spark + model load ---
spark = SparkSession.builder.appName("EPS_Dash").getOrCreate()
panel_ff = spark.read.parquet("/opt/spark/files/out/panel_ff.parquet")
model = PipelineModel.load("/opt/spark/files/out/eps_gbt_model")

# --- 2) Load historical data + metrics into Pandas ---
jdbc_opts = {
    "url":      spark.conf.get("spark.mysql.epsPredictor.url"),
    "user":     spark.conf.get("spark.mysql.epsPredictor.user"),
    "password": spark.conf.get("spark.mysql.epsPredictor.password"),
    "driver":   "com.mysql.cj.jdbc.Driver",
    "dbtable":  "eps_analysis"
}
eps_sdf = spark.read.format("jdbc").options(**jdbc_opts).load()
eps_pdf = (
    eps_sdf
      .withColumn("date", eps_sdf["date"].cast("date"))
      .select(
         "company_symbol","date",
         "reported_eps","predicted_eps","difference",
         "net_income","average_shares","income_from_continuing_operations"
      )
      .orderBy("company_symbol","date")
      .toPandas()
      .sort_values(["company_symbol","date"])
)
df = eps_pdf.copy()
df["date"] = pd.to_datetime(df["date"])

available_companies = sorted(df["company_symbol"].unique())
metrics = [
    {"label":"Net Income",                       "value":"net_income"},
    {"label":"Average Shares",                   "value":"average_shares"},
    {"label":"Income from Continuing Operations","value":"income_from_continuing_operations"},
]

# --- 3) Dash app with multi-page support ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- 4) App layout with styled navigation ---
app.layout = html.Div(
    
    style={'textAlign': 'center'},
    children=[
        html.H1("EPS Prediction ML Model and Financial ETL"),
        dcc.Location(id="url", refresh=False),
        html.Nav(
            children=[
                dcc.Link("EPS Historical Prediction", href="/", style={
                    "padding": "8px 16px",
                    "border": "1px solid #ccc",
                    "borderRadius": "4px",
                    "margin": "0 8px",
                    "color": "#000"
                }),
                dcc.Link("EPS Forecasting", href="/forecast", style={
                    "padding": "8px 16px",
                    "border": "1px solid #ccc",
                    "borderRadius": "4px",
                    "margin": "0 8px",
                    "color": "#000"
                }),
                dcc.Link("Company Key Metrics", href="/metrics", style={
                    "padding": "8px 16px",
                    "border": "1px solid #ccc",
                    "borderRadius": "4px",
                    "margin": "0 8px",
                    "color": "#000"
                }),
                dcc.Link("Multiple Company Comparison", href="/compare", style={
                    "padding": "8px 16px",
                    "border": "1px solid #ccc",
                    "borderRadius": "4px",
                    "color": "#000"
                }),
            ],
            style={"padding":"10px", "fontSize":"1.2em"}
        ),
        html.Div(id="page-content")
    ]
)

# --- 5) Home page layout ---
home_layout = html.Div(
    style={'display':'flex','flexDirection':'column','alignItems':'center'},
    children=[
        html.H2("EPS: Actual vs. Predicted"),
        html.Div([
            html.Label("Select Company:", style={"marginRight":"10px"}),
            dcc.Dropdown(
                id="company-dropdown",
                options=[{"label": c, "value": c} for c in available_companies],
                value=available_companies[0],
                clearable=False,
                style={
                'width': '70%',
                'minWidth': '300px',
                'margin': '0 auto',
                }
            )
        ], style={'margin':'20px 0'}),
        dcc.Graph(id="eps-graph", style={'width':'80%'}),
    ]
)

# --- 6) Forecasting page layout ---
forecast_layout = html.Div(
    style={'display':'flex','flexDirection':'column','alignItems':'center'},
    children=[
        html.H2("Future EPS Forecasting"),
        html.Div([
            html.Label("Select Company:", style={"marginRight":"10px"}),
            dcc.Dropdown(
                id="company-dropdown-forecast",
                options=[{"label": c, "value": c} for c in available_companies],
                value=available_companies[0],
                clearable=False,
                style={
                'width': '70%',
                'minWidth': '300px',
                'margin': '0 auto',
                }
            )
        ], style={'margin':'20px 0'}),
        html.H3("Enter Forecast Inputs for Next 4 Quarters"),
        dash_table.DataTable(
            id="future-input-table",
            columns=[
                {"name":"Date","id":"date","editable":False},
                {"name":"Estimated EPS","id":"estimated_eps","type":"numeric","editable":True},
                {"name":"Net Income","id":"net_income","type":"numeric","editable":True},
                {"name":"Sales","id":"sales","type":"numeric","editable":True},
                {"name":"Average Shares","id":"average_shares","type":"numeric","editable":True},
                {"name":"Income from Continuing Operations","id":"income_from_continuing_operations","type":"numeric","editable":True},
            ],
            data=[],
            editable=True,
            style_cell={"textAlign":"center"},
            style_table={"width":"80%","margin":"0 auto"}
        ),
        html.Br(),
        html.Button("Run Future Forecasts", id="future-predict-button", n_clicks=0),
        html.H3("Future EPS Predictions"),
        dash_table.DataTable(
            id="future-prediction-table",
            columns=[
                {"name":"Company","id":"company_symbol"},
                {"name":"Date","id":"date"},
                {"name":"Predicted EPS","id":"predicted_eps"},
            ],
            data=[],
            style_cell={"textAlign":"center"},
            style_table={"width":"60%","margin":"20px auto"}
        ),
        dcc.Graph(id="future-graph", style={'width':'60%', "padding": "8px 16px",
                    "border": "1px solid #ccc",
                    "borderRadius": "4px",
                    "margin": "0 8px",
                    "textDecoration": "none",
                    "color": "#000"}),
    ]
)

# --- 7) Key Metrics page layout ---
metrics_layout = html.Div(
    style={'display':'flex','flexDirection':'column','alignItems':'center'},
    children=[
        html.H2("Key Metrics by Quarter"),
        html.Div([
            html.Label("Select Company:", style={"marginRight":"10px"}),
            dcc.Dropdown(
                id="company-dropdown-metrics",
                options=[{"label": c, "value": c} for c in available_companies],
                value=available_companies[0],
                clearable=False,
                style={
                'width': '70%',
                'minWidth': '300px',
                'margin': '0 auto',
                }
            )
        ], style={'margin':'20px 0'}),
        dcc.Graph(id="net-income-graph", style={'width':'80%'}),
        dcc.Graph(id="average-shares-graph", style={'width':'80%'}),
        dcc.Graph(id="income-ops-graph", style={'width':'80%'}),
    ]
)

# --- 8) Comparison page layout ---
compare_layout = html.Div(
    style={'display':'flex','flexDirection':'column','alignItems':'center'},
    children=[
        html.H2("Category Comparison"),
        html.Div([
            html.Div([
                html.Label("Company 1", style={"marginRight":"10px"}),
                dcc.Dropdown(
                    id="cmp-company-1",
                    options=[{"label":c,"value":c} for c in available_companies],
                    value=available_companies[0],
                    clearable=False,
                    style={'width':'100%'}
                )
            ], style={"width":"24%","display":"inline-block","margin":"0 1%"}),
            html.Div([
                html.Label("Company 2", style={"marginRight":"10px"}),
                dcc.Dropdown(
                    id="cmp-company-2",
                    options=[{"label":c,"value":c} for c in available_companies],
                    value=available_companies[1],
                    clearable=False,
                    style={'width':'100%'}
                )
            ], style={"width":"24%","display":"inline-block","margin":"0 1%"}),
            html.Div([
                html.Label("Company 3", style={"marginRight":"10px"}),
                dcc.Dropdown(
                    id="cmp-company-3",
                    options=[{"label":c,"value":c} for c in available_companies],
                    value=available_companies[2],
                    clearable=False,
                    style={'width':'100%'}
                )
            ], style={"width":"24%","display":"inline-block","margin":"0 1%"}),
            html.Div([
                html.Label("Metric", style={"marginRight":"10px"}),
                dcc.Dropdown(
                    id="cmp-metric",
                    options=metrics,
                    value="net_income",
                    clearable=False,
                    style={'width':'100%'}
                )
            ], style={"width":"24%","display":"inline-block","margin":"0 1%"})
        ], style={"width":"80%","margin":"20px auto"}),
        dcc.Graph(id="comparison-bar-graph", style={'width':'80%'}),
    ]
)

# --- 9) Router callback ---
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/forecast":
        return forecast_layout
    elif pathname == "/metrics":
        return metrics_layout
    elif pathname == "/compare":
        return compare_layout
    else:
        return home_layout

# --- 10) Callbacks for Home page ---
@app.callback(
    Output("eps-graph", "figure"),
    Input("company-dropdown", "value")
)
def update_eps_graph(selected_company):
    dff = df[df["company_symbol"] == selected_company]
    return {
        "data": [
            go.Scatter(x=dff["date"], y=dff["reported_eps"], mode="lines+markers", name="Reported EPS"),
            go.Scatter(x=dff["date"], y=dff["predicted_eps"], mode="lines+markers", name="Predicted EPS"),
            go.Bar(x=dff["date"], y=dff["difference"].abs(), name="|Error|", opacity=0.4, yaxis="y2")
        ],
        "layout": go.Layout(
            title=f"EPS for {selected_company}",
            xaxis={"title": "Date"},
            yaxis={"title": "EPS ($)"},
            yaxis2={"title": "Absolute Error", "overlaying": "y", "side": "right"},
            legend={"x": 0, "y": 1},
            margin={"l": 60, "r": 60, "t": 50, "b": 50}
        )
    }

# --- 11) Callbacks for Forecasting page ---
@app.callback(
    Output("future-input-table", "data"),
    Input("company-dropdown-forecast", "value")
)
def update_input_table(selected_company):
    last_date = df[df["company_symbol"] == selected_company]["date"].max()
    rows = []
    for i in range(1, 5):
        dt = last_date + relativedelta(months=3*i)
        rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "estimated_eps": None,
            "net_income": None,
            "sales": None,
            "average_shares": None,
            "income_from_continuing_operations": None
        })
    return rows

@app.callback(
    [Output("future-prediction-table", "data"), Output("future-graph", "figure")],
    Input("future-predict-button", "n_clicks"),
    State("company-dropdown-forecast", "value"),
    State("future-input-table", "data")
)
def make_future_forecasts(n_clicks, company, table_data):
    if n_clicks < 1:
        return [], {}
    for r in table_data:
        if any(r[f] is None for f in [
            "estimated_eps", "net_income", "sales",
            "average_shares", "income_from_continuing_operations"
        ]):
            return [], {}

    schema = StructType([
        StructField("company_symbol", StringType(), False),
        StructField("date",           StringType(), False),
        StructField("reported_eps",   DoubleType(), True),
        StructField("estimated_eps",  DoubleType(), False),
        StructField("net_income",     LongType(),   False),
        StructField("sales",          DoubleType(), False),
        StructField("average_shares", DoubleType(), False),
        StructField("income_from_continuing_operations", DoubleType(), False)
    ])
    valid = []
    for r in table_data:
        valid.append({
            "company_symbol": company,
            "date":           r["date"],
            "reported_eps":   None,
            "estimated_eps":  float(r["estimated_eps"]),
            "net_income":     int(r["net_income"]),
            "sales":          float(r["sales"]),
            "average_shares": float(r["average_shares"]),
            "income_from_continuing_operations": float(r["income_from_continuing_operations"])
        })
    new_rows = spark.createDataFrame(valid, schema=schema).withColumn("date", col("date").cast(DateType()))

    ext = panel_ff.unionByName(new_rows, allowMissingColumns=True)
    wff = Window.partitionBy("company_symbol").orderBy("date").rowsBetween(Window.unboundedPreceding, 0)
    ext_ff = ext.withColumn("estimated_eps",   last("estimated_eps", ignorenulls=True).over(wff)) \
                .withColumn("net_income",      last("net_income",    ignorenulls=True).over(wff)) \
                .withColumn("sales",           last("sales",         ignorenulls=True).over(wff)) \
                .withColumn("average_shares",  last("average_shares",ignorenulls=True).over(wff)) \
                .withColumn("income_from_continuing_operations", last("income_from_continuing_operations",ignorenulls=True).over(wff)) \
                .withColumn("reported_eps_filled", last("reported_eps",  ignorenulls=True).over(wff))
    wl = Window.partitionBy("company_symbol").orderBy("date")
    ext_feat = ext_ff.withColumn("lag_eps",    coalesce(lag("reported_eps_filled",1).over(wl), col("reported_eps_filled"))) \
                     .withColumn("lag_netinc", coalesce(lag("net_income",1).over(wl),    col("net_income"))) \
                     .withColumn("eps_growth", coalesce(expr("(reported_eps_filled - lag_eps)/lag_eps"), lit(0.0))) \
                     .withColumn("inc_growth", coalesce(expr("(net_income - lag_netinc)/lag_netinc"), lit(0.0))) \
                     .withColumn("eps_surprise", coalesce(expr("(reported_eps_filled - estimated_eps)/estimated_eps"), lit(0.0))) \
                     .withColumn("date_ts",      unix_timestamp("date","yyyy-MM-dd").cast("double")) \
                     .withColumn("year",         year("date")) \
                     .withColumn("quarter",      quarter("date"))

    pred_ext = model.transform(ext_feat)
    four_dates = [r["date"] for r in table_data]
    future_df = pred_ext.filter((col("company_symbol")==company) & col("date").isin(four_dates)) \
                        .withColumn("predicted_eps", round(col("prediction"),2)) \
                        .select("company_symbol","date","predicted_eps")
    pdf = future_df.toPandas()
    pdf["date"] = pd.to_datetime(pdf["date"]).dt.strftime("%Y-%m-%d")

    hist = df[df["company_symbol"]==company]
    fig = {
        "data": [
            go.Scatter(x=hist["date"], y=hist["reported_eps"], mode="lines+markers", name="Reported EPS"),
            go.Scatter(x=hist["date"], y=hist["predicted_eps"], mode="lines+markers", name="Predicted EPS"),
            go.Scatter(x=pdf["date"],  y=pdf["predicted_eps"], mode="lines+markers", name="Future Predicted", line={"dash":"dash"})
        ],
        "layout": go.Layout(
            title=f"Extended Forecast for {company}",
            xaxis={"title":"Date","tickformat":"%Y-%m-%d"},
            yaxis={"title":"EPS ($)"},
            legend={"x":0,"y":1},
            margin={"l":60,"r":60,"t":50,"b":50}
        )
    }

    return pdf.sort_values("date").to_dict("records"), fig

# --- 12) Callbacks for Key Metrics page ---
@app.callback(
    [Output("net-income-graph", "figure"),
     Output("average-shares-graph", "figure"),
     Output("income-ops-graph", "figure")],
    Input("company-dropdown-metrics", "value")
)
def update_metric_graphs(selected_company):
    dff = df[df["company_symbol"] == selected_company].copy()
    dff["quarter_label"] = dff["date"].dt.to_period("Q").astype(str)

    net_income_fig = {
        "data":[ go.Scatter(x=dff["quarter_label"], y=dff["net_income"], mode="lines+markers") ],
        "layout":go.Layout(
            title=f"Net Income by Quarter: {selected_company}",
            xaxis={"title":"Quarter"},
            yaxis={"title":"Net Income"},
            margin={"l":60,"r":20,"t":50,"b":50}
        )
    }

    avg_shares_fig = {
        "data":[ go.Scatter(x=dff["quarter_label"], y=dff["average_shares"], mode="lines+markers") ],
        "layout":go.Layout(
            title=f"Average Shares by Quarter: {selected_company}",
            xaxis={"title":"Quarter"},
            yaxis={"title":"Average Shares"},
            margin={"l":60,"r":20,"t":50,"b":50}
        )
    }

    income_ops_fig = {
        "data":[ go.Scatter(x=dff["quarter_label"], y=dff["income_from_continuing_operations"], mode="lines+markers") ],
        "layout":go.Layout(
            title=f"Income from Continuing Ops by Quarter: {selected_company}",
            xaxis={"title":"Quarter"},
            yaxis={"title":"Income from Continuing Ops"},
            margin={"l":60,"r":20,"t":50,"b":50}
        )
    }

    return net_income_fig, avg_shares_fig, income_ops_fig

# --- 13) Callback for Comparison page ---
@app.callback(
    Output("comparison-bar-graph", "figure"),
    [
        Input("cmp-company-1", "value"),
        Input("cmp-company-2", "value"),
        Input("cmp-company-3", "value"),
        Input("cmp-metric",    "value")
    ]
)
def update_comparison_bar(c1, c2, c3, metric):
    companies = [c1, c2, c3]
    values = []
    for comp in companies:
        subset = df[df["company_symbol"] == comp]
        values.append(subset[metric].iloc[-1] if not subset.empty else 0)

    return {
        "data":[ go.Bar(x=companies, y=values, name=metric) ],
        "layout":go.Layout(
            title=f"{metric.replace('_',' ').title()} Comparison",
            xaxis={"title":"Company"},
            yaxis={"title": metric.replace('_',' ').title()},
            margin={"l":60,"r":20,"t":50,"b":50}
        )
    }

# --- 14) Run server ---
if __name__=="__main__":
    app.run(host="0.0.0.0", port=8050, debug=True, use_reloader=False)