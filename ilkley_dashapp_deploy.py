# Purpose is to be able to explore the ilkley fire risk, with annual stats down the side and a histogram of values
import warnings
import pandas as pd
import dash
from dash import Dash, html, dcc
from flask_caching import Cache  # Pip install required
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly_express as px
import dash_leaflet as dl
import numpy as np
import weightedstats as ws
import plotly.graph_objects as go
import gunicorn


warnings.simplefilter("ignore")
# SET UP ---------------------------------------------------------------------------------------------------------------

# Import dataframes from github directory
risk_2020_df = pd.read_csv(
    'https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/2020_pixel_risk_scores.csv')
risk_2021_df = pd.read_csv(
    'https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/2021_pixel_risk_scores.csv')
risk_2022_df = pd.read_csv(
    'https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/2022_pixel_risk_scores.csv')
risk_2023_df = pd.read_csv(
    'https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/2023_pixel_risk_scores.csv')

risk_avg_df = pd.read_csv("https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/All%20years%20averaged_pixel_risk_scores.csv")

risk_2020_path = "https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/folium_2020_png.png"
risk_2021_path = "https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/folium_2021_png.png"
risk_2022_path = "https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/folium_2022_png.png"
risk_2023_path = "https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/folium_2023_png.png"
risk_all_years_avg_path = "https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/folium_all_years_avg_png.png"

# Create a dictionary of styles - keep to a 60-30-10 (primary, secondary, tertiary colour scheme)
styles_dict = {"primary_col": "#EAE0D5",
               "secondary_col": "#C6AC8F",
               "tertiary_col": "#22333B",
               "padding": "5px",
               "inferno_map": [
                   "#000004",
                   "#1b0c41",
                   "#4a0c6b",
                   "#781c6d",
                   "#a52c60",
                   "#cf4446",
                   "#ed6925",
                   "#fb9a06",
                   "#f6d746"
               ],
               "risk_traffic_light": ["#7DDA58", "#FFDE59", "#FE9900", "#D20103"]}  # Green, yellow, orange, red

# Positioning of map Bounds
bottom_left_lat, bottom_left_lon, top_right_lat, top_right_lon = (53.87900431314598, -1.9033320916592142,
                                                                  53.92104377523524, -1.7742602376145118)
mid_lat, mid_lon = 53.90002404419061, -1.8387961646368631

# Set dash version (for Dash-mantine-components compatibility)
dash._dash_renderer._set_react_version('18.2.0')

# Instantiate Dash app and server
app = Dash(__name__,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.MINTY,
                                 dmc.styles.DATES])
server = app.server

# Create a flask Cache, used for cache.memoize
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple'})

# Creating Dash webapp sections ----------------------------------------------------------------------------------------
# TOP BANNER - HEADING
top_banner = html.Div(children=[
    dbc.Row(children=[
        dbc.Col(children=[
            html.H1(children=[
                "Ilkley Moor Remote Sensing-based Fire Risk: 2020-2023"],
                style={"font-weight": "bold",
                       "padding": styles_dict["padding"],
                       "text-align": "center",
                       "width": "100%",
                       "color": styles_dict["primary_col"]})
        ],
            width=12)
    ],
        style={"height": "10vh",
               "backgroundColor": styles_dict["tertiary_col"]})
],
    style={"width": "100vw"})

# SIDE PANEL - LAYER SELECTED HEADER; HR; MIN, MAX, STD, MEAN IN 2X2 BOX; HR; THEN HISTOGRAM OF PIXELS
side_panel = html.Div(children=[
    # Layer selected header
    dbc.Row(children=[
        dbc.Col(children=[
            html.H1(id="selected-layer-output",
                    className="hover-stats",
                    style={"font-weight": "bold",
                           "padding": styles_dict["padding"],
                           "text-align": "center",
                           "color": "black"})
        ],
            width=12)

    ],
        style={"height": "5vh"}),
    html.Hr(),
    # Stats grid
    dbc.Row(children=[
        dbc.Row(children=[
            dbc.Col(children=[
                html.H3("Minimum:",
                        style={"color": styles_dict["tertiary_col"],
                               "font-weight": "bold"}),
                html.H4(id="layer-min", className="hover-stats")
            ],
                width=6),
            dbc.Col(children=[
                html.H3("Maximum:",
                        style={"color": styles_dict["tertiary_col"],
                               "font-weight": "bold"}),
                html.H4(id="layer-max", className="hover-stats")
            ],
                width=6)
        ],
            style={"padding": "20px"}),
        dbc.Row(children=[
            dbc.Col(children=[
                html.H3("Average:",
                        style={"color": styles_dict["tertiary_col"],
                               "font-weight": "bold"}),
                html.H4(id="layer-mean", className="hover-stats")
            ],
                width=6),
            dbc.Col(children=[
                html.H3("Standard Deviation",
                        style={"color": styles_dict["tertiary_col"],
                               "font-weight": "bold"}),
                html.H4(id="layer-std", className="hover-stats")
            ],
                width=6)
        ],
            style={"padding": "20px"})
    ]),
    html.Hr(),
    dbc.Row(children=[
        dbc.Col(children=[
            html.Div(children=[
                dcc.Graph(id="risk-histogram")
            ])
        ],
            width=12)
    ],
        style={"padding": "30px"})
])

# Dash-leaflet map and controls
leaflet_map = html.Div([
    dcc.Store(id="selected-layer-in-store"),  # Store whatever the selected layer is in here
    dl.Map(center=[mid_lat, mid_lon],
           zoom=14,
           maxZoom=18,
           minZoom=13,
           id="leaflet-map",
           style={"height": "700px"},
           children=[
               # Scale bar:
               dl.Colorbar(colorscale=styles_dict["inferno_map"],
                           height=200,
                           width=20,
                           min=6,
                           max=28,
                           position="bottomleft",
                           tooltip=True),

               dl.LayersControl([
                   # Base layer - background tile map. Cannot be switched on and off.
                   dl.BaseLayer(dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                             maxZoom=18,
                                             minZoom=13),
                                checked=True,
                                name="Base Map"),

                   # Overlay layer - RIsk scores 2020 --> 2023 + Avg
                   dl.Overlay(children=[
                       dl.ImageOverlay(url=risk_2020_path,
                                       bounds=[[bottom_left_lat, bottom_left_lon], [top_right_lat, top_right_lon]],
                                       id="risk-2020")
                   ],
                       name="2020_risk",
                       checked=False),

                   dl.Overlay(children=[
                       dl.ImageOverlay(url=risk_2021_path,
                                       bounds=[[bottom_left_lat, bottom_left_lon], [top_right_lat, top_right_lon]],
                                       id="risk-2021")
                   ],
                       name="2021_risk",
                       checked=False),
                   dl.Overlay(children=[
                       dl.ImageOverlay(url=risk_2022_path,
                                       bounds=[[bottom_left_lat, bottom_left_lon], [top_right_lat, top_right_lon]],
                                       id="risk-2022")
                   ],
                       name="2022_risk",
                       checked=False),
                   dl.Overlay(children=[
                       dl.ImageOverlay(url=risk_2023_path,
                                       bounds=[[bottom_left_lat, bottom_left_lon], [top_right_lat, top_right_lon]],
                                       id="risk-2023")
                   ],
                       name="2023_risk",
                       checked=False),
                   dl.Overlay(children=[
                       dl.ImageOverlay(url=risk_all_years_avg_path,
                                       bounds=[[bottom_left_lat, bottom_left_lon], [top_right_lat, top_right_lon]],
                                       id="risk-all-years")
                   ],
                       name="all_years_avg_risk",
                       checked=False)

               ],
                   id="layer-control"  # LayerControl id can access what overlays are visible from this in a callback
               )
           ]),
],
    style={"height": "60vh",
           "width": "65vw",
           "marginTop": "20px"})

# CONTENT PANEL
content = html.Div([
    dbc.Row([
        dbc.Col([
            leaflet_map
        ],
            width=12)
    ],
        style={"height": "55vh"}),
    dbc.Row([
        dbc.Col([
            html.Div(children=[
                dcc.Graph(id="risk-pie")
            ])
        ],
            width=5),
        # separating empty plot
        dbc.Col([
            html.Div()
        ],
            width=2),
        dbc.Col([
            html.Div(children=[
                dcc.Graph(id="risk-gauge")
            ])
        ],
            width=5)
    ],
        style={"height": "30vh"})
])

# Defining the app layout
app.layout = dmc.MantineProvider(
    dbc.Container([
        dbc.Row([
            dbc.Col(top_banner, width=12)
        ],
            style={"height": "10vh"}),
        dbc.Row([
            dbc.Col(side_panel, width=4,
                    style={"backgroundColor": styles_dict["secondary_col"]}),
            dbc.Col(content, width=8)
        ],
            style={"height": "90vh"})
    ],
        style={"height": "100vh",
               "backgroundColor": styles_dict["primary_col"]},
        fluid=True),
    id="mantine-provider"
)


# Defining helper and callback functions -----------------------------------------------------------------------------

# Function to load stats df - cached
@cache.memoize(timeout=60)
def load_stats_df():
    df = pd.read_csv('https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/stats.csv')
    return df


# Function to load correct risk score df
@cache.memoize(timeout=60)
def load_risk_df(layer):
    layer_index = {"2020_risk": "2020_pixel_risk_scores",
                   "2021_risk": "2021_pixel_risk_scores",
                   "2022_risk": "2022_pixel_risk_scores",
                   "2023_risk": "2023_pixel_risk_scores",
                   "all_years_avg_risk": "All years averaged_pixel_risk_scores"}
    df = pd.read_csv(
        f"https://raw.githubusercontent.com/twrighta/ilkley-moor-firerisk/main/assets/{layer_index[layer]}.csv")
    return df


# Function to replace a layer name with a proper title name - for histogram
@cache.memoize(timeout=60)
def convert_layer_to_title(layer):
    layer_title_dict = {"2020_risk": "Risk Scores: 2020",
                        "2021_risk": "Risk Scores: 2021",
                        "2022_risk": "Risk Scores: 2022",
                        "2023_risk": "Risk Scores: 2023",
                        "all_years_avg_risk": "Risk Scores: Averaged 2020-2023"}
    return layer_title_dict[layer]


# Function to return what layers are selected into the dcc.store:
@app.callback(
    Output("selected-layer-in-store", "data"),
    Input("layer-control", "overlays")
)
def store_overlays(overlays):
    if len(overlays) >= 1:
        return overlays[-1]
    else:
        return None


# Function to access the stored layer and return its name in a title-suitable format
@app.callback(
    Output("selected-layer-output", "children"),
    Input("selected-layer-in-store", "data")
)
def get_overlay_name(overlay_name):
    if overlay_name:
        if overlay_name == "2020_risk":
            return "2020"
        elif overlay_name == "2021_risk":
            return "2021"
        elif overlay_name == "2022_risk":
            return "2022"
        elif overlay_name == "2023_risk":
            return "2023"
        elif overlay_name == "all_years_avg_risk":
            return "2020-2023 Average"
    else:
        return "No layer is selected"


# Function to find and return a chosen layer's stats
@cache.memoize(timeout=60)
@app.callback(
    [Output("layer-min", "children"),
     Output("layer-max", "children"),
     Output("layer-mean", "children"),
     Output("layer-std", "children")],
    Input("selected-layer-in-store", "data")
)
def return_layer_stats(layer):
    if layer:
        df = load_stats_df()

        # Generate stats based on layer selection
        layer_index = {"2020_risk": 0,
                       "2021_risk": 1,
                       "2022_risk": 2,
                       "2023_risk": 3,
                       "all_years_avg_risk": 4}
        layer_min = df.iloc[0, layer_index[layer]]
        layer_max = df.iloc[1, layer_index[layer]]
        layer_mean = round(df.iloc[2, layer_index[layer]], 3)
        layer_std = round(df.iloc[4, layer_index[layer]], 3)

        return layer_min, layer_max, layer_mean, layer_std
    else:
        return 0, 0, 0, 0


# Function to display histogram of the layer's risk values:
@app.callback(
    Output("risk-histogram", "figure"),
    Input("selected-layer-in-store", "data")
)
def display_pixel_histogram(layer):
    # If there is a layer selected:
    if layer:
        df = load_risk_df(layer)
        hist = px.histogram(data_frame=df,
                            x="Risk Score",
                            y="Count",
                            title=f"<b>Risk Score Distribution</b>", # <b>{convert_layer_to_title(layer)}
                            template="seaborn",
                            nbins=20,
                            color_discrete_sequence=[styles_dict["tertiary_col"]])
        hist.update_layout(plot_bgcolor="rgba(198, 172, 143, 1)",
                           paper_bgcolor="rgba(198, 172, 143, 1)")
        hist.update_yaxes(gridcolor=styles_dict["primary_col"])
        hist.update_xaxes(gridcolor=styles_dict["primary_col"])
        return hist
    else:
        blank_plot = px.scatter(title="<b>Please click on a layer to see its stats")
        blank_plot.update_layout(plot_bgcolor="rgba(198, 172, 143, 1)",
                                 paper_bgcolor="rgba(198, 172, 143, 1)")
        blank_plot.update_xaxes(gridcolor=styles_dict["primary_col"],
                                griddash="dot")
        blank_plot.update_yaxes(gridcolor=styles_dict["primary_col"],
                                griddash="dot")

        return blank_plot


# Helper function to convert risk score df into categories for the pie chart
@cache.memoize(timeout=60)
def categorize_risk_scores(df):
    risk_mask = [
        (df["Risk Score"] >= 0) & (df["Risk Score"] < 12),
        (df["Risk Score"] >= 12) & (df["Risk Score"] < 16),
        (df["Risk Score"] >= 16) & (df["Risk Score"] < 22),
        (df["Risk Score"] >= 22)
    ]
    risk_level_desc = ["Low", "Medium", "High", "Very High"]

    # create risk level column
    df["Risk Level"] = np.select(risk_mask, risk_level_desc, "Low")

    # group the df
    grouped_df = df.groupby(by="Risk Level",
                            as_index=False).sum()

    return grouped_df


# Function to display a pie chart of low, medium, high, very high risk
@app.callback(
    Output("risk-pie", "figure"),
    Input("selected-layer-in-store", "data")
)
def risk_level_pie(layer):
    if layer:
        df = load_risk_df(layer)

        # process layer's df
        grouped = categorize_risk_scores(df)

        # create pie
        pie = px.pie(data_frame=grouped,
                     values="Count",
                     names="Risk Level",
                     color="Risk Level",
                     template="seaborn",
                     title=f"<b>Risk Level Proportions</b>",
                     color_discrete_map={"Low": styles_dict["risk_traffic_light"][0],
                                         "Medium": styles_dict["risk_traffic_light"][1],
                                         "High": styles_dict["risk_traffic_light"][2],
                                         "Very High": styles_dict["risk_traffic_light"][3],
                                         })
        pie.update_layout(plot_bgcolor="rgba(234, 224, 213, 1)",
                          paper_bgcolor="rgba(234, 224, 213, 1)")
        return pie
    else:
        blank_plot = px.scatter(title="<b>Please click on a layer to see its stats")
        blank_plot.update_layout(plot_bgcolor="rgba(234, 224, 213, 1)",
                                 paper_bgcolor="rgba(234, 224, 213, 1)")
        blank_plot.update_xaxes(gridcolor=styles_dict["secondary_col"],
                                griddash="dot")
        blank_plot.update_yaxes(gridcolor=styles_dict["secondary_col"],
                                griddash="dot")
        return blank_plot


# Helper Function to calculate a weighted mean risk score of a layer, and the risk description
@cache.memoize(timeout=60)
def calculate_layer_weighted_mean(layer):
    if layer:
        df = load_risk_df(layer)
        risks = np.array(df["Risk Score"], dtype="float")
        weights = np.array(df["Count"])

        weighted_mean = ws.weighted_mean(risks, weights)

        if weighted_mean <= 8:
            risk_desc = "Low"
        elif 8 < weighted_mean <= 16:
            risk_desc = "Medium"
        elif 16 < weighted_mean <= 24:
            risk_desc = "High"
        else:
            risk_desc = "Very High"

        return [weighted_mean, risk_desc]
    else:
        return [None, None]


# Function to display a fuel gauge of the median risk, and whereabouts it lies on the risk score (low, medium, high,
# very high)
# delta is average of 2020-2023
@app.callback(
    Output("risk-gauge", "figure"),
    Input("selected-layer-in-store", "data"),
)
def risk_fuel_gauge(layer):
    median, risk_desc = calculate_layer_weighted_mean(layer)
    if median:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=median,
                title={"text": f"Overall Risk Level: {risk_desc}",
                       "font": {"size": 16,
                                "weight": "bold",
                                "color": styles_dict["tertiary_col"]}},
                delta={"reference": 16.5, "increasing": {"color": "red"}},
                gauge={
                    "axis": {"range": [0, 30],
                             "tickwidth": 1,
                             "tickcolor": styles_dict["tertiary_col"]},
                    "bar": {"color": styles_dict["tertiary_col"]},
                    "bgcolor": styles_dict["primary_col"],
                    "steps": [
                        {"range": [0, 8], "color": styles_dict["risk_traffic_light"][0]},
                        {"range": [8, 16], "color": styles_dict["risk_traffic_light"][1]},
                        {"range": [16, 24], "color": styles_dict["risk_traffic_light"][2]},
                        {"range": [24, 30], "color": styles_dict["risk_traffic_light"][3]}
                    ],
                    "threshold": {
                        "line": {"color": "red",
                                 "width": 1},
                        "thickness": 0.5,
                        "value": 29
                    }
                }
            )
        )
        gauge.update_layout(plot_bgcolor="rgba(234, 224, 213, 1)",
                            paper_bgcolor="rgba(234, 224, 213, 1)")
        return gauge
    else:
        blank_plot = px.scatter(title="<b>Please click on a layer to see its stats</b>")
        blank_plot.update_layout(plot_bgcolor="rgba(234, 224, 213, 1)",
                                 paper_bgcolor="rgba(234, 224, 213, 1)")
        blank_plot.update_xaxes(gridcolor=styles_dict["secondary_col"],
                                griddash="dot")
        blank_plot.update_yaxes(gridcolor=styles_dict["secondary_col"],
                                griddash="dot")
        return blank_plot


# Run app --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
