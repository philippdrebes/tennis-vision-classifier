

# Display points on a confusion matrix, with tooltips that contain their corresponding images   # https://dash.plotly.com/dash-core-components/tooltip
# pip install dash plotly pandas scikit-learn opencv-python joblib dash_bootstrap_components pillow

from pathlib import Path
import logging
import pandas as pd
import json


import base64

from dash import Dash, dcc, html, Input, Output, no_update, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

import uvicorn
from asgiref.wsgi import WsgiToAsgi

from PIL import Image
from io import BytesIO
import base64

plotly_colors = px.colors.qualitative.Plotly


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dir_src = Path(__file__).parent
# dir_src = Path('/mnt/c/Users/8377/switchdrive/SyncVM/w HSLU S3/BS S3 Computer Vision/tennis-vision-classifier/src')
dir_data = dir_src.parent / 'data'
(dir_data / 'debugging_images').mkdir(parents=True, exist_ok=True)


def encode_image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_category(ground_truth, prediction):
    if ground_truth == prediction:
        return "True Positive" if ground_truth == 1 else "True Negative"
    else:
        return "False Positive" if prediction == 1 else "False Negative"



with open(dir_data / "confusion_stats.json", "r") as file:
    confusion_stats = json.load(file)


df_dash = pd.read_csv(dir_data / 'index_image_filepaths_classifications_all_combined_coords.csv')



fig = go.Figure()

# Loop through each classifier to add as separate series
for classifier in [col.replace('_coord_x', '') for col in df_dash.columns if col.endswith('_coord_x')]:
    df_subset = df_dash[[f"{classifier}_coord_x", f"{classifier}_coord_y", "image_relative_filepath", "classifier_class", classifier]].dropna()
    fig.add_trace(go.Scatter(
        name=classifier,
        x=df_subset[f"{classifier}_coord_x"],
        y=df_subset[f"{classifier}_coord_y"],
        mode='markers',
        marker=dict(size=10, opacity=0.5),
        customdata=df_subset[["image_relative_filepath", "classifier_class", classifier]],
        hovertemplate="<b>%{customdata[0]}</b><br>Classifier Class: %{customdata[1]}<br>Prediction: %{customdata[2]}<extra></extra>"
    ))
fig.update_traces(
    hoverinfo="none",   # overrriding the normal Plotly hover behaviour, because we are doing fancy things with the hover tooltips
    hovertemplate=None,
    )

fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1, line=dict(color="Black", width=2) )
fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5, line=dict(color="Black", width=2) )
fig.update_layout(
    xaxis=dict(range=[0, 1], tickvals=[0.25, 0.75], ticktext=['Prediction = True', 'Prediction = False'], title='', showgrid=False, zeroline=False, side='top' ),
    yaxis=dict(range=[0, 1], tickvals=[0.25, 0.75], ticktext=['Ground = False', 'Ground = True'], title='', showgrid=False, zeroline=False ),
    # Ensure the graph is square
    autosize=False, width=550, height=550, margin=dict(l=50, r=50, b=50, t=50),
    legend=dict( x=0.5, y=-0.3, orientation="h", yanchor="bottom", xanchor="center"),
    )



@callback(
    Output("graph-tooltip-2", "show"),
    Output("graph-tooltip-2", "bbox"),
    Output("graph-tooltip-2", "children"),
    Output("graph-tooltip-2", "background_color"),  # Add this line to output the background color
    Input("graph-2-dcc", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update, no_update

    point_index = hoverData["points"][0]["pointIndex"]
    curve_number = hoverData["points"][0]["curveNumber"]
    classifier_name = fig.data[curve_number].name
    image_path = df_dash.iloc[point_index]["image_relative_filepath"]
    im_url = f"data:image/jpeg;base64,{encode_image_to_base64(dir_data / image_path)}"

    bbox = hoverData["points"][0]["bbox"]    # bbox is for tooltip positioning
    background_color = plotly_colors[curve_number % len(plotly_colors)]

    ground_truth = df_dash.iloc[point_index]["classifier_class"]
    prediction = df_dash.iloc[point_index][classifier_name]

    stats = confusion_stats[classifier_name]
    total_points = int(sum(stats.values()))

    table_html = html.Table([
        html.Thead(
            html.Tr([
                html.Th("", style={'border': 'none'}),
                html.Th("Prediction = True", style={'borderBottom': '1px solid black', 'borderRight': '1px solid black'}),
                html.Th("Prediction = False", style={'borderBottom': '1px solid black'}),
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td("Ground = True", style={'borderRight': '1px solid black'}),
                html.Td(f"{stats['True Positive']}/{total_points} = {stats['True Positive %']}", style={'border': '1px solid black', 'backgroundColor': f"rgba(0, 0, 0, {float(stats['True Positive %']) / 1.5})"}),
                html.Td(f"{stats['False Positive']}/{total_points} = {stats['False Positive %']}", style={'border': '1px solid black', 'backgroundColor': f"rgba(0, 0, 0, {float(stats['False Positive %']) / 1.5})"}),
            ], style={'borderBottom': '1px solid black'}),
            html.Tr([
                html.Td("Ground = False", style={'borderRight': '1px solid black'}),
                html.Td(f"{stats['False Negative']}/{total_points} = {stats['False Negative %']}", style={'border': '1px solid black', 'backgroundColor': f"rgba(0, 0, 0, {float(stats['False Negative %']) / 1.5})"}),
                html.Td(f"{stats['True Negative']}/{total_points} = {stats['True Negative %']}", style={'border': '1px solid black', 'backgroundColor': f"rgba(0, 0, 0, {float(stats['True Negative %']) / 1.5})"}),
            ])
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse'})

    children = [
        html.Div(style={'backgroundColor': background_color}, children=[
        html.Img(src=im_url, style={"width": "150px"}),
        html.P(f"classification category: {get_category(ground_truth, prediction)}"),
        html.P(f"classifier_class: {ground_truth}"),
        html.P(f"{classifier_name}: {prediction}"),
        table_html,
        html.Table([
            html.Tbody([
                html.Tr([
                    html.Td("Accuracy"),
                    html.Td(f"{stats['Accuracy']}")
                ]),
                html.Tr([
                    html.Td("Precision"),
                    html.Td(f"{stats['Precision']}")
                ]),
                html.Tr([
                    html.Td("Recall"),
                    html.Td(f"{stats['Recall']}")
                ]),
                html.Tr([
                    html.Td("F1 Score"),
                    html.Td(f"{stats['F1 Score']}")
                ]),
                html.Tr([
                    html.Td("Matthews Correlation Coefficient"),
                    html.Td(f"{stats['Matthews Correlation Coefficient']}")
                ])
            ])
        ])])]
    return True, bbox, children, background_color



app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.scripts.config.serve_locally=True    # To avoid 'error loading dependencies' https://github.com/plotly/dash/issues/125     also try disabling adblock and cookie block
app.title = 'Tennis Classifiers Confusion Matrix'
asgi_app = WsgiToAsgi(app.server)

app.layout = dbc.Container(fluid=True, children=[
    html.H1('Tennis Classifiers Confusion Matrix', className="mb-1 text-center"),
    dbc.Row(
        dbc.Col(
            html.Div(
                dcc.Graph(id="graph-2-dcc", figure=fig, config={'displayModeBar': False}),
                style={'maxWidth': '550px', 'maxHeight': '550px', 'margin': '0 auto'}  # Center in column
            ),
            width=12,
        ),
        justify="center",
    ),
    dcc.Tooltip(id="graph-tooltip-2", direction='right'),
], style={'paddingTop': '10px'})

if __name__ == "__main__":
    uvicorn.run(asgi_app, host="0.0.0.0", port=8081)
    # app.run(debug=True)
