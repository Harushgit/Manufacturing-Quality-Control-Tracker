import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table
import pandas as pd
import io
import base64
from flask_caching import Cache
import nbformat
from nbclient import NotebookClient
import os
import subprocess
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

# Initialize the app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Manufacturing Quality Control Analysis System"
server = app.server

# Configure caching
cache = Cache(app.server, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 1500})

# Define paths for the notebook and HTML output
notebook_paths = {
    "eda": r"C:\Users\njhar\Downloads\EDA.ipynb",
    "defect": r"C:\Users\njhar\Downloads\DEFECT.ipynb",
    "trend": r"C:\Users\njhar\Downloads\TIMESERIES.ipynb",
    "root": r"C:\Users\njhar\Downloads\ROOTCAUSE.ipynb",
    "predictive": r"C:\Users\njhar\Downloads\PREDICTIVE.ipynb"
}

html_output_paths = {
    "eda": r"C:\Users\njhar\Downloads\EDA.html",
    "defect": r"C:\Users\njhar\Downloads\DEFECT.html",
    "trend": r"C:\Users\njhar\Downloads\TIMESERIES.html",
    "root": r"C:\Users\njhar\Downloads\ROOTCAUSE.html",
    "predictive": r"C:\Users\njhar\Downloads\PREDICTIVE.html"
}

# Load the trained YOLO model
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)  # Load the model directly from the saved weights
print("✅ Model loaded successfully!")

# Create upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Sidebar layout
def sidebar():
    return html.Div([
        html.H2("📊 Menu", className="display-6 text-center", style={"color": "white"}),
        html.Hr(style={"border-top": "2px solid white"}),
        dbc.Nav([
            dbc.NavLink("🏠 Home", href="/", active="exact", className="nav-link-custom"),
            dbc.NavLink("📂 Dataset Upload", href="/upload", active="exact", className="nav-link-custom"),
            dbc.NavLink("📈 EDA", href="/eda", active="exact", className="nav-link-custom"),
            dbc.NavLink("🔍 Defect Detection", href="/defect-detection", active="exact", className="nav-link-custom"),
            dbc.NavLink("📊 Trend Analysis", href="/trend-analysis", active="exact", className="nav-link-custom"),
            dbc.NavLink("🛠 Root Cause Analysis", href="/root-cause", active="exact", className="nav-link-custom"),
            dbc.NavLink("⚙️ Predictive Maintenance", href="/predictive-maintenance", active="exact", className="nav-link-custom"),
            dbc.NavLink("📊 Dashboard", href="/dashboard", active="exact", className="nav-link-custom"),
            dbc.NavLink("📷 PCB Defect Detection", href="/pcb-defect-detection", active="exact", className="nav-link-custom"),
        ], vertical=True, pills=True)
    ], style={
        "position": "fixed",
        "top": "50px",
        "left": 0,
        "bottom": 0,
        "width": "260px",
        "padding": "15px",
        "background": "linear-gradient(135deg, #1a1a2e, #16213e)",
        "height": "calc(100vh - 50px)",
        "overflow-y": "auto",
        "box-shadow": "4px 0px 10px rgba(0,0,0,0.2)",
        "color": "white",
    })

# Navbar
def navbar():
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("🏠 Home", href="/", className="nav-item-custom")),
            dbc.NavItem(dbc.NavLink("📜 About", href="/about", className="nav-item-custom")),
            dbc.NavItem(dbc.NavLink("📁 Project Details", href="/project-details", className="nav-item-custom")),
            dbc.NavItem(dbc.NavLink("🛠 Tools Used", href="/tools", className="nav-item-custom")),
        ],
        brand="⚙️ Manufacturing Quality Monitoring System",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-4",
        style={"height": "50px", "box-shadow": "0px 4px 8px rgba(0,0,0,0.1)"}
    )

# Content area
def content():
    return html.Div(id="page-content", style={
        "margin-left": "280px",
        "padding": "30px",
        "background-color": "#f4f4f4",
        "border-radius": "10px",
        "box-shadow": "0px 4px 10px rgba(0,0,0,0.1)"
    })

# PCB Defect Detection page
def pcb_defect_detection_page():
    return html.Div([
        html.H1("PCB Defect Detection", style={'textAlign': 'center'}),
        
        # Upload Image
        dcc.Upload(
            id="upload-image",
            children=html.Button("Upload Image", style={"fontSize": 20}),
            multiple=False
        ),

        # Display uploaded image
        html.Div(id="output-image-upload"),

        # Process & Detect button
        html.Button("Detect Defects", id="detect-button", n_clicks=0, style={"fontSize": 20, "marginTop": "20px"}),

        # Display results
        html.Div(id="output-detection")
    ])

# App layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar(),
    navbar(),
    content(),
])

# Callback to update page content
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/upload":
        return upload_page()
    elif pathname == "/eda":
        return html.Div([
            html.H1("📊 Exploratory Data Analysis (EDA)",style={'textAlign': 'Center','fontSize': '40px'}),
            run_notebook_page("eda")
        ])
    elif pathname == "/defect-detection":
        return html.Div([
            html.H1("✅ Defect Detection",style={'textAlign': 'Center','fontSize': '40px'}),
            run_notebook_page("defect")
        ])
    elif pathname == "/trend-analysis":
        return html.Div([
            html.H1("📊 Trend Analysis",style={'textAlign': 'Center','fontSize': '40px'}),
            run_notebook_page("trend")
        ])    
    elif pathname == "/root-cause":
        return html.Div([
            html.H1("🛠 Root Cause Analysis",style={'textAlign': 'Center','fontSize': '40px'}),
            run_notebook_page("root")
        ])
    elif pathname == "/predictive-maintenance":
        return html.Div([
            html.H1("⚙️ Predictive Maintenance",style={'textAlign': 'Center','fontSize': '40px'}),
            run_notebook_page("predictive")
        ])
    elif pathname == "/dashboard":
        return html.Div([
            html.H3("📊 Dashboard"),
            html.Iframe(
                src=app.get_asset_url("HOME.html"),  # Ensure "HOME.html" is in the "assets" directory
                style={"width": "100%", "height": "100vh", "border": "none"}
            )
        ])
    elif pathname == "/pcb-defect-detection":
        return pcb_defect_detection_page()
    elif pathname == "/about":
        return html.Div([
            html.H2("📜 About the Project"),
            html.P("👉 This project is a Manufacturing Quality Monitoring System designed to improve product quality "
           "through predictive maintenance and real-time monitoring."),

            html.P("👉 The Manufacturing Quality Control Tracker is designed to monitor, analyze, and improve the quality "
           "of products in a manufacturing process. By leveraging real-time data and statistical analysis, "
           "it identifies defects, tracks performance metrics, and provides actionable insights to enhance overall production quality."),

            html.P("👉 This tool ensures compliance with industry standards and helps in reducing waste, improving efficiency, "
           "and maintaining consistent product quality."),
        ])
    elif pathname == "/project-details":
        return html.Div([
            html.H2("📁 Project Details"),
            html.Ul([
                html.Li("📊 Exploratory Data Analysis (EDA)"),
                html.Li("✅ Defect detection using machine learning"),
                html.Li("📉 Trend analysis and process monitoring"),
                html.Li("🔍 Root cause identification"),
                html.Li("⚙️ Predictive maintenance with ML models"),
                html.Li("📊 Interactive dashboard for Users"),
                html.Li("📷 PCB Defect Detection using YOLO")
            ])
        ])
    elif pathname == "/tools":
        return html.Div([
            html.H2("🛠 Tools Used"),
            html.Ul([
                html.Li("🐍 Python"),
                html.Li("📊 Matplotlib & Plotly"),
                html.Li("🤖 Machine Learning Models"),
                html.Li("📈 Tableau for Visualization"),
                html.Li("🛠 Pandas & NumPy for data processing"),
                html.Li("📚 YOLO model version8 with Nano"),
                html.Li("💻 Dash Framework")
            ])
        ])
    return html.Div([
        html.H1("🏠 Welcome to the Home Page"),
        html.P("🙋 The Menu bar Helps to explore the different sections of the project."),
        html.P("                                                                    "),
        html.Img(
            src=app.get_asset_url("AdobeStock_425106784_Preview.jpeg"),
            style={"border-radius": "10px", "margin-top": "15px"})
    ])

# ========== 📂 DATA UPLOAD PAGE ==========

def upload_page():
    return html.Div([
        html.H2("📂 Upload Your Dataset"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["📁 Click to Select Files"]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin-bottom": "20px",
            },
            multiple=False
        ),
        html.Div(id="output-data-upload"),
        dash_table.DataTable(id="uploaded-data-preview", style_table={"overflowX": "auto"}),
    ], style={"padding": "20px"})

# ========== CALLBACK TO HANDLE FILE UPLOAD ==========
def parse_contents(contents, filename):
    """Handles uploaded files, detects encoding, and loads data."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        for encoding in ["utf-8", "ISO-8859-1", "utf-16"]:
            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
                elif filename.endswith(".xlsx"):
                    df = pd.read_excel(io.BytesIO(decoded))
                elif filename.endswith(".json"):
                    df = pd.read_json(io.StringIO(decoded.decode(encoding)))
                elif filename.endswith(".txt"):
                    df = pd.read_csv(io.StringIO(decoded.decode(encoding)), delimiter="\t")
                else:
                    return "❌ Unsupported file format. Please upload CSV, Excel, JSON, or TXT.", None
                return df.to_dict("records"), [{"name": col, "id": col} for col in df.columns]
            except Exception:
                continue
        return "❌ Error reading file. Check encoding or file format.", None
    except Exception as e:
        return f"❌ An error occurred: {str(e)}", None

@app.callback(
    [Output("output-data-upload", "children"),
     Output("uploaded-data-preview", "data"),
     Output("uploaded-data-preview", "columns")],
    [Input("upload-data", "contents"),
     Input("url", "pathname")],  # Reload data on page change
    [State("upload-data", "filename")]
)
def update_output(contents, pathname, filename):
    if contents is not None:
        data, columns = parse_contents(contents, filename)
        if isinstance(data, str):
            return html.Div(data), None, None  # Return error message

        # Store dataset in cache
        cache.set("uploaded_data", data)
        cache.set("uploaded_columns", columns)

        limited_data = data[:5]  # Show first 5 rows
        return (
            dash_table.DataTable(data=limited_data, columns=columns, style_table={"overflowX": "auto"}),
            limited_data,
            columns
        )

    # Load from cache when page reloads
    cached_data = cache.get("uploaded_data")
    cached_columns = cache.get("uploaded_columns")
    if cached_data and cached_columns:
        return dash_table.DataTable(data=cached_data[:5], columns=cached_columns, style_table={"overflowX": "auto"}), cached_data[:5], cached_columns

    return html.Div("No file uploaded yet."), None, None

# Callback to process uploaded image
@app.callback(
    Output("output-image-upload", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename")
)
def display_uploaded_image(contents, filename):
    if contents is not None:
        # Convert base64 string to image
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        img_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save the uploaded image
        with open(img_path, "wb") as f:
            f.write(decoded)

        return html.Img(src=contents, style={"width": "50%", "marginTop": "20px"})

    return None
# ========== 📈 NOTEBOOK PAGE ==========
# Callback to run YOLO model
@app.callback(
    Output("output-detection", "children"),
    Input("detect-button", "n_clicks"),
    State("upload-image", "filename")
)
def detect_defects(n_clicks, filename):
    if n_clicks > 0 and filename is not None:
        img_path = os.path.join(UPLOAD_FOLDER, filename)

        # Run YOLO detection
        results = model(img_path)

        # Process result image
        img = results[0].plot()  # Draw bounding boxes on the first result
        output_path = os.path.join(UPLOAD_FOLDER, "detected_" + filename)
        cv2.imwrite(output_path, img)

        # Convert image to base64 for display
        _, buffer = cv2.imencode('.jpg', img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        img_src = f"data:image/jpeg;base64,{encoded_image}"

        return html.Img(src=img_src, style={"width": "50%", "marginTop": "20px"})

    return None

def run_notebook_page(notebook_type):
    cached_output = cache.get(f"{notebook_type}_output")
    if cached_output:
        return html.Div([
            html.Button("Run Notebook", id=f"run-button-{notebook_type}", n_clicks=0, style={'padding': '10px', 'fontSize': '20px'}),
            html.Div(id=f"output-{notebook_type}", style={'marginTop': '20px'}, children=[
                html.Iframe(
                    src=f"/assets/{notebook_type}.html",
                    style={"width": "100%", "height": "600px", "border": "none"}
                )
            ])
        ])
    else:
        return html.Div([
            html.H1("Click to See your Outputs", style={'textAlign': 'Center','fontSize': '20px'}),
            html.Button("Run Notebook", id=f"run-button-{notebook_type}", n_clicks=0, style={'padding': '10px', 'fontSize': '20px'}),
            html.Div(id=f"output-{notebook_type}", style={'marginTop': '20px'})
        ])

@app.callback(
    Output("output-eda", "children"),
    Input("run-button-eda", "n_clicks")
)
@cache.memoize(timeout=1500)  # Cache the output for 900 seconds
def run_notebook_eda(n_clicks):
    return run_notebook("eda", notebook_paths["eda"], html_output_paths["eda"], n_clicks)

@app.callback(
    Output("output-defect", "children"),
    Input("run-button-defect", "n_clicks")
)
@cache.memoize(timeout=1500)  # Cache the output for 900 seconds
def run_notebook_defect(n_clicks):
    return run_notebook("defect", notebook_paths["defect"], html_output_paths["defect"], n_clicks)

@app.callback(
    Output("output-trend", "children"),
    Input("run-button-trend", "n_clicks")
)
@cache.memoize(timeout=1500)  # Cache the output for 900 seconds
def run_notebook_trend(n_clicks):
    return run_notebook("trend", notebook_paths["trend"], html_output_paths["trend"], n_clicks)

@app.callback(
    Output("output-root", "children"),
    Input("run-button-root", "n_clicks")
)
@cache.memoize(timeout=1500)  # Cache the output for 900 seconds
def run_notebook_root(n_clicks):
    return run_notebook("root", notebook_paths["root"], html_output_paths["root"], n_clicks)

@app.callback(
    Output("output-predictive", "children"),
    Input("run-button-predictive", "n_clicks")
)
@cache.memoize(timeout=1500)  # Cache the output for 900 seconds
def run_notebook_predictive(n_clicks):
    return run_notebook("predictive", notebook_paths["predictive"], html_output_paths["predictive"], n_clicks)

def install_missing_packages():
    required_packages = [
        "pandas",
        "scikit-learn",
        "imblearn",
        "nbformat",
        "nbclient",
        "joblib"
    ]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def run_notebook(notebook_type, notebook_path, html_output_path, n_clicks):
    if n_clicks > 0:
        try:
            # Install missing packages
            install_missing_packages()

            # Load the notebook
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # Replace input function with default values
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    cell.source = cell.source.replace("input(", "# input(")

            # Execute the notebook
            client = NotebookClient(nb)
            client.execute()

            # Save the executed notebook
            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

            # Convert executed notebook to HTML (only output, no code)
            result = subprocess.run(
                ["python", "-m", "jupyter", "nbconvert", "--to", "html", "--no-input", "--output", html_output_path, notebook_path],
                check=True,
                capture_output=True,
                text=True
            )

            # Log the output and error
            print(result.stdout)
            print(result.stderr)

            # Check for errors in the HTML conversion
            if result.returncode != 0:
                return html.Div(f"Error converting the notebook to HTML: {result.stderr}", style={'color': 'red'})

            # Move the generated HTML file to Dash's assets folder for display
            assets_dir = "assets"
            if not os.path.exists(assets_dir):
                os.makedirs(assets_dir)
            destination_path = os.path.join(assets_dir, f"{notebook_type}.html")
            if os.path.exists(destination_path):
                os.remove(destination_path)
            os.rename(html_output_path, destination_path)

            # Read the HTML content
            with open(destination_path, "r", encoding="utf-8") as file:
                html_content = file.read()

            # Cache the HTML content
            cache.set(f"{notebook_type}_output", html_content)

            # Return the HTML as an embedded Iframe
            return html.Iframe(
                src=f"/assets/{notebook_type}.html",
                style={"width": "100%", "height": "600px", "border": "none"}
            )

        except Exception as e:
            return html.Div(f"Error executing the notebook: {str(e)}", style={'color': 'red'})

    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, port=8060)
