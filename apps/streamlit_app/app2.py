# import streamlit as st
# import streamlit.components.v1 as components
# import json



import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os



data = {
    "character_text_splitter_chunks": {
        "context_recall": 0.59,
        "context_precision": 0.46,
        "faithfullness": 0.54,
        "total_tokens": 796.12,
        "latency_seconds": 2.19,
        "ingestion_time": 1.03
    },
    "recursive_text_splitter_chunks": {
        "context_recall": 0.59,
        "context_precision": 0.46,
        "faithfullness": 0.67,
        "total_tokens": 796.25,
        "latency_seconds": 1.61,
        "ingestion_time": 0.92
    },
    "token_text_splitter_chunks": {
        "context_recall": 0.81,
        "context_precision": 0.39,
        "faithfullness": 0.5,
        "total_tokens": 2578.88,
        "latency_seconds": 4.16,
        "ingestion_time": 0.45
    },
    "semantic_chunks": {
        "context_recall": 0.7,
        "context_precision": 0.36,
        "faithfullness": 0.4,
        "total_tokens": 1235.25,
        "latency_seconds": 2.61,
        "ingestion_time": 7.69
    }
}




def scatter_plot(plot_df):
    fig_scatter = px.scatter(
    plot_df,
    x="total_tokens",
    y="context_recall",
    color="strategy",                # Categorical column → Legend
    hover_name="strategy",           # Strategy name on hover
    size="latency_seconds",
    title="Quality vs. Cost Trade-off",
    size_max=35,
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Prism,
    labels={
        "strategy": "Chunking Strategy",
        "total_tokens": "Total Tokens (Cost)",
        "context_recall": "Context Recall (Quality)",
        "latency_seconds": "Latency (s)"
    }
)

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    fig_scatter.update_layout(
        title=dict(
            text="Quality vs. Cost Trade-off",
            x=0.5,
            font=dict(size=20, family="Inter")
        ),

        font=dict(
            family="Inter",
            color="white"
        ),

        legend=dict(
            title="Chunking Strategy",
            bgcolor="rgba(40,40,40,0.8)",
            bordercolor="gray",
            borderwidth=1,
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        ),

        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Inter"
        ),

        margin=dict(l=70, r=170, t=70, b=70)
    )

    # ---------------------------------------------------------
    # Grid Styling
    # ---------------------------------------------------------
    fig_scatter.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#ACA9A9"
    )

    fig_scatter.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#ACA9A9"
    )
    graph_html = fig_scatter.to_html(
    full_html=False,
    include_plotlyjs=False,
    config={
        "responsive": True,
        "displaylogo": False
    }
)

    return graph_html

    # st.plotly_chart(fig_scatter, use_container_width=True)




def bargraph(data):
    # Define a vibrant color palette designed to stand out against a dark background
    # Using neon/bright variants of the previous colors:
    # Bright Cyan, Bright Green, Bright Orange
    df = (pd.DataFrame(data)).T
    df = df.reset_index()
    df = df.copy()

    df = df.reset_index().rename(columns={"index": "strategy"})
    dark_theme_colors = ['#00E5FF', '#00FF88', '#FF9F43']

    fig_bar_dark = go.Figure(data=[
        go.Bar(
            name='Context Recall', 
            x=df["strategy"], 
            y=df['context_recall'],
            marker_color=dark_theme_colors[0],
            text=df['context_recall'],
            texttemplate='%{text:.2f}',
            textposition='outside',
            # Ensuring text on top of bars is bright (Plotly usually handles this
            # with templates, but explicitly setting it helps)
            textfont=dict(color='white', size=11) 
        ),
        go.Bar(
            name='Context Precision', 
            x=df["strategy"], 
            y=df['context_precision'],
            marker_color=dark_theme_colors[1],
            text=df['context_precision'],
            texttemplate='%{text:.2f}',
            textposition='outside',
            textfont=dict(color='white', size=11)
        ),
        go.Bar(
            name='Faithfulness', 
            x=df["strategy"], 
            y=df['faithfullness'], 
            marker_color=dark_theme_colors[2],
            text=df['faithfullness'],
            texttemplate='%{text:.2f}',
            textposition='outside',
            textfont=dict(color='white', size=11)
        )
    ])

    # Updating the layout for the dark theme and polished aesthetics
    fig_bar_dark.update_layout(
        # --- The Key Change ---
        template="plotly_dark", # Sets the entire canvas to dark mode
        
        barmode='group', 
        title="Retrieval vs. Generation Quality by Strategy",
        title_x=0.5,             # Centers the title
        title_font=dict(color='white', size=22, family="Inter"), # Classic dark-mode look

        yaxis_title="Score (0 to 1)",
        xaxis_title="Chunking Strategy",
        xaxis_tickangle=-45,     # Keep the readable angle
        uniformtext_minsize=10, 
        uniformtext_mode='hide',

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,              # Legend still on top
            xanchor="center",
            x=0.5,
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0)', # Full transparency against dark background
            bordercolor='rgba(255,255,255,0.2)' # Subtle light border
        ),
        margin=dict(t=100)       # Margin so title/legend don't overlap
    )

    # Optional: Ensure gridlines are subtle but visible against the dark canvas
    fig_bar_dark.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)',range=[0, 1.0])

    return fig_bar_dark.to_html(full_html=False,include_plotlyjs=False)
    # st.plotly_chart(fig_bar_dark, use_container_width=True)



def create_graphs(data):
    # data = state['combined_matrics']
    bar = bargraph(data)
    df = (pd.DataFrame(data)).T
    plot_df = df.reset_index().rename(columns={"index": "strategy"})
    scatter  = scatter_plot(plot_df)

    
    # 1. Set your directory
    output_dir = r"D:\Practive Projects\Chunking_Eval\src\report_design"
    scatter_file = f"scatterplot.html"
    bar_file = f"bargraph.html"

    # 2. CREATE the folder if it doesn't exist (the magic line)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Combine them into a full path
    final_destination1 = os.path.join(output_dir, scatter_file)
    final_destination2 = os.path.join(output_dir, bar_file)

    # 4. Save the file
    with open(final_destination1, "w", encoding="utf-8") as f1:
        f1.write(scatter)

    # 5. Save the file
    with open(final_destination2, "w", encoding="utf-8") as f2:
        f2.write(bar)

    print(f"Success! Graph Saved ")

    return {}

create_graphs(data)

from pathlib import Path
from jinja2 import Template, FileSystemLoader, Environment

path = Path("D:/Practive Projects/Chunking_Eval/src/report_design/template.html")
path1 = Path("D:/Practive Projects/Chunking_Eval/src/report_design/scatterplot.html")
path2 = Path("D:/Practive Projects/Chunking_Eval/src/report_design/bargraph.html")

    
# 1. Setup the environment to look in the current folder
file_loader = FileSystemLoader(path.parent)
env = Environment(loader=file_loader)

# 2. Load your "template.html" file
template = env.get_template(path.name)
    
with open(path1._raw_paths[0], 'r',encoding='utf-8') as file:
    scatter = file.read()

with open(path2._raw_paths[0], 'r',encoding='utf-8') as file:
    bar = file.read()

context_data = {"retrieval_generation_graph" : scatter,
                "quality_cost_graph" : bar, "llm_payload":{
        "retrieval_card": {
            "headline": "",
            "body": "",
            "hidden_story": ""
        },
        "generation_card": {
            "headline": "",
            "body": "",
            "hidden_story": ""
        },
        "efficiency_card": {
            "headline": "",
            "body": "",
            "hidden_story": ""
        }
    }
}

rendered_html = template.render(context_data) 

# 1. Set your directory
output_dir = r"D:\Practive Projects\Chunking_Eval\src\output"
file_name = f"graph_reprot.html"

# 2. CREATE the folder if it doesn't exist (the magic line)
os.makedirs(output_dir, exist_ok=True)

# 3. Combine them into a full path
final_destination = os.path.join(output_dir, file_name)

# 4. Save the file
with open(final_destination, "w", encoding="utf-8") as f:
    f.write(rendered_html)

print(f"Success! graph Report saved ")
