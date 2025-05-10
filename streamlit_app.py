import json
import os
from typing import List

import networkx as nx
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from annotated_text import annotated_text
from streamlit_extras.badges import badge

from scripts.similarity.get_score import *
from scripts.utils import get_filenames_from_dir
from scripts.utils.logger import init_logging_config

from nltk.corpus import stopwords
import string

nltk.download('stopwords')
nltk.download('punkt')

# ---------- Helper Functions ----------

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return tokens

def create_annotated_text(input_string: str, word_list: List[str], annotation: str, color_code: str):
    tokens = clean_text(input_string)
    word_set = set(word_list)
    annotated_output = [(token, annotation, color_code) if token in word_set else token for token in tokens]
    return annotated_output

def create_star_graph(nodes_and_weights, title):
    G = nx.Graph()
    central_node = "resume"
    G.add_node(central_node)
    for node, weight in nodes_and_weights:
        G.add_node(node)
        G.add_edge(central_node, node, weight=weight * 100)

    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br># of connections: {len(list(G.adj[node]))}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text",
        marker=dict(
            showscale=True, colorscale="Rainbow", reversescale=True,
            color=[len(list(G.adj[n])) for n in G.nodes()],
            size=10,
            colorbar=dict(thickness=15, title="Node Connections", xanchor="left", titleside="right"),
            line_width=2
        ),
        text=node_text
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=title, titlefont_size=16, showlegend=False, hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------- Streamlit UI Setup ----------

st.set_page_config(
    page_title="Resume Matcher",
    page_icon="Assets/img/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Remove White Boxes Styling ----------
st.markdown("""
    <style>
        .block-container > div:nth-child(2) > div > div > div > div {
            box-shadow: none !important;
            border: none !important;
            background: transparent !important;
            padding: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- App Header ----------
st.markdown("""
    <h1 style='text-align: center; color: white; font-size: 2.75rem; font-weight: 600;'>🧠 Resume Matcher: Smarter Job Matching</h1>
    <p style='text-align: center; font-size: 1.2rem; color: #6c757d;'>Analyze, compare, and optimize your resume against job descriptions using NLP</p>
    <hr style='margin-top:20px; margin-bottom:30px;'>
""", unsafe_allow_html=True)

init_logging_config()
cwd = find_path("Resume-Matcher")
config_path = os.path.join(cwd, "scripts", "similarity")

# ---------- Resume Selection ----------
resume_csv_path = os.path.join("Data", "Raw", "UpdatedResumeDataSet.csv")
resume_df = pd.read_csv(resume_csv_path)
resume_df.dropna(subset=["Resume", "Category"], inplace=True)

st.subheader("📄 Select a Resume")
resume_index = st.selectbox("Choose from available resumes:", resume_df.index.tolist(), format_func=str)
selected_resume = resume_df.loc[resume_index, "Resume"]

selected_file = {
    "clean_data": selected_resume,
    "extracted_keywords": clean_text(selected_resume)[:20],
    "keyterms": [(kw, 0.05) for kw in clean_text(selected_resume)[:10]]
}
with st.expander("🔍 Resume Preview & Keywords"):
    st.caption("This is how your resume appears to an ATS.")
    st.markdown(f"<p style='color:white;'>{selected_file['clean_data']}</p>", unsafe_allow_html=True)
    st.write("**Extracted Keywords:**")
    annotated_text(create_annotated_text(selected_file["clean_data"], selected_file["extracted_keywords"], "KW", "#0B666A"))

# ---------- JD Selection ----------
jd_csv_path = os.path.join("Data", "Raw", "JobDescriptions.csv")
jd_df = pd.read_csv(jd_csv_path)
jd_df.dropna(subset=["Job Description"], inplace=True)

st.subheader("💼 Select a Job Description")
jd_options = [f"{i}: {desc[:100]}..." for i, desc in zip(jd_df.index, jd_df["Job Description"])]
selected_jd_label = st.selectbox("Choose JD:", jd_options)
jd_index = int(selected_jd_label.split(":")[0])

selected_jd_text = jd_df.loc[jd_index, "Job Description"]
selected_jd = {
    "clean_data": selected_jd_text,
    "extracted_keywords": clean_text(selected_jd_text)[:20],
    "keyterms": [(kw, 0.05) for kw in clean_text(selected_jd_text)[:10]]
}
with st.expander("🔍 JD Preview & Keywords"):
    st.write("#### Job Description Preview")
    st.markdown(f"<p style='color:white;'>{selected_jd['clean_data']}</p>", unsafe_allow_html=True)
    st.write("**Common Keywords Highlighted:**")
    annotated_text(create_annotated_text(selected_file["clean_data"], selected_jd["extracted_keywords"], "JD", "#F24C3D"))

# ---------- Comparison Charts ----------
st.subheader("📊 Resume vs JD - Key Term Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔎 Resume Key Terms**")
    create_star_graph(selected_file["keyterms"], "Entities from Resume")
    df_resume = pd.DataFrame(selected_file["keyterms"], columns=["Keyword", "Value"])
    fig1 = go.Figure(data=[go.Table(
        header=dict(values=["Keyword", "Value"], fill_color="#070A52", font=dict(color="white")),
        cells=dict(values=[df_resume.Keyword, df_resume.Value], fill_color="#6DA9E4")
    )])
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.treemap(df_resume, path=["Keyword"], values="Value", title="Resume Keyword Tree")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("**📌 JD Key Terms**")
    create_star_graph(selected_jd["keyterms"], "Entities from JD")
    df_jd = pd.DataFrame(selected_jd["keyterms"], columns=["Keyword", "Value"])
    fig3 = go.Figure(data=[go.Table(
        header=dict(values=["Keyword", "Value"], fill_color="#070A52", font=dict(color="white")),
        cells=dict(values=[df_jd.Keyword, df_jd.Value], fill_color="#6DA9E4")
    )])
    st.plotly_chart(fig3, use_container_width=True)
    fig4 = px.treemap(df_jd, path=["Keyword"], values="Value", title="JD Keyword Tree")
    st.plotly_chart(fig4, use_container_width=True)

# ---------- Similarity Score ----------
resume_string = " ".join(selected_file["extracted_keywords"])
jd_string = " ".join(selected_jd["extracted_keywords"])
result = get_score(resume_string, jd_string)
similarity_score = round(result[0].score * 100, 2)

score_color = "green" if similarity_score >= 75 else "orange" if similarity_score >= 60 else "red"
score_bg = "#d4edda" if score_color == "green" else "#fff3cd" if score_color == "orange" else "#f8d7da"

st.markdown(f"""
<div style="padding:1em; background-color:{score_bg}; border-radius:10px; text-align:center;">
    <h2>📊 Similarity Score: <span style='color:{score_color};'>{similarity_score}%</span></h2>
</div>
""", unsafe_allow_html=True)

# ---------- Back to Top ----------
st.markdown("[:arrow_up: Back to Top](#resume-matcher-smarter-job-matching)")