import sys
import os
import traceback
import pandas as pd
from typing import cast
# 1. Get the absolute path to the root of your project (two levels up)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# 2. Add the root directory to Python's system path
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.orchestration.graph_builder import workflow
from src.config.settings import Settings
from src.orchestration.agent_state import AgentState

import tempfile
import json
import streamlit as st
import logging
logger = logging.getLogger("app") 

settings = Settings()

st.title('🎯 RAG Evaluation Dashboard')

st.write("Welcome to the pipeline")

st.header("1. Upload Evaluation Docs")

col1, col2 = st.columns(2)

with col1:
    uploaded_pdf = st.file_uploader("Upload Source Document", type = ['pdf'])

with col2:
    uploaded_json = st.file_uploader("Upload Golden QA Dataset", type = ['json'])

if uploaded_json and uploaded_pdf:
    st.success("✅ Files loaded successfully! Ready for configuration.")
else:
    st.info("Please upload both a PDF and your QA JSON file to proceed.")



st.sidebar.header("⚙️ Pipeline Settings")
chunk_size = st.sidebar.slider("Chunk Size (Tokens/Characters)", min_value=100, max_value= 2000, value = settings.chunk_size, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value= 500, value = settings.chunk_overlap, step=10)


st.sidebar.markdown("---")
st.header("2. Execute Pipeline")
st.sidebar.write("Evaluation Strategies:")

st.sidebar.checkbox("Character Splitter", value= True, disabled= True)
st.sidebar.checkbox("Recursive Splitter", value= True, disabled= True)
st.sidebar.checkbox("Token Splitter", value= True, disabled= True)
st.sidebar.checkbox("Semantic Splitter", value= True, disabled= True)


st.markdown("---")

if st.button("🚀 Run Evaluation Pipeline", use_container_width= True):
   

    if not uploaded_pdf or not uploaded_json:
        st.error("Hold on! You need to upload both the PDF and JSON files first.")
    else:
        actual_file_name = uploaded_pdf.name
        with st.spinner("Running chunking strategies and LLM evaluation... This may take a minute."):
            df_results = None
            try:
                # 1. Parse the JSON file directly from RAM
                golden_data = json.load(uploaded_json)
                
                # 2. Save the PDF to a temporary file so pdfplumber can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.getvalue())
                    temp_pdf_path = tmp_file.name

                # 3. Create the Initial State to pass to LangGraph
                initial_state = cast(AgentState,{
                    "file_path": temp_pdf_path,
                    "golden_dataset": golden_data,
                    # We can pass the Streamlit slider values directly to your Settings or State here!
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "file_name" : actual_file_name})

                # 4. 🔥 EXECUTE THE BACKEND PIPELINE 🔥
                with st.status("Starting Pipeline...", expanded=True) as status:
        
                    # Invoke the graph using stream_mode="updates"
                    # We use stream_mode="updates" to see which node returned data
                    for chunk in workflow.stream(initial_state, stream_mode="updates"):
                        
                        # chunk['data'] contains a dictionary where keys are node names
                        for node_name, state_update in chunk.items():
                            # Update the UI with the node that just finished
                            status.write(f"✅ Finished node: **{node_name}**")
                            status.update(label=f"Currently running: {node_name}...")
                    
                    # Finalize the status once the loop finishes
                    status.update(label="Pipeline Complete!", state="complete", expanded=False)
                # 5. Clean up the temporary file (Security Best Practice)
                os.remove(temp_pdf_path)
                                
               
            except Exception as e:
                # If your backend crashes, it will show the error elegantly on the UI instead of breaking the app
                st.error(f"Pipeline failed: {traceback.format_exc()}")
            # 🛑 BREAK OUT OF THE SPINNER BLOCK HERE 🛑
            # Notice the indentation! We are back in the main column.
            
            st.success("Pipeline Execution Complete!")
            st.header("📊 Evaluation Results")
            
            with open(r"D:\Practive Projects\Chunking_Eval\src\output\chunking_report.html", "r", encoding="utf-8") as f:
                html_data = f.read()

            st.download_button(
                label="📥 Download Dashboard to View",
                data=html_data,
                file_name="chunking_report.html",
                mime="text/html"
)