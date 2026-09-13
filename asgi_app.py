"""
ASGI entry point for Vercel deployment.
Wraps the Streamlit app as an ASGI-compatible object.
"""
import os
import streamlit as st

# Resolve absolute path so it works regardless of working directory
_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py")

app = st.App(_script_path)

if __name__ == "__main__":
    app.run()
