"""
AI Video Assistant — Streamlit UI

Loads the existing project notebooks (00_llm_config.ipynb ... 11_orchestrator.ipynb)
directly, the same way main.ipynb's `%run` chain does — but %run is an IPython/Jupyter
magic and doesn't exist in a plain script like this one. `_load_notebook()` below gets
the same effect: it reads each notebook's code cells and executes them, in order, into
one shared namespace, skipping any Jupyter-only magic lines (e.g. `%%capture`, `!pip`).

Run from inside this same folder (so the relative notebook/fonts/downloads paths
resolve correctly):

    streamlit run streamlit_app.py
"""

import json
import linecache
import os
import tempfile
import uuid

import streamlit as st

# --------------------------------------------------------------------------- #
# 1. Load the pipeline (00_llm_config.ipynb ... 11_orchestrator.ipynb)
# --------------------------------------------------------------------------- #

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))

_MODULES = [
    "00_llm_config.ipynb",
    "01_audio_processor.ipynb",
    "02_transcriber.ipynb",
    "03_summarizer.ipynb",
    "04_extractor.ipynb",
    "05_vector_store.ipynb",
    "06_rag_engine.ipynb",
    "07_report_generator.ipynb",
    "08_llamaindex_retriever.ipynb",
    "09_agents.ipynb",
    "10_guardrails.ipynb",
    "11_orchestrator.ipynb",
]


def _load_notebook(path: str, namespace: dict) -> None:
    """Execute every code cell of a notebook into `namespace`, in order — the
    script equivalent of `%run ./file.ipynb`. Skips magic/shell lines, since
    those only exist inside a Jupyter kernel (this project's module notebooks
    never rely on them; only main.ipynb's install cell does, which this app
    doesn't need — install once via requirements.txt instead)."""
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cell_index = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not src.strip() or src.strip().startswith(("%", "!")):
            continue
        cell_index += 1
        # A synthetic filename (not a real path), so a traceback raised inside
        # this cell can't fall back to linecache opening the real .ipynb file
        # on disk for source context — that showed raw JSON instead of code
        # (e.g. a traceback pointing at `"source": [`), since compile()'s
        # `path` argument previously WAS the real .ipynb path.
        cell_filename = f"{path} (cell {cell_index})"
        linecache.cache[cell_filename] = (
            len(src), None, src.splitlines(keepends=True), cell_filename,
        )
        exec(compile(src, cell_filename, "exec"), namespace)


@st.cache_resource(show_spinner="Loading pipeline (first run only)...")
def load_pipeline():
    namespace = {"__name__": "video_assistant_pipeline"}
    for nb_file in _MODULES:
        _load_notebook(os.path.join(NOTEBOOK_DIR, nb_file), namespace)
    return namespace


try:
    _pipeline = load_pipeline()
    run_new_meeting = _pipeline["run_new_meeting"]
    ask_meeting_question = _pipeline["ask_meeting_question"]
    generate_transcript_pdf = _pipeline["generate_transcript_pdf"]
except Exception as e:
    st.set_page_config(page_title="AI Video Assistant — Error", page_icon="⚠️")
    st.error(
        "Couldn't load the pipeline. This usually means a missing API key, a "
        "missing dependency, or a notebook file that isn't in this folder."
    )
    st.exception(e)
    st.stop()


# --------------------------------------------------------------------------- #
# 1b. Patch download_youtube_audio to support YouTube cookies from st.secrets
#
# Cloud deployments (Streamlit Community Cloud, etc.) run on datacenter IPs
# that YouTube blocks. To bypass this, add a [youtube] section to your app
# secrets (Settings → Secrets) with a `cookies` key containing the contents
# of a Netscape-format cookies.txt file exported from a logged-in browser.
#
# Example secrets.toml entry:
#   [youtube]
#   cookies = """
#   # Netscape HTTP Cookie File
#   .youtube.com  TRUE  /  TRUE  ...
#   """
# --------------------------------------------------------------------------- #

def _cloud_download_youtube_audio(url: str) -> str:
    """Cloud-safe replacement for download_youtube_audio from the notebook.

    Tries multiple YouTube player clients in order. Injects cookies and/or a
    proxy from st.secrets when present to bypass datacenter IP blocking.

    Add to Streamlit secrets (Settings → Secrets):
      [youtube]
      cookies = \"\"\"<Netscape cookie file content>\"\"\"
      # Optional — a proxy helps when cookies alone aren't enough:
      proxy = "http://user:pass@proxy-host:port"
    """
    import glob as _glob
    import tempfile as _tempfile
    import yt_dlp as _yt_dlp

    DOWNLOAD_DIR = _pipeline.get("DOWNLOAD_DIR", "downloads")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # --- Read cookies from st.secrets (use [] not .get() — more reliable) ---
    cookies_content = None
    try:
        cookies_content = st.secrets["youtube"]["cookies"]
    except (KeyError, AttributeError, Exception):
        pass

    # --- Read optional proxy from st.secrets ---
    proxy = None
    try:
        proxy = st.secrets["youtube"]["proxy"]
    except (KeyError, AttributeError, Exception):
        pass

    cookie_file_path = None
    if cookies_content:
        try:
            tf = _tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            )
            tf.write(cookies_content)
            tf.flush()
            tf.close()
            cookie_file_path = tf.name
        except Exception:
            cookie_file_path = None

    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    # tv_embedded / mweb are often less restricted on cloud IPs
    clients_to_try = ["tv_embedded", "ios", "mweb", "web", "web_creator", "android"]
    last_error = None

    for client in clients_to_try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "extractor_args": {"youtube": {"player_client": [client]}},
            "retries": 5,
            "fragment_retries": 5,
            "continuedl": False,
            "nopart": True,
        }
        if cookie_file_path:
            ydl_opts["cookiefile"] = cookie_file_path
        if proxy:
            ydl_opts["proxy"] = proxy

        try:
            with _yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                guessed_filename = ydl.prepare_filename(info)

            # Locate the converted WAV file
            filename = None
            for d in info.get("requested_downloads", []) or []:
                candidate = d.get("filepath") or d.get("_filename")
                if candidate and os.path.exists(candidate):
                    filename = candidate
                    break
            if filename is None:
                base, _ = os.path.splitext(guessed_filename)
                candidate = base + ".wav"
                if os.path.exists(candidate):
                    filename = candidate
            if filename is None:
                candidates = sorted(
                    _glob.glob(os.path.join(DOWNLOAD_DIR, "*.wav")),
                    key=os.path.getmtime,
                    reverse=True,
                )
                if candidates:
                    filename = candidates[0]

            if filename and os.path.exists(filename):
                return filename
        except Exception as e:
            last_error = e
            continue  # try next client

    # Clean up cookie file before raising
    if cookie_file_path:
        try:
            os.unlink(cookie_file_path)
        except OSError:
            pass

    raise RuntimeError(
        f"YouTube blocked the download from this server's IP address "
        f"(tried clients: {clients_to_try}). "
        f"Workaround: download the video locally and use the 'Upload a file' option instead. "
        f"Technical detail: {last_error}"
    )


if "download_youtube_audio" in _pipeline:
    _pipeline["download_youtube_audio"] = _cloud_download_youtube_audio





# --------------------------------------------------------------------------- #
# 2. Page setup + styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 3.5rem; padding-bottom: 3rem;}
        .report-title {font-size: 1.9rem; font-weight: 700; margin-bottom: 0.3rem;}
        .badge {
            display: inline-block; padding: 3px 12px; border-radius: 999px;
            background: #2b6cb0; color: white; font-size: 0.78rem;
            margin-right: 6px; font-weight: 600;
        }
        .stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 600; }
        .rtl-block {
            direction: rtl; text-align: right; font-size: 1.05rem; line-height: 1.9;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_md(text: str, language: str) -> None:
    """Render Markdown, flipped to right-to-left for Arabic so on-screen
    reading direction matches the PDF report."""
    if not text or not str(text).strip():
        st.caption("Nothing here yet.")
        return
    if language == "arabic":
        st.markdown(f'<div class="rtl-block">{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(text)


# --------------------------------------------------------------------------- #
# 3. Session state
# --------------------------------------------------------------------------- #

if "state" not in st.session_state:
    st.session_state.state = None  # PipelineState dict, once a recording has been processed

# A per-session id, used to build unique transcript-PDF filenames below so two
# browser sessions hitting "Download transcript as PDF" at the same time can
# never race on the same temp file (the same class of bug fixed in
# 09_agents.ipynb's content_agent for the main meeting report PDF).
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex


# --------------------------------------------------------------------------- #
# 4. Sidebar — new recording input
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("🎬 New Recording")

    input_mode = st.radio("Source", ["YouTube / URL", "Upload a file"], label_visibility="collapsed")

    source = None
    if input_mode == "YouTube / URL":
        source = st.text_input("YouTube URL or link", placeholder="https://www.youtube.com/watch?v=...")
        st.caption("⚠️ YouTube may block downloads from cloud servers. If it fails, download the audio locally and use 'Upload a file' instead.")
    else:
        uploaded = st.file_uploader("Audio or video file", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"])
        if uploaded is not None:
            tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            source = tmp_path
            st.caption(f"Ready: {uploaded.name}")

    st.divider()

    # Three independent settings — replaces the old single "language"
    # selector that used to control the ASR model, the transcript's output
    # language, AND the summary's output language all at once. Audio is
    # always auto-detected now (see 02_transcriber.ipynb), so "audio type"
    # only matters for picking the Hinglish-specialized ASR model; it no
    # longer forces a transcription language.
    audio_type = st.selectbox(
        "Audio type",
        ["auto", "hinglish"],
        format_func=lambda a: {
            "auto": "Auto-detect (most videos)",
            "hinglish": "Hindi-English mixed (Hinglish)",
        }[a],
        help="What kind of speech is in the audio. This picks which speech-recognition "
        "model runs — it does not translate anything by itself.",
    )
    transcript_language = st.selectbox(
        "Transcript language",
        ["english", "arabic"],
        format_func=lambda l: {"english": "English", "arabic": "Arabic"}[l],
        help="The language the 'Translated Transcript' tab is written in.",
    )
    summary_language = st.selectbox(
        "Summary language",
        ["english", "arabic"],
        format_func=lambda l: {
            "english": "English",
            "arabic": "Arabic (keeps tech terms in English)",
        }[l],
        help="The language the summary, action items, decisions, questions, and PDF report are written in.",
    )

    st.divider()

    # Disable the button while a request is already running, so a double
    # click (or a slow first click registering twice) can't kick off two
    # concurrent downloads writing to the same file — this is what caused
    # a "[WinError 32] file in use" failure when both runs raced on the
    # same downloaded file.
    is_processing = st.session_state.get("is_processing", False)
    process_clicked = st.button(
        "🚀 Process recording",
        type="primary",
        use_container_width=True,
        disabled=is_processing,
    )

    st.divider()
    st.caption("Whisper · Hinglish ASR · LlamaIndex · LangGraph · OpenRouter")

if process_clicked:
    if not source or not str(source).strip():
        st.sidebar.error("Give a YouTube URL or upload a file first.")
    else:
        st.session_state.is_processing = True
        try:
            with st.spinner("Running the pipeline — this can take a few minutes for longer recordings..."):
                st.session_state.state = run_new_meeting(
                    source, audio_type, transcript_language, summary_language
                )
        except Exception as _pipeline_err:
            _err_str = str(_pipeline_err)
            if "403" in _err_str or "YouTube blocked" in _err_str or "unable to download" in _err_str:
                st.error(
                    "**YouTube blocked the download** — cloud servers are often restricted by YouTube.\n\n"
                    "**Workaround (2 steps):**\n"
                    "1. Download the audio on your own PC:\n"
                    "   ```\n"
                    "   yt-dlp -x --audio-format mp3 \"PASTE_URL_HERE\"\n"
                    "   ```\n"
                    "2. Switch to **'Upload a file'** in the sidebar and upload the downloaded file."
                )
            else:
                st.exception(_pipeline_err)
        finally:
            st.session_state.is_processing = False


# --------------------------------------------------------------------------- #
# 5. Main area
# --------------------------------------------------------------------------- #

st.markdown("## 🎙️ AI Video Assistant")
st.caption("Turn a recording into a transcript, a structured summary, and a chat-ready meeting assistant.")

state = st.session_state.state

if state is None:
    st.info("👋 Add a YouTube URL or upload a file in the sidebar, then click **Process recording**.")
    st.stop()

# Fix: a blocked request used to hard-stop the whole page (st.stop()) even
# when a PREVIOUS successful run had already produced a summary/transcript/
# PDF still sitting in `state` — e.g. asking an empty/invalid follow-up
# question after a successful recording. guardrail_input only ever adds to
# `state["errors"]` and recomputes `state["blocked"]` for the CURRENT request
# (see 10_guardrails.ipynb) — it never deletes the earlier content agent's
# output from state. So the right check isn't just "is this blocked?", it's
# "do we have nothing at all to show?".
has_content = bool(str(state.get("transcript") or "").strip())

if state.get("blocked") and not has_content:
    # Nothing has ever been produced for this state (e.g. the very first
    # "process recording" request failed) — there's genuinely nothing to
    # show below, so stopping here is correct.
    st.error("This request was blocked:")
    for e in state.get("errors", []):
        st.write(f"- {e}")
    st.stop()

if state.get("blocked") and has_content:
    # Content already exists (e.g. a bad follow-up chat question) — surface
    # the error without hiding the summary/transcript/PDF that's already
    # available.
    st.error("Your last request was blocked:")
    for e in state.get("errors", []):
        st.write(f"- {e}")

# Non-blocking warnings still surface here
if state.get("errors") and not state.get("blocked"):
    with st.expander("⚠️ Warnings", expanded=True):
        for e in state["errors"]:
            st.warning(e)

summary_lang = state.get("summary_language", "english")
transcript_lang = state.get("transcript_language", "english")

st.markdown(f'<div class="report-title">{state.get("title") or "Untitled Recording"}</div>', unsafe_allow_html=True)
st.markdown(
    f'<span class="badge">Summary: {summary_lang.title()}</span>'
    f'<span class="badge">Transcript: {transcript_lang.title()}</span>'
    f'<span class="badge">Audio: {state.get("audio_type", "auto").title()}</span>',
    unsafe_allow_html=True,
)
st.write("")

col1, col2, col3 = st.columns(3)
col1.metric("Transcript length", f'{len(state.get("transcript") or "")} chars')
col2.metric("Questions asked", len(state.get("chat_history", [])))
col3.metric("PDF report", "✅ Ready" if state.get("pdf_path") else "—")

if state.get("pdf_path") and os.path.exists(state["pdf_path"]):
    with open(state["pdf_path"], "rb") as f:
        st.download_button(
            "⬇️ Download PDF report",
            data=f.read(),
            file_name="meeting_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

st.write("")
tab_summary, tab_details, tab_raw, tab_translated, tab_chat = st.tabs(
    ["📋 Summary", "✅ Action Items & Decisions", "📝 Raw Transcript", "🌐 Translated Transcript", "💬 Chat"]
)

with tab_summary:
    render_md(state.get("summary"), summary_lang)

with tab_details:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Action Items")
        render_md(state.get("action_items"), summary_lang)
    with c2:
        st.subheader("Key Decisions")
        render_md(state.get("key_decisions"), summary_lang)
    st.divider()
    st.subheader("Open Questions")
    render_md(state.get("open_questions"), summary_lang)


def _transcript_tab(text_key: str, pdf_suffix: str, caption: str, render_language: str) -> None:
    """Shared renderer for the Raw/Translated transcript tabs: a text area
    plus an on-demand PDF download. PDFs are regenerated on click rather than
    cached in state — cheap (no LLM call, just text layout) and keeps state
    small/serializable. The output path is unique per session+kind so two
    browser sessions downloading at the same time never race on one file."""
    text = state.get(text_key) or ""
    st.caption(caption)
    st.text_area("Transcript", text, height=420, label_visibility="collapsed", key=f"ta_{text_key}")

    if text.strip():
        pdf_path = os.path.join(
            tempfile.gettempdir(), f"transcript_{pdf_suffix}_{st.session_state.session_id}.pdf"
        )
        generate_transcript_pdf(
            text,
            title=state.get("title", ""),
            output_path=pdf_path,
            language=render_language,
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                f"⬇️ Download {pdf_suffix} transcript as PDF",
                data=f.read(),
                file_name=f"transcript_{pdf_suffix}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"dl_{text_key}",
            )


with tab_raw:
    # The raw transcript is auto-detected (see 02_transcriber.ipynb) and may
    # be in a different script than either chosen language — render it
    # left-to-right regardless, since we don't reliably know its script here,
    # and reshape/RTL only matters for genuinely Arabic text.
    _transcript_tab(
        "raw_transcript",
        "raw",
        "Exactly what was said, auto-detected — not translated.",
        render_language="english",
    )

with tab_translated:
    _transcript_tab(
        "transcript",
        "translated",
        f"Rewritten in the selected transcript language ({transcript_lang.title()}).",
        render_language=transcript_lang,
    )

with tab_chat:
    for turn in state.get("chat_history", []):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_md(turn["answer"], summary_lang)

    question = st.chat_input("Ask a question about this recording...")
    if question:
        with st.spinner("Thinking..."):
            st.session_state.state = ask_meeting_question(state, question)
        st.rerun()