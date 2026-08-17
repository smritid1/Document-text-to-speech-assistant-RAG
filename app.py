import tempfile
import streamlit as st

from src.pdf_loader import read_pdf
from src.chunking import split_text_into_chunk
from src.vector_database import create_vectordb
from src.rag import ask_question
from src.schema import ChatMessage, ChatRole
from src.tts import text_to_speech


st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

st.title("📄 Simple PDF Chatbot")


# ----------------------------
# Session State
# ----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# Cache generated audio so we don't regenerate it
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}


# ----------------------------
# Upload PDF
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf",
)

if uploaded_file and st.button("Load PDF"):

    with st.spinner("Reading and indexing your PDF..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        pages = read_pdf(tmp_path)

        texts, metadatas = split_text_into_chunk(
            pages,
            pdf_file=uploaded_file.name,
        )

        st.session_state.vectorstore = create_vectordb(
            texts,
            metadatas,
        )

        # Fresh chat
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.audio_cache = {}

    st.success("PDF loaded! Ask a question below.")


# ----------------------------
# Display Chat History
# ----------------------------
for idx, message in enumerate(st.session_state.messages):

    with st.chat_message(message["role"]):

        st.write(message["content"])

        # Only assistant messages have TTS
        if message["role"] == "assistant":

            if st.button("🔊 Read Answer", key=f"tts_{idx}"):

                # Generate only once
                if idx not in st.session_state.audio_cache:

                    with st.spinner("Generating speech..."):

                        audio_path = text_to_speech(
                            message["raw_answer"]
                        )

                    st.session_state.audio_cache[idx] = audio_path

                st.audio(
                    st.session_state.audio_cache[idx],
                    autoplay=True,
                )


# ----------------------------
# Chat
# ----------------------------
if st.session_state.vectorstore is not None:

    question = st.chat_input(
        "Ask something about the PDF"
    )

    if question:

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.write(question)

        # Assistant
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = ask_question(
                    st.session_state.vectorstore,
                    question,
                    st.session_state.history,
                )

            answer_text = result.answer

            if result.pages:
                answer_text += f"\n\n📄 Source pages: {result.pages}"

            st.write(answer_text)

            # Store assistant response
            assistant_index = len(st.session_state.messages)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "raw_answer": result.answer,  # without page numbers
                }
            )

            # TTS button
            if st.button("🔊 Read Answer", key=f"tts_new_{assistant_index}"):

                with st.spinner("Generating speech..."):

                    audio_path = text_to_speech(result.answer)

                st.session_state.audio_cache[
                    assistant_index
                ] = audio_path

                st.audio(audio_path, autoplay=True)

        # Save conversation history
        st.session_state.history.append(
            ChatMessage(
                role=ChatRole.USER,
                content=result.question,
            )
        )

        st.session_state.history.append(
            ChatMessage(
                role=ChatRole.ASSISTANT,
                content=result.answer,
            )
        )

else:
    st.info(
        "Upload a PDF and click 'Load PDF' to start chatting."
    )