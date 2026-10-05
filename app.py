import os
import re
import json
import hashlib

import streamlit as st
import chromadb

from dotenv import load_dotenv
from pypdf import PdfReader
from PIL import Image

from google import genai
from google.genai import types


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="CampusLens AI",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# 2. API KEY
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "GEMINI_API_KEY was not found.\n\n"
        "Please create a .env file and add:\n\n"
        "GEMINI_API_KEY=your_api_key"
    )
    st.stop()


client = genai.Client(api_key=API_KEY)


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

GENERATION_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "gemini-embedding-001"


# ============================================================
# 4. CHROMA DATABASE
# ============================================================

@st.cache_resource
def get_chroma_collection():

    db = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = db.get_or_create_collection(
        name="campuslens_documents",
        metadata={
            "hnsw:space": "cosine"
        }
    )

    return collection


collection = get_chroma_collection()


# ============================================================
# 5. GEMINI TEXT EMBEDDING
# ============================================================

def create_embedding(text):

    try:

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )

        return result.embeddings[0].values

    except Exception as e:

        raise RuntimeError(
            f"Embedding error: {str(e)}"
        )


def create_query_embedding(text):

    try:

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )

        return result.embeddings[0].values

    except Exception as e:

        raise RuntimeError(
            f"Query embedding error: {str(e)}"
        )


# ============================================================
# 6. GEMINI TEXT GENERATION
# ============================================================

def generate_text(prompt):

    try:

        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

        if response.text:
            return response.text

        return "The AI did not generate a response."

    except Exception as e:

        error_message = str(e)

        if "404" in error_message:
            return (
                "Gemini model is not available for this API key. "
                "Please check the available Gemini models."
            )

        if "401" in error_message or "403" in error_message:
            return (
                "Gemini API authentication failed. "
                "Please check your API key."
            )

        return f"Gemini error: {error_message}"


# ============================================================
# 7. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text and text.strip():

            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


# ============================================================
# 8. TEXT CHUNKING
# ============================================================

def create_chunks(pages, chunk_size=1200, overlap=200):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        words = text.split()

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk_text = " ".join(
                words[start:end]
            )

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# ============================================================
# 9. CREATE UNIQUE ID
# ============================================================

def create_document_id(filename, chunk_number):

    raw = f"{filename}_{chunk_number}"

    return hashlib.md5(
        raw.encode()
    ).hexdigest()


# ============================================================
# 10. PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    pages = extract_pdf_text(uploaded_file)

    if not pages:

        return 0, 0

    chunks = create_chunks(pages)

    added_chunks = 0

    progress = st.progress(0)

    for index, chunk in enumerate(chunks):

        chunk_text = chunk["text"]

        try:

            embedding = create_embedding(
                chunk_text
            )

            document_id = create_document_id(
                uploaded_file.name,
                index
            )

            collection.upsert(
                ids=[document_id],

                documents=[chunk_text],

                embeddings=[embedding],

                metadatas=[{
                    "filename": uploaded_file.name,
                    "page": chunk["page"],
                    "chunk": index
                }]
            )

            added_chunks += 1

            progress.progress(
                (index + 1) / len(chunks)
            )

        except Exception as e:

            st.warning(
                f"Could not process chunk {index + 1}: {e}"
            )

    progress.empty()

    return len(pages), added_chunks


# ============================================================
# 11. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(question, number_of_results=5):

    total_documents = collection.count()

    if total_documents == 0:
        return []

    query_embedding = create_query_embedding(
        question
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            number_of_results,
            total_documents
        ),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = []

    if not results["documents"]:
        return documents

    for index, document in enumerate(
        results["documents"][0]
    ):

        metadata = results["metadatas"][0][index]

        documents.append({
            "text": document,
            "page": metadata.get("page", "?"),
            "filename": metadata.get(
                "filename",
                "Unknown"
            ),
            "distance": results["distances"][0][index]
        })

    return documents


# ============================================================
# 12. RAG QUESTION ANSWERING
# ============================================================

def answer_question(question):

    documents = retrieve_documents(question)

    if not documents:

        return (
            "No study material is available. "
            "Please upload a PDF first."
        ), []

    context_parts = []

    for document in documents:

        context_parts.append(
            f"""
SOURCE: {document['filename']}
PAGE: {document['page']}

{document['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are CampusLens, an AI study assistant.

Answer the student's question using the
provided study material.

IMPORTANT RULES:

1. Use the supplied context as the primary source.
2. Do not invent information.
3. If the answer is not available in the
   supplied material, say so clearly.
4. Explain difficult concepts in simple language.
5. Give examples when useful.
6. Mention the relevant source page numbers.

STUDY MATERIAL:

{context}


STUDENT QUESTION:

{question}


Give the answer in a clear educational format.
"""

    answer = generate_text(prompt)

    return answer, documents


# ============================================================
# 13. DOCUMENT SUMMARY
# ============================================================

def generate_summary():

    total = collection.count()

    if total == 0:

        return (
            "Please upload a PDF before generating "
            "a summary."
        )

    results = collection.get(
        limit=min(total, 30),
        include=["documents", "metadatas"]
    )

    documents = results["documents"]

    combined_text = "\n\n".join(
        documents
    )

    prompt = f"""
You are an AI study assistant.

Create an exam-friendly summary of the
following study material.

Include:

1. Introduction
2. Important concepts
3. Definitions
4. Key points
5. Important examples
6. Short revision points
7. Conclusion

Use simple English.

STUDY MATERIAL:

{combined_text}
"""

    return generate_text(prompt)


# ============================================================
# 14. QUIZ GENERATOR
# ============================================================

def generate_quiz():

    total = collection.count()

    if total == 0:
        return []

    results = collection.get(
        limit=min(total, 30),
        include=["documents"]
    )

    study_text = "\n\n".join(
        results["documents"]
    )

    prompt = f"""
You are an educational quiz generator.

Create exactly 5 multiple-choice questions
from the study material.

Each question must contain:

- question
- four options
- correct answer
- explanation

Return ONLY valid JSON.

Use this exact format:

[
  {{
    "question": "Question",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Option A",
    "explanation": "Explanation"
  }}
]

STUDY MATERIAL:

{study_text}
"""

    response = generate_text(prompt)

    try:

        match = re.search(
            r"\[.*\]",
            response,
            re.DOTALL
        )

        if not match:
            return []

        quiz_data = json.loads(
            match.group()
        )

        valid_questions = []

        for question in quiz_data:

            if not isinstance(
                question,
                dict
            ):
                continue

            if not all(
                key in question
                for key in [
                    "question",
                    "options",
                    "answer",
                    "explanation"
                ]
            ):
                continue

            if len(question["options"]) != 4:
                continue

            if question["answer"] not in question["options"]:
                continue

            valid_questions.append(
                question
            )

        return valid_questions[:5]

    except Exception:

        return []


# ============================================================
# 15. IMAGE ANALYZER
# ============================================================

def analyze_image(image_file, question):

    try:

        image_bytes = image_file.getvalue()

        mime_type = image_file.type

        prompt = f"""
You are CampusLens, a multimodal AI study assistant.

Analyze the uploaded educational image.

Student question:

{question}

Explain the image clearly and simply.

If it contains a:

- diagram, explain the components
- flowchart, explain the flow
- graph, explain what it represents
- technical image, explain the important concepts
- handwritten note, explain the readable content

Do not invent details that cannot be seen.
"""

        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
            ]
        )

        return response.text

    except Exception as e:

        return f"Image analysis error: {e}"


# ============================================================
# 16. SIDEBAR
# ============================================================

st.sidebar.title("🎓 CampusLens")

st.sidebar.write(
    "AI-Powered Smart Study Assistant"
)

menu = st.sidebar.radio(
    "Choose a feature",
    [
        "🏠 Home",
        "📚 Upload Notes",
        "💬 Ask AI",
        "📝 Summary",
        "🧠 Quiz",
        "🖼️ Image Analyzer"
    ]
)


# ============================================================
# 17. HOME
# ============================================================

if menu == "🏠 Home":

    st.title("🎓 CampusLens AI")

    st.subheader(
        "Your Personal AI Study Assistant"
    )

    st.write(
        """
        CampusLens helps students understand
        their study materials using AI, RAG,
        document search and multimodal AI.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Stored Chunks",
            collection.count()
        )

    with col2:

        st.metric(
            "AI Features",
            "5"
        )

    with col3:

        st.metric(
            "RAG",
            "Enabled"
        )

    st.divider()

    st.markdown(
        """
        ### Features

        📚 **PDF Study Material**

        Upload your notes and textbooks.

        💬 **Ask AI**

        Ask questions based on your uploaded material.

        📝 **Smart Summary**

        Convert lengthy notes into revision material.

        🧠 **AI Quiz**

        Generate practice MCQs.

        🖼️ **Image Analyzer**

        Upload diagrams and educational images.
        """
    )


# ============================================================
# 18. UPLOAD NOTES
# ============================================================

elif menu == "📚 Upload Notes":

    st.title("📚 Upload Study Material")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        st.info(
            f"Selected file: {uploaded_file.name}"
        )

        if st.button(
            "🚀 Process PDF",
            type="primary"
        ):

            with st.spinner(
                "Reading and indexing your PDF..."
            ):

                try:

                    pages, chunks = process_pdf(
                        uploaded_file
                    )

                    st.success(
                        f"""
                        PDF processed successfully!

                        Pages: {pages}

                        Chunks created: {chunks}
                        """
                    )

                except Exception as e:

                    st.error(
                        f"PDF processing failed: {e}"
                    )


# ============================================================
# 19. ASK AI
# ============================================================

elif menu == "💬 Ask AI":

    st.title("💬 Ask AI About Your Notes")

    if collection.count() == 0:

        st.warning(
            "Please upload and process a PDF first."
        )

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: Explain Booth's "
            "multiplication algorithm."
        )
    )

    if st.button(
        "🔍 Ask CampusLens",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching your notes and generating answer..."
            ):

                answer, sources = answer_question(
                    question
                )

            st.subheader("🤖 AI Answer")

            st.write(answer)

            if sources:

                st.subheader(
                    "📚 Sources Used"
                )

                for source in sources:

                    st.caption(
                        f"📄 {source['filename']} "
                        f"| Page {source['page']}"
                    )


# ============================================================
# 20. SUMMARY
# ============================================================

elif menu == "📝 Summary":

    st.title("📝 AI Study Summary")

    if collection.count() == 0:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        if st.button(
            "✨ Generate Summary",
            type="primary"
        ):

            with st.spinner(
                "Creating your summary..."
            ):

                summary = generate_summary()

            st.markdown(summary)


# ============================================================
# 21. QUIZ
# ============================================================

elif menu == "🧠 Quiz":

    st.title("🧠 AI Practice Quiz")

    if collection.count() == 0:

        st.warning(
            "Please upload study material first."
        )

    else:

        if st.button(
            "🎯 Generate Quiz",
            type="primary"
        ):

            with st.spinner(
                "Generating questions..."
            ):

                quiz = generate_quiz()

            if quiz:

                st.session_state[
                    "quiz"
                ] = quiz

                st.session_state[
                    "quiz_submitted"
                ] = False

                st.success(
                    "Quiz generated successfully!"
                )

            else:

                st.error(
                    "Could not generate the quiz. "
                    "Please try again."
                )

        if "quiz" in st.session_state:

            quiz = st.session_state["quiz"]

            answers = []

            st.divider()

            for index, question in enumerate(
                quiz
            ):

                st.markdown(
                    f"""
                    ### Question {index + 1}

                    {question['question']}
                    """
                )

                selected = st.radio(
                    "Choose an answer:",
                    question["options"],
                    key=f"quiz_{index}"
                )

                answers.append(
                    selected
                )

            st.divider()

            if st.button(
                "✅ Submit Quiz"
            ):

                score = 0

                for question, answer in zip(
                    quiz,
                    answers
                ):

                    if answer == question["answer"]:
                        score += 1

                st.session_state[
                    "quiz_score"
                ] = score

                st.session_state[
                    "quiz_submitted"
                ] = True

            if st.session_state.get(
                "quiz_submitted",
                False
            ):

                score = st.session_state[
                    "quiz_score"
                ]

                st.success(
                    f"🎉 Your Score: "
                    f"{score}/{len(quiz)}"
                )

                for index, question in enumerate(
                    quiz
                ):

                    st.write(
                        f"**Q{index + 1}: "
                        f"Correct Answer: "
                        f"{question['answer']}**"
                    )

                    st.caption(
                        question["explanation"]
                    )


# ============================================================
# 22. IMAGE ANALYZER
# ============================================================

elif menu == "🖼️ Image Analyzer":

    st.title(
        "🖼️ Multimodal AI Image Analyzer"
    )

    st.write(
        "Upload a diagram, chart, screenshot "
        "or educational image."
    )

    image_file = st.file_uploader(
        "Upload Image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ]
    )

    question = st.text_input(
        "What would you like to know?",
        placeholder=(
            "Explain this diagram in simple words."
        )
    )

    if image_file:

        image = Image.open(
            image_file
        )

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "🔎 Analyze Image",
            type="primary"
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                result = analyze_image(
                    image_file,
                    question or
                    "Explain this image."
                )

            st.subheader(
                "🤖 AI Explanation"
            )

            st.write(result)


# ============================================================
# 23. FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "CampusLens AI | Full Stack AI Project"
)