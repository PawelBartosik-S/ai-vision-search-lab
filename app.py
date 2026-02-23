import streamlit as st
import os
import time
import base64
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests
from google import genai
from google.genai import types
from datetime import datetime


# --- 1. KONFIGURACJA I INICJALIZACJA ---
load_dotenv()
client_gemini = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1'} # Wymuszamy wersję stabilną zamiast v1beta
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HF_TOKEN = os.getenv("HF_TOKEN")
COLLECTION_NAME = "znajdywacz_zdjec_v2026_labs"
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@st.cache_resource
def get_qdrant_client():
    q_host = os.getenv("QDRANT_HOST")
    q_api_key = os.getenv("QDRANT_API_KEY")
    client = None

    if q_host and q_api_key:
        try:
            client = QdrantClient(url=q_host, api_key=q_api_key, timeout=10)
            client.get_collections()
            st.sidebar.success("☁️ Połączono z Qdrant Cloud")
        except Exception as e:
            st.sidebar.error(f"❌ Chmura nie odpowiada: {str(e)}")
            client = QdrantClient(path="./qdrant_data")
    else:
        st.sidebar.warning("🏠 Tryb lokalny (Brak danych w .env)")
        client = QdrantClient(path="./qdrant_data")

    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
    except Exception as e:
        st.error(f"Błąd podczas tworzenia kolekcji: {e}")
    return client

qdrant_client = get_qdrant_client()

# --- 2. FUNKCJE BACKENDOWE ---

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def generate_image_description_free(image_path):
    if not os.getenv("GOOGLE_API_KEY"):
        return "Błąd: Brak GOOGLE_API_KEY"

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # POPRAWKA: Używamy aktualnie wspieranej, stabilnej nazwy modelu bez sufiksu '-latest'
        response = client_gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "Opisz krótko to zdjęcie po polsku, podaj kolory i obiekty."
            ]
        )
        
        if response.text:
           return f"{response.text}\n\n---STATS---\n📊 Model: Gemini 2.5 Flash"
        return "Błąd: Brak tekstu w odpowiedzi."
    except Exception as e:  # <--- TO TEGO BRAKOWAŁO!
        return f"Błąd Gemini: {str(e)}"
def generate_image_description(image_path, model_name):
    base64_image = encode_image(image_path)
    prompt = "Opisz to zdjęcie po polsku, podaj konkretne detale."
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ],
        }]
        response = openai_client.chat.completions.create(model=model_name, messages=messages, max_tokens=500)
        content = response.choices[0].message.content
        return content + f"\n\n---STATS---\n📊 Model: {model_name} | ⚡ Tokeny: {response.usage.total_tokens}"
    except Exception as e:
        return f"Błąd modelu OpenAI: {str(e)}"

def get_text_embedding(text):
    clean_text = text.split("---STATS---")[0]
    response = openai_client.embeddings.create(input=clean_text, model="text-embedding-3-small")
    return response.data[0].embedding

def save_to_vector_db(image_path, description, model_used):
    embedding = get_text_embedding(description)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"path": image_path, "description": description, "model": model_used, "timestamp": time.time()}
    )
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])

def search_images(query_text, limit=4):
    query_vector = get_text_embedding(query_text)
    response = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=limit)
    return response.points

# --- 3. INTERFEJS STREAMLIT ---
st.set_page_config(page_title="AI Vision Lab 2026", layout="wide", page_icon="🔬")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

st.title("🔬 AI Vision Lab: Multimodal Search")

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("Autoryzacja Badacza")
        login_input = st.text_input("Podaj identyfikator")
        if st.button("Uruchom System", width="stretch"):
            if login_input:
                st.session_state.username = login_input
                st.session_state.logged_in = True
                st.rerun()
else:
    tab1, tab2, tab3 = st.tabs(["📤 Laboratorium Analizy", "🔍 Eksplorator Wektorowy", "⚙️ Konsola Systemowa"])

    with tab1:
        st.markdown("### 📷 Nowy Eksperyment Wizualny")
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            use_free_model = st.toggle("🆓 Używaj darmowego modelu (Google Gemini)", value=False)
            selected_model = st.radio("Wybierz silnik OpenAI:", ["gpt-4o-mini", "gpt-4o"], disabled=use_free_model)
        
        with col_m2:
            uploaded_files = st.file_uploader("Wrzuć zdjęcia", accept_multiple_files=True)
        
        if uploaded_files and st.button("⚡ ROZPOCZNIJ PROCESOWANIE", width="stretch"):
            progress_bar = st.progress(0)
            for idx, uploaded_file in enumerate(uploaded_files):
                path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.status(f"Analizuję {uploaded_file.name}...", expanded=True) as status:
                    if use_free_model:
                        desc = generate_image_description_free(path)
                        model_tag = "Gemini 2.5 Flash"
                    else:
                        desc = generate_image_description(path, selected_model)
                        model_tag = selected_model
                    
                    if "Błąd" not in desc and "⏳" not in desc:
                        save_to_vector_db(path, desc, model_tag)
                        st.image(path, width=200)
                        st.write(desc)
                        status.update(label=f"Zakończono: {uploaded_file.name}", state="complete")
                    else:
                        st.error(desc)
                        status.update(label="Przerwano procesowanie", state="error")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            st.balloons()

    with tab2:
        st.markdown("### 🔍 Wyszukiwanie Semantyczne")
        query = st.text_input("Wpisz czego szukasz:")
        if query:
            results = search_images(query)
            if results:
                grid = st.columns(2)
                for idx, hit in enumerate(results):
                    with grid[idx % 2]:
                        with st.container(border=True):
                            st.image(hit.payload["path"], width="stretch")
                            full_desc = hit.payload.get('description', '')
                            st.markdown(f"**Opis:** {full_desc.split('---STATS---')[0]}")
                            with st.expander("🔬 Dane techniczne"):
                                # Pobieramy timestamp z payloadu
                                ts = hit.payload.get('timestamp')
                                if ts:
                                    # Zamieniamy sekundową liczbę na format: DD.MM.YYYY HH:MM
                                    readable_date = datetime.fromtimestamp(ts).strftime('%d.%m.%Y %H:%M')
                                else:
                                    readable_date = "Brak daty"

                                # Wyświetlamy rozszerzone statystyki
                                st.caption(f"🗓️ Data: {readable_date}")
                                st.caption(f"🤖 Model: {hit.payload.get('model')}")
                                st.caption(f"🎯 Score: {hit.score:.4f}")
    with tab3:
        st.subheader("Zarządzanie bazą")
        st.warning("⚠️ Uwaga: Operacje poniżej są nieodwracalne.")
        confirm_reset = st.checkbox("Potwierdzam, że chcę zarządzać kolekcją")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🧹 Czyść lokalny folder zdjęć", disabled=not confirm_reset):
                for f in os.listdir(UPLOAD_FOLDER):
                    os.remove(os.path.join(UPLOAD_FOLDER, f))
                st.success("Folder wyczyszczony.")
        with col_c2:
            if st.button("🔥 RESETUJ KOLEKCJĘ QDRANT", disabled=not confirm_reset):
                qdrant_client.delete_collection(COLLECTION_NAME)
                st.success("Kolekcja usunięta.")
                st.rerun()