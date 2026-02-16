import streamlit as st
import os
import time
import base64
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 1. KONFIGURACJA I INICJALIZACJA ---
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COLLECTION_NAME = "znajdywacz_zdjec_v2026_labs"
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@st.cache_resource
def get_qdrant_client():
    # --- POPRAWKA: LOGIKA HYBRYDOWA (CLOUD / LOCAL) ---
    q_host = os.getenv("QDRANT_HOST")
    q_api_key = os.getenv("QDRANT_API_KEY")

    if q_host and q_api_key:
        # Łączymy z Qdrant Cloud
        client = QdrantClient(url=q_host, api_key=q_api_key)
        st.sidebar.success("☁️ Połączono z Qdrant Cloud")
    else:
        # Tryb awaryjny: lokalny
        client = QdrantClient(path="./qdrant_data")
        st.sidebar.warning("🏠 Tryb lokalny (Brak kluczy Cloud)")

    # Automatyczne tworzenie kolekcji jeśli nie istnieje
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    return client

qdrant_client = get_qdrant_client()

# --- 2. FUNKCJE BACKENDOWE (LOGIKA AI) ---

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_description(image_path, model_name):
    base64_image = encode_image(image_path)
    # BAJER: Bardziej szczegółowy prompt dla lepszej jakości wyszukiwania
    prompt = """Jesteś ekspertem analizy obrazu. Opisz to zdjęcie po polsku, 
    uwzględniając: 1. Główny temat, 2. Kolorystykę, 3. Kontekst/Emocje, 4. Detale tła. 
    Bądź konkretny, ale obrazowy."""

    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ],
        }]
        
        params = {"model": model_name, "messages": messages}
        
        # Specyficzne ustawienia dla modeli 2026 (GPT-5 family)
        if "gpt-5" in model_name:
            params["max_completion_tokens"] = 500
        else:
            params["temperature"] = 0.7
            params["max_tokens"] = 500

        response = openai_client.chat.completions.create(**params)
        
        usage = response.usage
        reasoning = getattr(usage, 'reasoning_tokens', 0)
        content = response.choices[0].message.content
        
        # BAJER: Separator statystyk dla czystości embeddingu
        stats_separator = "---STATS---"
        stats = f"{stats_separator}\n📊 Model: {model_name} | 🧠 Myślenie: {reasoning} | ⚡ Tokeny: {usage.total_tokens}"
        
        return content + "\n\n" + stats

    except Exception as e:
        return f"Błąd modelu {model_name}: {str(e)}"

def get_text_embedding(text):
    # Czyścimy tekst ze statystyk, by embedding był czysty (tylko sens zdjęcia)
    clean_text = text.split("---STATS---")[0]
    response = openai_client.embeddings.create(
        input=clean_text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def save_to_vector_db(image_path, description, model_used):
    embedding = get_text_embedding(description)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={
            "path": image_path, 
            "description": description,
            "model": model_used,
            "timestamp": time.time()
        }
    )
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])

def search_images(query_text, limit=4):
    query_vector = get_text_embedding(query_text)
    # Zmieniono na query_points dla nowszych wersji biblioteki
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    )
    return response.points

# --- 3. INTERFEJS STREAMLIT ---

st.set_page_config(page_title="AI Vision Lab 2026", layout="wide", page_icon="🔬")

# Custom CSS dla bajeranckiego wyglądu
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stTextArea textarea { background-color: #1e2130; color: #00ffcc; font-family: 'Courier New', monospace; }
    </style>
    """, unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

st.title("🔬 AI Vision Lab: Multimodal Search")

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("Autoryzacja Badacza")
        login_input = st.text_input("Podaj identyfikator")
        if st.button("Uruchom System", use_container_width=True):
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
            selected_model = st.radio(
                "Wybierz silnik AI:", 
                ["gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-5", "gpt-5.2"]
            )
        with col_m2:
            uploaded_files = st.file_uploader("Wrzuć zdjęcia do analizy", accept_multiple_files=True)
        
        if uploaded_files and st.button("⚡ ROZPOCZNIJ PROCESOWANIE", use_container_width=True):
            progress_bar = st.progress(0)
            for idx, uploaded_file in enumerate(uploaded_files):
                path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.status(f"Analizuję {uploaded_file.name}...", expanded=True) as status:
                    desc = generate_image_description(path, selected_model)
                    save_to_vector_db(path, desc, selected_model)
                    st.image(path, width=200)
                    st.write(desc)
                    status.update(label=f"Zakończono: {uploaded_file.name}", state="complete")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            st.balloons()

    with tab2:
        st.markdown("### 🔍 Wyszukiwanie Semantyczne")
        query = st.text_input("Wpisz czego szukasz (np. 'zachód słońca nad morzem' lub 'zdjęcie z dużą ilością zieleni')")
        
        if query:
            results = search_images(query)
            if results:
                # Bajer: dynamiczny układ siatki
                grid = st.columns(2)
                for idx, hit in enumerate(results):
                    with grid[idx % 2]:
                        # Kontener na każde zdjęcie dla estetyki
                        with st.container(border=True):
                            st.image(hit.payload["path"], use_container_width=True)
                            # Rozdzielamy opis od statystyk do wyświetlenia
                            full_desc = hit.payload.get('description', '')
                            clean_desc = full_desc.split("---STATS---")[0]
                            stats_desc = full_desc.split("---STATS---")[-1] if "---STATS---" in full_desc else ""
                            
                            st.markdown(f"**Opis:** {clean_desc}")
                            with st.expander("🔬 Dane techniczne"):
                                st.caption(f"Prawdopodobieństwo (Score): {hit.score:.4f}")
                                st.caption(stats_desc)
            else:
                st.warning("Baza milczy. Brak pasujących wektorów.")

    with tab3:
        st.subheader("Zarządzanie bazą")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🧹 Czyść lokalny folder zdjęć"):
                for f in os.listdir(UPLOAD_FOLDER):
                    os.remove(os.path.join(UPLOAD_FOLDER, f))
                st.success("Folder wyczyszczony.")
        with col_c2:
            if st.button("🔥 RESETUJ KOLEKCJĘ QDRANT"):
                qdrant_client.delete_collection(COLLECTION_NAME)
                st.rerun()

    with st.sidebar:
        st.markdown(f"---")
        st.markdown(f"Zalogowany: **{st.session_state.username}**")
        if st.button("Wyloguj system"):
            st.session_state.logged_in = False
            st.rerun()