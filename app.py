import streamlit as st
import os
import time
import base64
import uuid
import shutil
import hashlib
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google import genai
from google.genai import types
from datetime import datetime

# --- KONFIGURACJA BEZPIECZEŃSTWA ---
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

with st.sidebar:
    st.header("🔐 Autoryzacja")
    if not st.session_state["authenticated"]:
        pwd_input = st.text_input("Hasło administratora", type="password")
        if st.button("Zaloguj"):
            if pwd_input == ADMIN_PASSWORD:
                st.session_state["authenticated"] = True
                st.success("Zalogowano!")
                st.rerun()
            else:
                st.error("Błędne hasło!")
    else:
        st.success("Tryb Administratora aktywny")
        if st.button("Wyloguj"):
            st.session_state["authenticated"] = False
            st.rerun()

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
def delete_only_metadata(point_id):
    """Usuwa tylko wpis z Qdrant, pozostawiając plik na dysku nienaruszony."""
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[point_id]
        )
        return True
    except Exception as e:
        st.error(f"Błąd podczas usuwania wpisu: {e}")
        return False
    
def get_indexed_paths():
    """Pobiera listę ścieżek do zdjęć, które już są w Qdrant."""
    try:
        # Pobieramy punkty z bazy (limit ustawiony wysoko, by objąć bazę)
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        points = results[0]
        return [p.payload.get("path") for p in points if p.payload.get("path")]
    except Exception:
        return []

def get_file_hash(path):
    """Oblicza skrót (hash) pliku, by sprawdzić czy są identyczne."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def delete_point_from_db(point_id, image_path):
    OLD_IMAGES_FOLDER = "old_images"
    os.makedirs(OLD_IMAGES_FOLDER, exist_ok=True)

    try:
        # 1. Usuwamy wpis z Qdrant
        qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=[point_id])
        
        # 2. Logika bezpiecznego przenoszenia
        if image_path and os.path.exists(image_path):
            file_name = os.path.basename(image_path)
            name_part, ext_part = os.path.splitext(file_name)
            target_path = os.path.join(OLD_IMAGES_FOLDER, file_name)

            # Jeśli plik o tej nazwie już istnieje w archiwum
            if os.path.exists(target_path):
                # Porównaj zawartość
                if get_file_hash(image_path) == get_file_hash(target_path):
                    # Pliki są identyczne - możemy po prostu usunąć oryginał
                    os.remove(image_path)
                    return True
                else:
                    # Pliki są inne, ale mają tę samą nazwę - generujemy nową nazwę
                    counter = 1
                    while os.path.exists(target_path):
                        new_name = f"{name_part}_{counter}{ext_part}"
                        target_path = os.path.join(OLD_IMAGES_FOLDER, new_name)
                        counter += 1
            
            # Przenosimy (teraz target_path jest albo oryginalny, albo unikalny)
            shutil.move(image_path, target_path)
            return True
        return True
            
    except Exception as e:
        st.error(f"Błąd podczas operacji: {e}")
        return False
    
def mass_index_folder():
    """Skanuje folder i indeksuje brakujące zdjęcia."""
    indexed_paths = get_indexed_paths()
    all_files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    new_files = [f for f in all_files if f not in indexed_paths]
    return new_files

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

# Używamy tylko JEDNEGO spójnego systemu kluczy w session_state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

st.title("🔬 AI Vision Lab: Multimodal Search")

# Zakładki są zawsze widoczne, ale ich treść zależy od autoryzacji
tab1, tab2, tab3 = st.tabs(["📤 Laboratorium Analizy", "🔍 Eksplorator Wektorowy", "⚙️ Konsola Systemowa"])

with tab1:
    st.markdown("### 📷 Nowy Eksperyment Wizualny")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        use_free_model = st.toggle("🆓 Używaj darmowego modelu (Google Gemini)", value=True)
        selected_model = st.radio("Wybierz model OpenAI:", ["gpt-4o-mini", "gpt-4o"], disabled=use_free_model)
    
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
                
                if "Błąd" not in desc:
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
                        img_path = hit.payload.get("path")
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, width="stretch")
                        else:
                            st.warning("⚠️ Plik zdjęcia został przeniesiony lub usunięty.")
                            if st.session_state["authenticated"] and st.button("🧹 Usuń martwy wpis", key=f"clean_{hit.id}"):
                                delete_point_from_db(hit.id, img_path)
                                st.rerun()

                        full_desc = hit.payload.get('description', '')
                        st.markdown(f"**Opis:** {full_desc.split('---STATS---')[0]}")
                        
                        with st.expander("🔬 Dane techniczne"):
                            ts = hit.payload.get('timestamp')
                            readable_date = datetime.fromtimestamp(ts).strftime('%d.%m.%Y %H:%M') if ts else "Brak daty"
                            st.caption(f"🗓️ Data: {readable_date} | 🤖 Model: {hit.payload.get('model')} | 🎯 Score: {hit.score:.4f}")
                            
                            if st.session_state["authenticated"]:
                                col_del1, col_del2 = st.columns(2)
                                with col_del1:
                                    if st.button("🗑️ Usuń tylko opis", key=f"only_meta_{hit.id}", width="stretch"):
                                        if delete_only_metadata(hit.id):
                                            st.rerun()
                                with col_del2:
                                    if st.button("📦 Archiwizuj foto i usuń wpis", key=f"arch_{hit.id}", width="stretch"):
                                        if delete_point_from_db(hit.id, img_path):
                                            st.rerun()
                            else:
                                st.info("Zaloguj się w sidebarze, aby zarządzać tym wpisem.")

with tab3:
    if st.session_state["authenticated"]:
        st.subheader("📦 Masowe Przetwarzanie")
        st.info("Skanuje folder `uploaded_images` pod kątem nowych plików.")
        
        if st.button("🔎 Skanuj i Indeksuj Nowe Zdjęcia", width="stretch"):
            files_to_process = mass_index_folder()
            if not files_to_process:
                st.success("Brak nowych zdjęć do dodania.")
            else:
                p_bar = st.progress(0)
                for idx, f_path in enumerate(files_to_process):
                    desc = generate_image_description_free(f_path)
                    if "Błąd" not in desc:
                        save_to_vector_db(f_path, desc, "Gemini 2.5 Flash")
                    p_bar.progress((idx + 1) / len(files_to_process))
                st.rerun()

        st.divider()
        st.subheader("⚠️ Strefa Niebezpieczna")
        confirm_reset = st.checkbox("Potwierdzam chęć usunięcia danych")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧹 Czyść folder zdjęć", disabled=not confirm_reset, width="stretch"):
                for f in os.listdir(UPLOAD_FOLDER):
                    os.remove(os.path.join(UPLOAD_FOLDER, f))
                st.success("Wyczyszczono!")
        with c2:
            if st.button("🔥 RESETUJ QDRANT", disabled=not confirm_reset, width="stretch"):
                qdrant_client.delete_collection(COLLECTION_NAME)
                st.rerun()
    else:
        st.header("⚙️ Konsola Systemowa")
        st.warning("Ta sekcja jest dostępna tylko dla administratora. Zaloguj się w panelu bocznym.")