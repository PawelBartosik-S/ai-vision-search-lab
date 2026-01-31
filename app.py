import streamlit as st
import os
import time
import base64
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


st.sidebar.info("Wersja: 1.1 - Eksperymentalna Diagnostyka Tokenów")

# TODO: Dodać automatyczną migrację do Qdrant Cloud

# --- 1. KONFIGURACJA I INICJALIZACJA ---
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COLLECTION_NAME = "znajdywacz_zdjec_v2026_labs"
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(path="./qdrant_data")
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
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
    prompt = "Opisz krótko i konkretnie to zdjęcie po polsku. Skup się na faktach."

    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ],
        }]
        # Przygotowanie parametrów zapytania
        params = {
            "model": model_name, 
            "messages": messages,
        }
        
        # Logika specyficzna dla generacji modeli
        if "gpt-5" in model_name:
            # Modele GPT-5 nie wspierają temperatury innej niż 1
            params["max_completion_tokens"] = 600
        else:
            # Modele GPT-4o i starsze wspierają temperaturę i max_tokens
            params["temperature"] = 0.7
            params["max_tokens"] = 600
        
        # Zmniejszamy limit - czasem modele 5-mini przy dużym limicie 'wiszą'
        if "gpt-5" in model_name:
            params["max_completion_tokens"] = 400 
        else:
            params["max_tokens"] = 400

        response = openai_client.chat.completions.create(**params)
        
        usage = response.usage
        reasoning = getattr(usage, 'reasoning_tokens', 0)
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason # Sprawdzamy dlaczego skończył
        
        stats = f"\n\n[Statystyki | Wyjściowe: {usage.completion_tokens} | Myślenie: {reasoning} | Powód końca: {finish_reason}]"
        
        if content and len(content.strip()) > 0:
            return content + stats
        
        # Jeśli content jest pusty, ale tokeny poleciały:
        return f"Model 'przemielił' {usage.completion_tokens} tokenów, ale odmówił wypisania tekstu (Powód: {finish_reason})." + stats

    except Exception as e:
        return f"Błąd krytyczny modelu {model_name}: {str(e)}"

def get_text_embedding(text):
    # Czyścimy tekst ze statystyk przed tworzeniem embeddingu (szukamy tylko po sensie opisu)
    clean_text = text.split("[Statystyki:")[0]
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
            "model": model_used
        }
    )
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])

def search_images(query_text, limit=4):
    query_vector = get_text_embedding(query_text)
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    )
    return response.points

def reset_database():
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
    except:
        pass
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

# --- 3. INTERFEJS STREAMLIT ---

st.set_page_config(page_title="AI Vision Lab 2026", layout="wide", page_icon="🔬")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

st.title("🔬 AI Vision Lab: Foto-Wyszukiwarka")

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("Logowanie do laboratorium")
        login_input = st.text_input("Użytkownik")
        if st.button("Wejdź", width="stretch"):
            if login_input:
                st.session_state.username = login_input
                st.session_state.logged_in = True
                st.rerun()
else:
    tab1, tab2, tab3 = st.tabs(["📤 Analiza Zdjęć", "🔍 Wyszukiwarka", "⚙️ Zarządzanie"])

    with tab1:
        st.info("Wybierz model i wrzuć zdjęcie, aby zobaczyć jak AI je 'rozpracowuje'.")
        selected_model = st.selectbox(
            "Model do testów:", 
            ["gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-5", "gpt-5.1", "gpt-5.2"]
        )
        uploaded_files = st.file_uploader("Dodaj pliki", accept_multiple_files=True)
        
        if uploaded_files and st.button("Rozpocznij eksperyment", width="stretch"):
            for idx, uploaded_file in enumerate(uploaded_files):
                path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner(f'Analizuję przez {selected_model}...'):
                    desc = generate_image_description(path, selected_model)
                    
                    st.success(f"Analiza zakończona dla: {uploaded_file.name}")
                    
                    # DODAJEMY UNIKALNY KLUCZ (key) używając indeksu i nazwy pliku
                    st.text_area(
                        label=f"Wynik dla {uploaded_file.name}:", 
                        value=desc, 
                        height=150, 
                        key=f"text_area_{idx}_{uploaded_file.name}"
                    )
                    
                    save_to_vector_db(path, desc, selected_model)
            st.toast("Zdjęcia zindeksowane!")

    with tab2:
        query = st.text_input("Opisz zdjęcie, by je znaleźć:")
        if query:
            results = search_images(query)
            if results:
                cols = st.columns(2)
                for idx, hit in enumerate(results):
                    with cols[idx % 2]:
                        st.image(hit.payload["path"], width="stretch")
                        with st.expander("🔍 Metadane i Statystyki AI"):
                            st.write(f"**Użyty model:** `{hit.payload.get('model', 'N/A')}`")
                            st.write(f"**Score:** `{hit.score:.4f}`")
                            st.write(f"**Opis i Tokeny:**")
                            st.caption(hit.payload.get('description', 'Brak opisu'))
            else:
                st.warning("Brak wyników w bazie.")

    with tab3:
        st.subheader("Ustawienia systemowe")
        if st.button("🔥 CAŁKOWITY RESET BAZY", width="stretch"):
            reset_database()
            st.success("Baza wyczyszczona. Wszystkie eksperymenty usunięte.")
            st.rerun()

    with st.sidebar:
        st.write(f"Badacz: **{st.session_state.username}**")
        if st.button("Wyloguj"):
            st.session_state.logged_in = False
            st.rerun()