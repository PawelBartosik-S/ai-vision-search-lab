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

# --- 1. SŁOWNIK JĘZYKOWY ---
LANGUAGES = {
    "PL": {
        "title": "👁️ Sokole Oko AI: Magiczny Detektyw Wizualny",
        "auth_header": "🔐 Autoryzacja",
        "pwd_label": "Hasło administratora",
        "login_btn": "Zaloguj",
        "logout_btn": "Wyloguj",
        "auth_success": "✅ Zalogowano!",
        "auth_error": "❌ Błędne hasło!",
        "admin_active": "Tryb Administratora aktywny",
        "qdrant_cloud": "☁️ Połączono z Qdrant Cloud",
        "qdrant_error": "❌ Chmura nie odpowiada: ",
        "qdrant_local": "🏠 Tryb lokalny (Baza na dysku)",
        "tab1": "📤 Laboratorium Analizy",
        "tab2": "🔍 Eksplorator Wektorowy",
        "tab3": "⚙️ Konsola Systemowa",
        "toggle_free": "🆓 Używaj darmowego modelu (Google Gemini)",
        "radio_model": "Wybierz model OpenAI:",
        "uploader": "Wrzuć zdjęcia",
        "btn_process": "⚡ ROZPOCZNIJ PROCESOWANIE",
        "status_analyzing": "Analizuję {}...",
        "status_done": "Zakończono: {}",
        "status_error": "Przerwano procesowanie",
        "search_label": "Wpisz czego szukasz:",
        "auto_translate": "🌐 Przetłumacz wyniki na bieżący język",
        "dead_link": "⚠️ Zdjęcie zostało przeniesione lub usunięte.",
        "btn_clean_dead": "🧹 Usuń martwy wpis",
        "tech_data": "🔬 Dane techniczne",
        "date_label": "🗓️ Data",
        "score_label": "🎯 Score",
        "btn_del_meta": "🗑️ Usuń tylko opis",
        "btn_archive": "📦 Archiwizuj foto i usuń wpis",
        "login_info": "Zaloguj się, aby zarządzać wpisem.",
        "mass_header": "📦 Masowe Przetwarzanie",
        "mass_info": "Skanuje folder `uploaded_images` pod kątem nowych plików.",
        "scan_btn": "🔎 Skanuj i Indeksuj Nowe Zdjęcia",
        "no_new": "Brak nowych zdjęć do dodania.",
        "danger_zone": "⚠️ Strefa Niebezpieczna",
        "confirm_check": "Potwierdzam chęć zarządzania kolekcją",
        "btn_clean_folder": "🧹 Czyść folder zdjęć",
        "btn_reset_qdrant": "🔥 RESETUJ KOLEKCJĘ QDRANT",
        "desc_label": "**Opis:** ",
        "success_clean": "Wyczyszczono!",
        "error_generic": "Wystąpił błąd: ",
    },
    "EN": {
        "title": "🔬 AI Vision Lab: Multimodal Search",
        "auth_header": "🔐 Authorization",
        "pwd_label": "Admin password",
        "login_btn": "Log In",
        "logout_btn": "Log Out",
        "auth_success": "✅ Logged in!",
        "auth_error": "❌ Incorrect password!",
        "admin_active": "Admin Mode active",
        "qdrant_cloud": "☁️ Connected to Qdrant Cloud",
        "qdrant_error": "❌ Cloud not responding: ",
        "qdrant_local": "🏠 Local Mode (Disk database)",
        "tab1": "📤 Analysis Lab",
        "tab2": "🔍 Vector Explorer",
        "tab3": "⚙️ System Console",
        "toggle_free": "🆓 Use free model (Google Gemini)",
        "radio_model": "Select OpenAI model:",
        "uploader": "Upload images",
        "btn_process": "⚡ START PROCESSING",
        "status_analyzing": "Analyzing {}...",
        "status_done": "Done: {}",
        "status_error": "Processing interrupted",
        "search_label": "Type what you're looking for:",
        "auto_translate": "🌐 Translate results to current language",
        "dead_link": "⚠️ Image file moved or deleted.",
        "btn_clean_dead": "🧹 Remove dead entry",
        "tech_data": "🔬 Technical Data",
        "date_label": "🗓️ Date",
        "score_label": "🎯 Score",
        "btn_del_meta": "🗑️ Delete only description",
        "btn_archive": "📦 Archive photo and delete record",
        "login_info": "Please log in to manage this entry.",
        "mass_header": "📦 Mass Processing",
        "mass_info": "Scanning `uploaded_images` folder for new files.",
        "scan_btn": "🔎 Scan and Index New Images",
        "no_new": "No new images found.",
        "danger_zone": "⚠️ Danger Zone",
        "confirm_check": "I confirm I want to manage the collection",
        "btn_clean_folder": "🧹 Clean image folder",
        "btn_reset_qdrant": "🔥 RESET QDRANT COLLECTION",
        "desc_label": "**Description:** ",
        "success_clean": "Cleaned up!",
        "error_generic": "An error occurred: ",
    },
}

# --- 2. KONFIGURACJA I INICJALIZACJA ---
load_dotenv()
st.set_page_config(page_title="AI Vision Lab 2026", layout="wide", page_icon="🔬")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Sidebar Language Selection
with st.sidebar:
    lang_code = st.selectbox("🌐 Language", ["PL", "EN"])
    t = LANGUAGES[lang_code]

    st.divider()

    # Auth Section
    ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")
    st.header(t["auth_header"])
    if not st.session_state["authenticated"]:
        pwd_input = st.text_input(t["pwd_label"], type="password")
        if st.button(t["login_btn"]):
            if pwd_input == ADMIN_PASSWORD:
                st.session_state["authenticated"] = True
                st.success(t["auth_success"])
                st.rerun()
            else:
                st.error(t["auth_error"])
    else:
        st.success(t["admin_active"])
        if st.button(t["logout_btn"]):
            st.session_state["authenticated"] = False
            st.rerun()

# Klienci API
client_gemini = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"), http_options={"api_version": "v1beta"}
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COLLECTION_NAME = "znajdywacz_zdjec_v2026_labs"
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@st.cache_resource
def get_qdrant_client():
    q_host = os.getenv("QDRANT_HOST")
    q_api_key = os.getenv("QDRANT_API_KEY")
    client = None
    status = "local"
    err = None

    if q_host and q_api_key:
        try:
            client = QdrantClient(url=q_host, api_key=q_api_key, timeout=10)
            client.get_collections()
            status = "cloud"
        except Exception as e:
            err = str(e)
            status = "error"
            client = QdrantClient(path="./qdrant_data")
    else:
        client = QdrantClient(path="./qdrant_data")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    return client, status, err


qdrant_client, conn_status, conn_err = get_qdrant_client()

# Wyświetlanie statusu Qdrant w Sidebarze
if conn_status == "cloud":
    st.sidebar.success(t["qdrant_cloud"])
elif conn_status == "error":
    st.sidebar.error(f"{t['qdrant_error']} {conn_err}")
else:
    st.sidebar.warning(t["qdrant_local"])

# --- 3. FUNKCJE BACKENDOWE ---


def delete_only_metadata(point_id):
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME, points_selector=[point_id]
        )
        return True
    except Exception as e:
        st.error(f"{t['error_generic']} {e}")
        return False


def get_indexed_paths():
    """Pobiera listę ścieżek do zdjęć, które już są w Qdrant."""
    try:
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        points = results[0]
        return [p.payload.get("path") for p in points if p.payload.get("path")]
    except Exception:
        return []


# PRZYWRÓCONA FUNKCJA - Potrzebna do sprawdzania, czy obrazy się nie dublują
def get_file_hash(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


# PRZYWRÓCONA BEZPIECZNA FUNKCJA KASOWANIA (Oryginalna)
def delete_point_from_db(point_id, image_path):
    OLD_IMAGES_FOLDER = "old_images"
    os.makedirs(OLD_IMAGES_FOLDER, exist_ok=True)
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME, points_selector=[point_id]
        )
        if image_path and os.path.exists(image_path):
            file_name = os.path.basename(image_path)
            name_part, ext_part = os.path.splitext(file_name)
            target_path = os.path.join(OLD_IMAGES_FOLDER, file_name)

            if os.path.exists(target_path):
                if get_file_hash(image_path) == get_file_hash(target_path):
                    os.remove(image_path)
                    return True
                else:
                    counter = 1
                    while os.path.exists(target_path):
                        new_name = f"{name_part}_{counter}{ext_part}"
                        target_path = os.path.join(OLD_IMAGES_FOLDER, new_name)
                        counter += 1

            shutil.move(image_path, target_path)
        return True
    except Exception as e:
        st.error(f"{t['error_generic']} {e}")
        return False


# POPRAWIONY MODEL NA gemini-3-flash-preview
def generate_image_description_free(image_path, lang="PL"):
    prompt = (
        "Opisz krótko to zdjęcie po polsku, podaj kolory i obiekty."
        if lang == "PL"
        else "Briefly describe this photo in English, specify colors and objects."
    )
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        response = client_gemini.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )
        return f"{response.text}\n\n---STATS---\n📊 Model: Gemini 3 Flash Preview"
    except Exception as e:
        return f"Error Gemini: {str(e)}"


def generate_image_description(image_path, model_name, lang="PL"):
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    prompt = (
        "Opisz krótko to zdjęcie po polsku, podaj kolory i obiekty."
        if lang == "PL"
        else "Briefly describe this photo in English, specify colors and objects."
    )
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=500,
        )
        return (
            response.choices[0].message.content
            + f"\n\n---STATS---\n📊 Model: {model_name}"
        )
    except Exception as e:
        return f"Error OpenAI: {str(e)}"


def save_to_vector_db(image_path, description, model_used):
    clean_text = description.split("---STATS---")[0]
    emb_res = openai_client.embeddings.create(
        input=clean_text, model="text-embedding-3-small"
    )
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=emb_res.data[0].embedding,
        payload={
            "path": image_path,
            "description": description,
            "model": model_used,
            "timestamp": time.time(),
        },
    )
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])


@st.cache_data(show_spinner=False)
def translate_description(text, target_lang):
    """Tłumaczy opis na wybrany język w locie, korzystając z pamięci podręcznej (cache)."""
    if not text:
        return ""

    # Usuwamy na chwilę statystyki, żeby ich nie tłumaczyć
    clean_text = text.split("---STATS---")[0].strip()

    prompt = f"Przetłumacz poniższy opis obrazu na język {'polski' if target_lang == 'PL' else 'angielski'}. Zwróć TYLKO przetłumaczony tekst, bez żadnych dodatkowych komentarzy:\n\n{clean_text}"

    try:
        # Używamy najtańszego i najszybszego modelu do tłumaczeń
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Błąd tłumaczenia: {str(e)}] {clean_text}"


# --- 4. INTERFEJS ---
st.title(t["title"])
tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

with tab1:
    st.markdown(f"### {t['tab1']}")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        use_free_model = st.toggle(t["toggle_free"], value=True)
        selected_model = st.radio(
            t["radio_model"],
            ["gpt-4o-mini", "gpt-4o", "gpt-5.4"],
            disabled=use_free_model,
        )
    with col_m2:
        uploaded_files = st.file_uploader(t["uploader"], accept_multiple_files=True)

    if uploaded_files and st.button(t["btn_process"], width="stretch"):
        p_bar = st.progress(0)
        for idx, file in enumerate(uploaded_files):
            path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            with st.status(
                t["status_analyzing"].format(file.name), expanded=True
            ) as status:
                desc = (
                    generate_image_description_free(path, lang=lang_code)
                    if use_free_model
                    else generate_image_description(
                        path, selected_model, lang=lang_code
                    )
                )
                if "Error" not in desc:
                    # Poprawiona nazwa używanego modelu przy zapisie do wektorów (Dla Gemini 2.5 Flash)
                    save_to_vector_db(
                        path,
                        desc,
                        "Gemini 3 Flash Preview" if use_free_model else selected_model,
                    )
                    st.image(path, width=200)
                    st.write(desc)
                    status.update(
                        label=t["status_done"].format(file.name), state="complete"
                    )
                else:
                    st.error(desc)
                    status.update(label=t["status_error"], state="error")
            p_bar.progress((idx + 1) / len(uploaded_files))

with tab2:
    st.markdown(f"### {t['tab2']}")

    # Rozdzielamy pole wyszukiwania i przełącznik tłumaczenia na dwie kolumny
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        query = st.text_input(t["search_label"])
    with col_s2:
        st.write("")  # Pusty wiersz dla wyrównania w pionie
        st.write("")
        auto_translate = st.toggle(t["auto_translate"], value=False)

    if query:
        emb_query = (
            openai_client.embeddings.create(input=query, model="text-embedding-3-small")
            .data[0]
            .embedding
        )
        hits = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, query=emb_query, limit=4
        ).points

        grid = st.columns(2)
        for i, hit in enumerate(hits):
            with grid[i % 2]:
                with st.container(border=True):
                    path = hit.payload.get("path")
                    if path and os.path.exists(path):
                        st.image(path, width="stretch")
                    else:
                        st.warning(t["dead_link"])
                        if st.session_state["authenticated"] and st.button(
                            t["btn_clean_dead"], key=f"c_{hit.id}"
                        ):
                            delete_only_metadata(hit.id)
                            st.rerun()

                    # --- LOGIKA TŁUMACZENIA ---
                    original_desc = hit.payload.get("description", "").split(
                        "---STATS---"
                    )[0]

                    if auto_translate:
                        # Wywołujemy naszą funkcję (dzięki @st.cache_data nie spowolni to aplikacji)
                        display_desc = translate_description(original_desc, lang_code)
                    else:
                        display_desc = original_desc

                    st.markdown(f"{t['desc_label']} {display_desc}")
                    # --------------------------

                    with st.expander(t["tech_data"]):
                        ts = hit.payload.get("timestamp")
                        date = (
                            datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
                            if ts
                            else "---"
                        )
                        st.caption(
                            f"{t['date_label']}: {date} | 🤖: {hit.payload.get('model')} | 🎯: {hit.score:.4f}"
                        )
                        if st.session_state["authenticated"]:
                            c1, c2 = st.columns(2)
                            if c1.button(t["btn_del_meta"], key=f"m_{hit.id}"):
                                if delete_only_metadata(hit.id):
                                    st.rerun()
                            if c2.button(t["btn_archive"], key=f"a_{hit.id}"):
                                if delete_point_from_db(hit.id, path):
                                    st.rerun()
                        else:
                            st.info(t["login_info"])

with tab3:
    if st.session_state["authenticated"]:
        st.subheader(t["mass_header"])
        st.info(t["mass_info"])
        if st.button(t["scan_btn"], width="stretch"):
            indexed = get_indexed_paths()
            new = [
                os.path.join(UPLOAD_FOLDER, f)
                for f in os.listdir(UPLOAD_FOLDER)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
                and os.path.join(UPLOAD_FOLDER, f) not in indexed
            ]
            if not new:
                st.success(t["no_new"])
            else:
                pb = st.progress(0)
                for i, f_path in enumerate(new):
                    desc = generate_image_description_free(f_path, lang=lang_code)
                    if "Error" not in desc:
                        # Poprawiona nazwa zapisanego modelu z "Gemini 2.0" na "Gemini 2.5 Flash"
                        save_to_vector_db(f_path, desc, "Gemini 3 Flash Preview")
                    pb.progress((i + 1) / len(new))
                st.rerun()

        st.divider()
        st.subheader(t["danger_zone"])
        confirm = st.checkbox(t["confirm_check"])
        c1, c2 = st.columns(2)
        if c1.button(t["btn_clean_folder"], disabled=not confirm):
            for f in os.listdir(UPLOAD_FOLDER):
                os.remove(os.path.join(UPLOAD_FOLDER, f))
            st.success(t["success_clean"])
        if c2.button(t["btn_reset_qdrant"], disabled=not confirm):
            qdrant_client.delete_collection(COLLECTION_NAME)
            st.rerun()
    else:
        st.warning(t["login_info"])
