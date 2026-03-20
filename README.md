# 👁️ Sokole Oko AI / AI Vision Lab 2026

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-2.1+-red.svg)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant_Cloud-orange.svg)
![AI](https://img.shields.io/badge/Models-Gemini_3_|_GPT--5.3-green.svg)

## 🇵🇱 O Projekcie
Inteligentny system zarządzania archiwum zdjęć, wykorzystujący najnowocześniejsze modele multimodalne (LMM) do automatycznego indeksowania i semantycznego przeszukiwania obrazów. Projekt stworzony w ramach zaawansowanego modułu kursu Data Science.

### 🔥 Kluczowe funkcjonalności:
- **Wielojęzyczność (PL/EN):** Pełne wsparcie interfejsu i generowania opisów w dwóch językach.
- **Multimodalność 2026:** Integracja z **Google Gemini 3 Flash** (darmowy tier) oraz **OpenAI GPT-5.3 Pro**.
- **Wyszukiwanie Wektorowe:** Błyskawiczna analiza semantyczna dzięki **Qdrant Cloud**.
- **Zarządzanie Archiwum:** Funkcje masowego skanowania folderów, archiwizacji zdjęć i czyszczenia metadanych.
- **Optymalizacja:** Pełna obsługa `reasoning_tokens` i monitorowanie kosztów API.

---

## 🇬🇧 About The Project
An intelligent image archive management system that leverages cutting-edge multimodal models (LMM) for automatic indexing and semantic image retrieval. Developed as part of an advanced Data Science curriculum.

### 🔥 Key Features:
- **Bilingual Support (PL/EN):** Full UI and AI description support in both Polish and English.
- **2026 Multimodality:** Native integration with **Google Gemini 3 Flash** and **OpenAI GPT-5.3 Pro**.
- **Vector Search:** Instant semantic analysis powered by **Qdrant Cloud**.
- **Archive Management:** Bulk folder scanning, image archiving, and metadata management tools.
- **Optimization:** Full support for `reasoning_tokens` and real-time API cost monitoring.

---

## 🛠️ Stos technologiczny / Tech Stack
- **AI/ML:** Google GenAI (Gemini 3), OpenAI API (GPT-5.3, text-embedding-4-large)
- **Vector DB:** Qdrant (Cloud Hybrid Search)
- **Frontend:** Streamlit 2.x (Modern UI with width-stretch support)
- **DevOps:** Python 3.11+, Ruff (Linting), Git

## ⚙️ Instalacja / Installation

1. **Sklonuj repozytorium / Clone the repo:**
   ```bash
   git clone [https://github.com/TWOJA-NAZWA/Znajdywacz-zdjec.git](https://github.com/TWOJA-NAZWA/Znajdywacz-zdjec.git)
   cd Znajdywacz-zdjec
   # 👁️ Sokole Oko AI / AI Vision Lab 2026

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-2.1+-red.svg)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant_Cloud-orange.svg)
![AI](https://img.shields.io/badge/Models-Gemini_3_|_GPT--5.3-green.svg)

## 🇵🇱 O Projekcie
Inteligentny system zarządzania archiwum zdjęć, wykorzystujący najnowocześniejsze modele multimodalne (LMM) do automatycznego indeksowania i semantycznego przeszukiwania obrazów. Projekt stworzony w ramach zaawansowanego modułu kursu Data Science.

### 🔥 Kluczowe funkcjonalności:
- **Wielojęzyczność (PL/EN):** Pełne wsparcie interfejsu i generowania opisów w dwóch językach.
- **Multimodalność 2026:** Integracja z **Google Gemini 3 Flash** (darmowy tier) oraz **OpenAI GPT-5.3 Pro**.
- **Wyszukiwanie Wektorowe:** Błyskawiczna analiza semantyczna dzięki **Qdrant Cloud**.
- **Zarządzanie Archiwum:** Funkcje masowego skanowania folderów, archiwizacji zdjęć i czyszczenia metadanych.
- **Optymalizacja:** Pełna obsługa `reasoning_tokens` i monitorowanie kosztów API.

---

## 🇬🇧 About The Project
An intelligent image archive management system that leverages cutting-edge multimodal models (LMM) for automatic indexing and semantic image retrieval. Developed as part of an advanced Data Science curriculum.

### 🔥 Key Features:
- **Bilingual Support (PL/EN):** Full UI and AI description support in both Polish and English.
- **2026 Multimodality:** Native integration with **Google Gemini 3 Flash** and **OpenAI GPT-5.3 Pro**.
- **Vector Search:** Instant semantic analysis powered by **Qdrant Cloud**.
- **Archive Management:** Bulk folder scanning, image archiving, and metadata management tools.
- **Optimization:** Full support for `reasoning_tokens` and real-time API cost monitoring.

---

## 🛠️ Stos technologiczny / Tech Stack
- **AI/ML:** Google GenAI (Gemini 3), OpenAI API (GPT-5.3, text-embedding-4-large)
- **Vector DB:** Qdrant (Cloud Hybrid Search)
- **Frontend:** Streamlit 2.x (Modern UI with width-stretch support)
- **DevOps:** Python 3.11+, Ruff (Linting), Git

## ⚙️ Instalacja / Installation

1. **Sklonuj repozytorium / Clone the repo:**
   ```bash
   git clone [https://github.com/TWOJA-NAZWA/Znajdywacz-zdjec.git](https://github.com/TWOJA-NAZWA/Znajdywacz-zdjec.git)
   cd Znajdywacz-zdjec

2. Środowisko i zależności / Environment & Dependencies
Zalecane użycie środowiska wirtualnego / Virtual environment recommended:
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

3. Konfiguracja zmiennych / Environment Variables
Utwórz plik .env w głównym katalogu / Create a .env file in the root directory:

GOOGLE_API_KEY=your_gemini_3_key
OPENAI_API_KEY=your_gpt_key
QDRANT_HOST=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key

Ważne: Plik .env jest ignorowany przez git dla Twojego bezpieczeństwa.
Important: The .env file is ignored by git for your security.

Następnie utwórz folder .streamlit i plik secrets.toml dla hasła admina / Create .streamlit/secrets.toml for admin password:
# .streamlit/secrets.toml for admin password:
ADMIN_PASSWORD = "twoje_tajne_haslo"
ADMIN_PASSWORD = "your_secret_password"

Wskazówka: Oba te pliki powinny być dodane do .gitignore, aby nie wyciekły do sieci.

Tip: Both of these files should be added to .gitignore to prevent them from being leaked online.

4. Uruchomienie aplikacji / Run the app

streamlit run app.py

⚖️ Notka prawna / Legal Notice

Logotypy Google Gemini oraz OpenAI wykorzystane w grafice podglądu (Social Preview) są znakami towarowymi należącymi do ich odpowiednich właścicieli. Zostały użyte wyłącznie w celach informacyjnych, aby wskazać na integrację technologiczną projektu z tymi modelami AI. Projekt ma charakter edukacyjny i niekomercyjny.

The Google Gemini and OpenAI logos used in the Social Preview image are trademarks of their respective owners. They are used for informational purposes only to indicate the project's technological integration with these AI models. The project is educational and non-commercial.