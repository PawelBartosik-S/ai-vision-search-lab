# 📸 Znajdywacz Zdjęć 2026 (AI Image Search Lab)

### Projekt na zaliczenie przedostatniego modułu kursu Data Science

System inteligentnego zarządzania archiwum zdjęć, który wykorzystuje multimodalne modele Large Language Models (LLM) do automatycznego opisywania obrazów oraz bazę wektorową do błyskawicznego wyszukiwania semantycznego.

## 🚀 Główne funkcjonalności
- **Multimodalna Analiza:** Wykorzystanie modeli `gpt-4o` oraz `gpt-5` do generowania szczegółowych opisów (detekcja obiektów, tekstów i kontekstu).
- **Wyszukiwanie Semantyczne:** Zastosowanie bazy wektorowej **Qdrant** oraz modelu `text-embedding-3-small` do wyszukiwania po sensie zapytania (a nie tylko słowach kluczowych).
- **Monitorowanie Tokenów:** Diagnostyka procesu wnioskowania dzięki podglądowi `reasoning_tokens` (specyficznych dla modeli GPT-5).
- **Interfejs Streamlit:** Intuicyjny panel podzielony na procesowanie danych, wyszukiwarkę oraz zarządzanie bazą.

## 🛠️ Stos technologiczny
- **Język:** Python 3.10+
- **AI/ML:** OpenAI API (Vision, Embeddings)
- **Baza danych:** Qdrant (Vector Database)
- **Frontend:** Streamlit
- **Zarządzanie projektem:** Trello (Metodyka Agile/Scrum)

## 📋 Struktura projektu
- `app.py` - główna aplikacja Streamlit
- `uploaded_images/` - lokalne archiwum przetworzonych zdjęć
- `qdrant_data/` - lokalna persystencja bazy wektorowej
- `.env` - klucze API (plik pominięty w repozytorium)

## ⚙️ Instalacja i uruchomienie
1. Sklonuj repozytorium:
   ```bash
   git clone [https://github.com/TWOJA-NAZWA/Znajdywacz-zdjec.git](https://github.com/TWOJA-NAZWA/Znajdywacz-zdjec.git)