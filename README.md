📸 Znajdywacz Zdjęć 2026 (AI Image Search Lab)
Projekt do przedostatniego modułu kursu Data Science
System inteligentnego zarządzania archiwum zdjęć, który wykorzystuje multimodalne modele Large Language Models (LLM) do automatycznego opisywania obrazów oraz bazę wektorową do błyskawicznego wyszukiwania semantycznego.

🚀 Główne funkcjonalności
Multimodalna Analiza: Wykorzystanie modeli gpt-4o oraz gpt-5 do generowania szczegółowych opisów (detekcja obiektów, tekstów i kontekstu).
Wyszukiwanie Semantyczne: Zastosowanie bazy wektorowej Qdrant oraz modelu text-embedding-3-small do wyszukiwania po sensie zapytania (a nie tylko słowach kluczowych).
Monitorowanie Tokenów: Diagnostyka procesu wnioskowania dzięki podglądowi reasoning_tokens (specyficznych dla modeli GPT-5).
Interfejs Streamlit: Intuicyjny panel podzielony na procesowanie danych, wyszukiwarkę oraz zarządzanie bazą.
🛠️ Stos technologiczny
Język: Python 3.10+
AI/ML: OpenAI API (Vision, Embeddings)
Baza danych: Qdrant (Vector Database)
Frontend: Streamlit
Zarządzanie projektem: Trello (Metodyka Agile/Scrum)
📋 Struktura projektu
app.py - główna aplikacja Streamlit
uploaded_images/ - lokalne archiwum przetworzonych zdjęć
qdrant_data/ - lokalna persystencja bazy wektorowej
.env - klucze API (plik pominięty w repozytorium)
⚙️ Instalacja i uruchomienie
Sklonuj repozytorium:
git clone [https://github.com/PawelBartosik-S/ai-vision-search-lab.git](https://github.com/PawelBartosik-S/ai-vision-search-lab.git)
Zainstaluj biblioteki:
pip install -r requirements.txt 3. Skonfiguruj plik .env dodając swój OPENAI_API_KEY.

Uruchom aplikację:
streamlit run app.py

👨‍🔬 Cele edukacyjne (Product Owner Perspective) Celem projektu było nie tylko dostarczenie kodu, ale przejście przez pełny cykl życia produktu:

Zarządzanie Backlogiem w Trello.

Implementacja z zachowaniem GitFlow (praca na branchach, Pull Requests).

Benchmarking modeli pod kątem jakości opisów i zużycia tokenów.