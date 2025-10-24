# MedBot: AI-Powered Medical Chatbot with Doctor Recommendation ⚕️

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![AI-Platform](https://img.shields.io/badge/AI_Platform-Google_Gemini-orange.svg)](https://ai.google.dev/)
[![VectorDB](https://img.shields.io/badge/Vector_DB-Pinecone-4C5C6F.svg)](https://www.pinecone.io/)
[![Maps-API](https://img.shields.io/badge/Maps_API-Google_Places-red.svg)](https://developers.google.com/maps/documentation/places/web-service/overview)

## ✨ Project Overview

In an era where quick access to reliable health information and local healthcare services is paramount, the "MedBot" project stands as an innovative solution. This AI-powered Medical Chatbot provides instant, medically accurate answers to user queries and offers real-time, location-based recommendations for nearby doctors and clinics. Developed as a Flask-based web application, MedBot leverages cutting-edge AI and API integrations to create a comprehensive and user-friendly healthcare assistant.

## 💡 Problem Solved

Users often face challenges in:
1.  **Accessing Trustworthy Medical Information:** Sifting through unreliable online sources for health-related questions.
2.  **Navigating Healthcare Options:** Efficiently finding suitable and nearby medical professionals when needed.

MedBot addresses these by offering a centralized, intelligent platform for both knowledge retrieval and service discovery.

## 🚀 Key Features

* **Intelligent Medical Q&A (RAG System):**
    * Utilizes a Retrieval-Augmented Generation (RAG) framework powered by **Google Gemini API** for natural language understanding and generation.
    * Employs a **Pinecone vector database** for efficient semantic search across a vast medical knowledge base (ingested from a PDF document).
    * Provides context-aware and medically accurate responses to complex health queries.
* **Real-time Doctor Recommendation:**
    * Integrates with the **Google Maps Places API** to fetch real-time data on nearby doctors, clinics, and hospitals.
    * Leverages user's geolocation to calculate and display distances, offering highly relevant local recommendations.
    * Displays essential doctor details including name, address, and a direct link to view on Google Maps.
* **Intuitive Web Interface:**
    * A clean, responsive Flask-based web interface for seamless user interaction.
    * Clear chat history display for an engaging conversational experience.
* **Scalable Architecture:**
    * Designed with modular components for easy maintenance and future expansions.

## 🛠️ Technology Stack

* **Backend Framework:** Python Flask
* **Language Models:** Google Gemini API
* **Vector Database:** Pinecone
* **RAG Orchestration:** LangChain
* **Embeddings:** Sentence-Transformers (e.g., `all-MiniLM-L6-v2`)
* **Geolocation & Maps:** Google Maps Places API, Geopy
* **PDF Processing:** PyPDF
* **Environment Management:** `python-dotenv`, `venv`
* **Frontend:** HTML, CSS, JavaScript
* **Dependencies:** `langchain-core`, `langchain-pinecone`, `langchain-google-genai`, `langchain-huggingface`, `requests`, `pypdf`, `sentence-transformers`, `geopy`, `flask`, `python-dotenv`.

## ⚙️ Architecture & Data Flow

### **1. Data Ingestion & Knowledge Base Creation:**
* `Medical_book.pdf` is processed by `pypdf` and `langchain-text-splitters` into chunks.
* `Sentence-Transformers` generates vector embeddings for each chunk.
* These embeddings and text metadata are stored in the **Pinecone vector database** (index: `medicalbot`).

### **2. Medical Q&A Flow (RAG System):**
* User enters a query via the Flask web app.
* **LangChain** orchestrates:
    * User query is embedded.
    * Pinecone is queried for semantically similar text chunks.
    * Retrieved context and query are passed to **Google Gemini API**.
    * Gemini generates a coherent, context-aware answer.
* Response is displayed to the user.

### **3. Doctor Recommendation Flow:**
* User requests doctor recommendations.
* Frontend prompts for geolocation (HTML5 Geolocation API).
* Backend (Flask) uses user's coordinates to query **Google Maps Places API**.
* `Geopy` calculates distances to returned doctors/clinics.
* Formatted doctor details (name, address, distance, map link) are sent to the frontend.
* Response is displayed to the user.

## 🚀 Getting Started

### **Prerequisites**

* Python 3.9+ installed
* A Google Cloud Project with **Google Gemini API** (Generative Language API) and **Google Maps Places API** enabled.
* A Pinecone account (free tier is sufficient).
* API Keys for Google Gemini, Google Maps, and Pinecone.

### **Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/MedBot.git](https://github.com/yourusername/MedBot.git)
    cd MedBot
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # On Windows
    source venv/bin/activate # On macOS/Linux
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` contains all the dependencies listed in "Technology Stack")*

4.  **Create a `.env` file** in the project root and add your API keys:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    Maps_API_KEY="YOUR_GOOGLE_MAPS_PLACES_API_KEY"
    ```

### **Usage**

1.  **Ingest Medical Data into Pinecone:**
    ```bash
    python ingest.py
    ```
    *(Wait for "Successfully ingested..." message. This populates your Pinecone index.)*

2.  **Run the Flask Application:**
    ```bash
    python app.py
    ```

3.  **Access the Chatbot:** Open your web browser and navigate to `http://127.0.0.1:5000`.

## 📈 Impact & Future Enhancements

MedBot significantly improves the accessibility of health information and doctor discovery. Future plans include:

* Direct appointment scheduling integration.
* Expansion of the medical knowledge base with more diverse sources.
* Personalized health record integration (with robust privacy safeguards).
* Voice interaction capabilities.
* Multi-language support.

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

[ Swapnil Panda/LinkedIn Profile Link: www.linkedin.com/in/swapnil-panda-1207992ba ]


