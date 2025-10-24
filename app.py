import os
from flask import Flask, render_template, request, jsonify
import requests
from dotenv import load_dotenv
from geopy.distance import geodesic

# --- CORRECTED LangChain & Pinecone Imports ---
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if keys are loaded
if not all([GOOGLE_MAPS_API_KEY, PINECONE_API_KEY, GOOGLE_API_KEY]):
    raise ValueError("One or more API keys are missing from your .env file.")

# --- 2. LOAD MODELS ONCE (Fixes "Long Loading Time") ---
print("Loading embedding model...")
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
INDEX_NAME = "medicalbot"

print("Connecting to Pinecone index...")
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

print("Loading LLM (Gemini)...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)


# --- 4. SET UP RAG CHAIN ONCE ---
PROMPT_TEMPLATE = """
Use the following pieces of context from the medical book to answer the user's question.
If you don't know the answer from the context, just say that you don't have that information in your knowledge base. Don't try to make up an answer.
Context: {context}
Question: {question}
Helpful Answer:
"""
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
print("RAG chain is ready.")


# --- 5. FLASK APP & GOOGLE MAPS FUNCTIONS ---
app = Flask(__name__)

DISEASE_SPECIALTY_MAP = {
    "fever": "doctor",
    "diabetes": "doctor",
    "skin": "dermatologist",
    "acne": "dermatologist",
    "rash": "dermatologist",
    "bone": "orthopedic",
    "fracture": "orthopedic",
    "heart": "cardiologist",
    "chest pain": "cardiologist",
    "eye": "ophthalmologist",
    "mental": "psychiatrist",
    "anxiety": "psychiatrist",
    "depression": "psychiatrist"
}

def get_google_places_nearby(lat, lon, specialty_type, radius_km=50):
    """Fetches nearby places from Google Maps API."""
    place_type = "doctor" # Default to doctor
    keyword = specialty_type
    
    # Use "hospital" or "clinic" if requested
    if "hospital" in specialty_type.lower():
        place_type = "hospital"
        keyword = "hospital"
    elif "clinic" in specialty_type.lower():
        place_type = "clinic"
        keyword = "clinic"

    radius_m = int(radius_km * 1000)
    
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        f"location={lat},{lon}&radius={radius_m}&type={place_type}&keyword={keyword}&key={GOOGLE_MAPS_API_KEY}"
    )
    
    print(f"Requesting Google Places: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'OK':
            print(f"Error from Google Places API: {data.get('status')}")
            print(f"Error message: {data.get('error_message')}")
            return []
            
        return data.get('results', [])
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request to Google Places failed: {e}")
        return []

def get_specialty_from_query(user_query):
    """Extracts a specialty from the user's query."""
    user_query_lower = user_query.lower()
    for disease, specialty in DISEASE_SPECIALTY_MAP.items():
        if disease in user_query_lower:
            return specialty
    if "hospital" in user_query_lower:
        return "hospital"
    if "clinic" in user_query_lower:
        return "clinic"
    return "doctor"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    """Main chat endpoint. Decides to use RAG or Doctor Finder."""
    user_input = request.json.get("msg")
    user_location = request.json.get("location")
    
    if not user_input:
        return jsonify({"response": "Please enter a question."})

    doc_keywords = ["doctor", "recommend", "hospital", "clinic", "nearby", "specialist", "find me"]
    is_doctor_request = any(keyword in user_input.lower() for keyword in doc_keywords)

    if is_doctor_request:
        # --- PATH 1: DOCTOR FINDER ---
        if not user_location:
            return jsonify({"response": "I see you're looking for a doctor. **Please allow location access on your browser and ask again** so I can find one nearby."})
        
        try:
            lat, lon = map(float, user_location.split(","))
        except Exception:
            return jsonify({"response": "Your location format is invalid. Please refresh and allow location access."})
        
        specialty_type = get_specialty_from_query(user_input)
        print(f"Searching for specialty: {specialty_type} near {lat},{lon}")
        
        places = get_google_places_nearby(lat, lon, specialty_type)
        
        if places:
            response_text = f"Here are some {specialty_type}s and hospitals I found near you (within 50km):<br><br>"
            for place in places[:5]:  # Show top 5
                name = place.get("name", "N/A")
                address = place.get("vicinity", "N/A")
                p_lat = place["geometry"]["location"]["lat"]
                p_lon = place["geometry"]["location"]["lng"]
                
                distance_km = round(geodesic((lat, lon), (p_lat, p_lon)).km, 2)
                map_url = f"https://www.google.com/maps?q={p_lat},{p_lon}"
                
                response_text += (
                    f"<div class='doctor-card'>"
                    f"  <strong>Name:</strong> {name}<br>"
                    f"  <strong>Address:</strong> {address}<br>"
                    f"  <strong>Distance:</strong> {distance_km} km<br>" # Corrected: Added f"" and <br>
                    f"  <a href='{map_url}' target='_blank' class='map-link'>View on Map</a>" # Corrected: Added f""
                    f"</div><br>"
                )
            return jsonify({"response": response_text})
        else:
            return jsonify({"response": f"Sorry, I couldn't find any {specialty_type}s or hospitals near you. You might be outside the 50km search radius."})
    
    else:
        # --- PATH 2: RAG MEDICAL INFO ---
        try:
            print(f"Invoking RAG chain for: {user_input}")
            result = qa_chain.invoke(user_input)
            return jsonify({"response": result['result']})
        except Exception as e:
            print(f"Error invoking RAG chain: {e}")
        # --- MODIFIED LINE ---
            return jsonify({"response": f"Sorry, an error occurred: {str(e)}"}) # Show the actual error
if __name__ == "__main__":
    print("Starting Flask app on http://127.0.0.1:8080 ...")
    app.run(host="127.0.0.1", port=8080, debug=False)