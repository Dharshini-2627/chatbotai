import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "sustainability"

# ---------------------------------------------------
# Load models and database
# ---------------------------------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("./all-MiniLM-L6-v2")
    client = PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    return embed_model, collection

@st.cache_data
def load_csv_data():
    """Load the CSV for direct matching"""
    try:
        df = pd.read_csv("activities.csv")
        return df
    except:
        return None

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def extract_number(text):
    """Extract the first number from text (for km, hours, etc.)"""
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    return float(match.group(1)) if match else None

def calculate_emissions(activity_type, user_value, base_emission, base_value=20):
    """Scale emissions based on user input vs base value"""
    if user_value is None:
        return base_emission
    return round((user_value / base_value) * base_emission, 2)

def generate_response(user_input, retrieved_docs, retrieved_metas, df):
    """Generate a smart response based on retrieved data and user input"""
    
    if not retrieved_docs or not retrieved_metas:
        return "I couldn't find relevant data. Could you rephrase? For example: 'I drive 30 km daily' or 'I use AC for 6 hours'."
    
    # Extract number from user input
    user_number = extract_number(user_input)
    user_lower = user_input.lower()
    
    # Determine activity type
    activity_type = None
    base_value = 20  # default
    
    if any(word in user_lower for word in ['drive', 'car', 'petrol', 'vehicle', 'km']):
        activity_type = 'car'
        base_value = 20
    elif any(word in user_lower for word in ['bus', 'public transport']):
        activity_type = 'bus'
        base_value = 20
    elif any(word in user_lower for word in ['bike', 'bicycle', 'cycle']):
        activity_type = 'bike'
        base_value = 20
    elif any(word in user_lower for word in ['ac', 'air condition', 'cooling']):
        activity_type = 'ac'
        base_value = 8
    elif any(word in user_lower for word in ['meat', 'beef', 'non-veg', 'chicken']):
        activity_type = 'meat'
    elif any(word in user_lower for word in ['vegetarian', 'veg', 'plant']):
        activity_type = 'veg'
    elif any(word in user_lower for word in ['led', 'bulb', 'light']):
        activity_type = 'led'
        base_value = 5
    elif any(word in user_lower for word in ['online', 'shopping', 'delivery']):
        activity_type = 'online'
    
    # Get the best matching metadata
    top_meta = retrieved_metas[0]
    top_doc = retrieved_docs[0]
    
    # Try different possible key names for emission
    emission_value = top_meta.get("Avg_CO2_Emission(kg/day)") or top_meta.get("Avg_CO2_Emission") or "0"
    try:
        emission = float(emission_value)
    except (ValueError, TypeError):
        emission = 0.0
    
    category = top_meta.get("Category", "")
    tip = top_meta.get("Tip", "")
    
    # Calculate scaled emissions if user provided a number
    if user_number and activity_type in ['car', 'bus', 'bike', 'ac', 'led']:
        scaled_emission = calculate_emissions(activity_type, user_number, emission, base_value)
    else:
        scaled_emission = emission
    
    # Build response
    response = f"Your Estimated Daily CO₂ Emissions\n\n"
    response += f"**{scaled_emission} kg/day** from your described activity.\n\n"
    
    response += f"### 💡 Reduction Tips\n\n"
    
    # Provide specific tips based on activity
    if activity_type == 'car':
        response += f"1. **Switch to public transport or carpooling** → Save ~3-4 kg CO₂/day\n"
        response += f"2. **Try cycling for shorter distances** → Save {scaled_emission} kg CO₂/day\n"
        response += f"3. **Maintain proper tire pressure** → Reduce emissions by 5-10%\n"
    elif activity_type == 'bus':
        response += f"1. **Great choice!** Buses are much better than cars \n"
        response += f"2. **Consider cycling for trips under 5 km** → Save {scaled_emission} kg CO₂/day\n"
        response += f"3. **Combine errands into one trip** → Maximize efficiency\n"
    elif activity_type == 'bike':
        response += f"1. **Excellent!** Cycling produces zero emissions! \n"
        response += f"2. **Keep it up** and inspire others to cycle too!\n"
        response += f"3. **Maintain your bike regularly** for smooth rides\n"
    elif activity_type == 'ac':
        response += f"1. **Set temperature to 24-26°C** → Save ~1-2 kg CO₂/day\n"
        response += f"2. **Use ceiling fans alongside AC** → Reduce AC usage by 30%\n"
        response += f"3. **Service AC regularly** → Improve efficiency by 15-20%\n"
    elif activity_type == 'meat':
        response += f"1. **Try 'Meatless Mondays'** → Save ~1 kg CO₂/day\n"
        response += f"2. **Replace red meat with chicken/fish** → Save ~3-4 kg CO₂/day\n"
        response += f"3. **Increase plant-based meals** → Save up to {scaled_emission} kg CO₂/day\n"
    elif activity_type == 'veg':
        response += f"1. **Amazing choice!** Plant-based diets are much better \n"
        response += f"2. **Buy local produce** → Further reduce transport emissions\n"
        response += f"3. **Reduce food waste** → Save even more resources\n"
    else:
        # Generic tips from retrieved data
        response += f"1. **{tip}**\n"
        response += f"2. Consider alternatives in the '{category}' category\n"
        response += f"3. Track your progress over time for motivation\n"
    
    # Add alternatives from other retrieved results
    if len(retrieved_metas) > 1:
        response += f"\n### Alternative Options\n\n"
        for i in range(1, min(4, len(retrieved_metas))):
            alt_doc = retrieved_docs[i]
            alt_emission = retrieved_metas[i].get("Avg_CO2_Emission(kg/day)", "?")
            response += f"- **{alt_doc}**: {alt_emission} kg/day\n"
    
    response += f"\n###  Long-term Impact\n\n"
    yearly_saving = round(scaled_emission * 365, 1)
    response += f"If you reduce these emissions completely, you could save **{yearly_saving} kg CO₂ per year** "
    response += f"— that's equivalent to planting **{int(yearly_saving/20)} trees**! "
    
    return response

# ---------------------------------------------------
# Streamlit App UI
# ---------------------------------------------------
st.set_page_config(page_title="CO₂ Reduction Chatbot", page_icon="", layout="wide")
st.title("CO₂ Reduction Chatbot")
st.caption("Ask about your daily habits — get personalized tips to reduce emissions!")

embed_model, collection = load_models()
df = load_csv_data()

if "history" not in st.session_state:
    st.session_state["history"] = []

# Sidebar with examples
with st.sidebar:
    st.header("Example Questions")
    st.markdown("""
    - I drive 25 km daily in a petrol car
    - I use AC for 10 hours per day
    - I eat meat every day
    - I cycle to work (15 km)
    - I use LED bulbs for 8 hours
    - I order online 3 times a week
    """)
    
    if st.button("Clear Chat History"):
        st.session_state["history"] = []
        st.rerun()

# ---------------------------------------------------
# Chat input
# ---------------------------------------------------
user_input = st.chat_input("Ask something like 'I drive 25 km daily using a petrol car'...")

if user_input:
    # Retrieve similar data
    q_emb = embed_model.encode([user_input])[0].tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=6,
        include=["documents", "metadatas"]
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    # Generate smart response
    output = generate_response(user_input, docs, metas, df)

    # Save to chat history
    st.session_state["history"].append({"user": user_input, "bot": output})



# ---------------------------------------------------
# Display chat history
# ---------------------------------------------------
for chat in st.session_state["history"]:
    with st.chat_message("user"):
        st.markdown(chat['user'])
    with st.chat_message("assistant"):
        st.markdown(chat['bot'])
