import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLP Setup ---
@st.cache_resource 
def load_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)

load_nltk()

knowledge_base = {
    "responses": {
        "hello": "Hi! I'm the Hex Softwares bot. Ask me about our hours, shipping, or contact info!",
        "shipping": "We ship worldwide! Standard delivery takes 3-5 business days.",
        "refunds": "We offer full refunds within 30 days of purchase.",
        "contact": "You can reach Hex Softwares through the following channels: \n\n"
           "📧 **Email:** support@hexsoftwares.com \n"
           "📞 **Phone:** +1 (800) HEX-SOFT \n"
           "📍 **Office:** 123 Tech Plaza, Silicon Valley, CA \n"
           "🌐 **Support Portal:** [Click Here](https://hexsoftwares.com/support)",
        "hours": "Our office hours are 9:00 AM to 6:00 PM, Monday through Friday.",
        "thank you": "Welcome! Anything else?",
        "bye": "Have a nice day!"
    }
}

lemmer = nltk.stem.WordNetLemmatizer()

def lem_normalize(text):
    return [lemmer.lemmatize(token) for token in nltk.word_tokenize(text.lower().translate(dict((ord(p), None) for p in string.punctuation)))]

def get_ai_response(user_input):
    user_input = user_input.lower().strip()

    # 1. Instant Greeting Check (The "Fast Path")
    greetings = ["hello", "hi", "greetings", "hey", "sup"]
    if user_input in greetings:
        return knowledge_base["responses"]["hello"]
    
    ship = ["shipping", "ship", "shipped"]
    if user_input in ship:
        return knowledge_base["responses"]["shipping"]
    
    contact_keywords = ["contact", "contacts", "email", "phone", "address", "support"]
    if any(word in user_input for word in contact_keywords):
        return knowledge_base["responses"]["contact"]
    
    thanku = ["thank you", "thanks"]
    if user_input in thanku:
        return knowledge_base["responses"]["thank you"]
    
    bye = ["no", "not now", "bye", "goodbye"]
    if user_input in bye:
        return knowledge_base["responses"]["bye"]

    # 2. The "Empty/Punctuation" Guard
    if not user_input or all(char in string.punctuation for char in user_input):
        return "I didn't catch that. Could you say it again?"

    # 3. Mathematical Similarity (The "Deep Path")
    sentences = list(knowledge_base["responses"].values())
    sentences.append(user_input)
    
    # FIX: We REMOVED stop_words='english' so 'hello' isn't ignored
    tfidf_vec = TfidfVectorizer(tokenizer=lem_normalize, token_pattern=None) 
    tfidf = tfidf_vec.fit_transform(sentences)
    
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    
    # If the match is too weak, say not sure
    if vals.flatten()[idx] < 0.1:
        return "I'm not sure about that. Try asking about our shipping or hours!"
    
    return sentences[idx]

# --- Streamlit UI FIXES ---
st.set_page_config(page_title="HexBot Local", page_icon="🤖", layout="wide")
st.title("🤖 HexBot: Local Edition")

# 1. Create a container for messages with a scrollbar
# This prevents the messages from pushing the search box away
chat_placeholder = st.container(height=500) 

if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Render messages INSIDE the scrollable container
with chat_placeholder:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 3. Chat input logic (outside the container to stay at bottom)
if prompt := st.chat_input("Type here..."):
    # Immediately display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_placeholder:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Get and display response
    response = get_ai_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with chat_placeholder:
        with st.chat_message("assistant"):
            st.markdown(response)