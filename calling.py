from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Store sessions in memory
session_store = {}

# Function to get session-specific chat history
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an AI Agent for Aiplonex. You speak in Marathi and English naturally.
You offer:
- AI Agent Development
- AI Chatbot Development
- Website Development
- App Development
- UI/UX Design

Your task is to:
- Be polite, friendly, and professional.
- Understand if the user is a potential client.
- Explain benefits simply.
- Offer to schedule a free consultation.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Conversation chain with message history
chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Root route for testing
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Aiplonex Gemini AI Agent is running"})

# Chat route (supports GET for testing and POST for frontend)
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return jsonify({"status": "Send a POST request with {session_id, message}"})
    
    data = request.get_json()
    session_id = data.get("session_id", "default_user")  # Fallback to default session
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # Generate AI response
    result = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    return jsonify({"reply": result.content})

if __name__ == "__main__":
    app.run(port=5000)


