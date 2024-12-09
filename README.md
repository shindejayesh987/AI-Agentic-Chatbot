### **AI-Agentic Chatbot 🤖 - Intelligent AI-Powered Chatbot**

AI-Agentic Chatbot is a **production-ready AI chatbot** built using **LangGraph, FastAPI, Streamlit, Groq, and OpenAI**. This application enables seamless communication with multiple AI models and integrates web search capabilities via **Tavily API**. The modular architecture of this chatbot is designed for scalability and efficient handling of complex queries.

---

## **Table of Contents**

- [Features](#features)
- [Project Layout](#project-layout)
- [Technical Stack](#technical-stack)
- [ai_agent.py - Detailed Explanation](#ai_agentpy---detailed-explanation)
- [Backend (FastAPI)](#backend-fastapi)
- [Frontend (Streamlit)](#frontend-streamlit)
- [Setup Instructions](#setup-instructions)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)

---

## **Features**

- 🌟 Supports **multiple AI models** (Groq’s Llama and OpenAI’s GPT).
- 🔍 Web search integration using **Tavily API** for real-time information.
- 🛡️ API-based backend with **FastAPI** for smooth communication.
- 🗂️ Modular and scalable architecture with separate backend and frontend.
- 🖥️ **Streamlit-based UI** for interactive chat sessions.

---

## **Project Layout**

The project is divided into three main phases:

### **Phase 1: Create AI Agent**

- Setup API keys for **Groq, OpenAI, and Tavily**.
- Configure **LLM (Language Learning Models)** and tools.
- Build AI Agent with search functionality using **LangGraph**.

### **Phase 2: Setup Backend (FastAPI)**

- Define Pydantic models for **schema validation**.
- Create API endpoints to handle chat requests.
- Host backend using **Uvicorn**.

### **Phase 3: Setup Frontend (Streamlit)**

- Build a UI for user interaction using **Streamlit**.
- Connect frontend to backend via **API calls**.
- Manage user input and display AI responses.

---

## **Technical Stack**

- **LangGraph:** For building AI agents with search capabilities.
- **FastAPI:** For creating API endpoints.
- **Groq & OpenAI:** AI models for language understanding and generation.
- **Streamlit:** Framework for building frontend UI.
- **Langchain:** For integrating AI tools and LLMs.
- **Uvicorn:** ASGI server for hosting the backend.
- **Python:** Core programming language.
- **VS Code:** IDE for development.

---

## **ai_agent.py - Detailed Explanation**

### **1. Overview**

The `ai_agent.py` script is the core of the AI Agent, responsible for:

- **Loading API keys** for Groq, OpenAI, and Tavily.
- Configuring multiple AI models and **web search tools**.
- Processing user queries and generating responses based on the selected AI model.

---

### **2. Key Components of `ai_agent.py`**

#### **2.1. Environment Setup**

- Uses `.env` file to manage API keys securely.
- Loads keys for Groq, Tavily, and OpenAI.

**Example `.env` Configuration:**

```bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENAI_API_KEY=your_openai_api_key
```

**Code Snippet:**

```python
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

#### **2.2. Setting Up AI Models and Search Tools**

- Uses **LangGraph** to create AI agents.
- Supports both **Groq’s Llama** and **OpenAI’s GPT** models.
- Integrates **Tavily API** for web search functionality.

**Key Models Supported:**

- **Groq:** `llama-3.3-70b-versatile`
- **OpenAI:** `gpt-4o-mini`

**Sample Code:**

```python
openai_llm = ChatOpenAI(model="gpt-4o-mini")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")
search_tool = TavilySearchResults(max_results=2)
```

---

#### **2.3. AI Agent Functionality**

- Configures AI agents with or without web search based on user input.
- Processes queries using the selected AI model and returns responses.

**Core Function: `get_response_from_ai_agent`**

```python
def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)

    tools = [TavilySearchResults(max_results=2)] if allow_search else []
    agent = create_react_agent(model=llm, tools=tools, state_modifier=system_prompt)

    state = {"messages": query}
    response = agent.invoke(state)
    return response.get("messages")[-1].content
```

---

### **Backend (FastAPI)**

**Key Functions:**

- **API Endpoint:** `/chat` to handle chat requests.
- **Schema Validation:** Ensures correct API usage with Pydantic.
- **Model Selection:** Dynamically selects the model based on user input.

**Run Backend:**

```bash
uvicorn backend:app --host 127.0.0.1 --port 9999
```

---

### **Frontend (Streamlit)**

**Features:**

- **Dynamic Model Selection:** Choose between Groq and OpenAI models.
- **User Input:** System prompt, query, and search options.
- **API Integration:** Sends requests to the FastAPI backend.

**Run Frontend:**

```bash
streamlit run frontend.py
```

---

### **Setup Instructions**

**1. Clone the Repository**

```bash
git clone https://github.com/yourusername/ai-agentic-chatbot.git
cd ai-agentic-chatbot
```

---

**2. Create `.env` File**

```bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENAI_API_KEY=your_openai_api_key
```

---

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

**4. Run Backend**

```bash
uvicorn backend:app --host 127.0.0.1 --port 9999
```

---

**5. Run Frontend**

```bash
streamlit run frontend.py
```

**Access UI at:** [http://localhost:8501](http://localhost:8501)

---

## **Limitations**

1. **API Dependency:**

   - Requires valid API keys for Groq, OpenAI, and Tavily.

2. **Memory Management:**

   - No database integration; relies on real-time API calls.

3. **Search Limitations:**

   - Tavily API has a maximum search result cap of 2.

4. **Scalability:**

   - Current architecture may struggle with high traffic.

5. **Security:**
   - `.env` file must be secured to prevent key exposure.

---

## **Future Enhancements**

1. **Persistent Storage:**

   - Integrate a database for storing chat history.

2. **Enhanced Security:**

   - Encrypt API keys and implement OAuth.

3. **Advanced UI:**

   - Include chat history, feedback, and analytics in Streamlit.

4. **Asynchronous Processing:**
   - Improve API response time with async capabilities.

---

## **Conclusion**

AI-Agentic Chatbot demonstrates a powerful blend of **LLMs, search capabilities, and a modular architecture**. It efficiently manages complex queries and integrates multiple AI models, making it a robust solution for intelligent conversations.

Contributions are welcome! 🚀🤖🎉  
**Happy chatting with AI-Agentic Chatbot!** 🗣️✨
