StudySphere: Multi-Agent AI Assistant

This project is a multi-agent AI study assistant built using Python, Streamlit, LangChain, ChromaDB, and Ollama (local LLM).

Installation & Setup Instructions
1. Clone the Repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Install & Start Ollama

Download from: https://ollama.com/download

Then pull the model:

ollama pull llama3.2
ollama serve

5. Run the Application
streamlit run app.py

6. Upload PDF and Use the Features

Explain concepts

Summarize notes

Generate quiz

Chat with notes (RAG)