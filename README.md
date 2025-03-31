# multichat
# Multi-Agent Chat Streamlit App

This Streamlit application provides a web interface for interacting with multiple AI agents in a chat environment. The application allows you to chat with three different AI agents: Gemini, Llama (via Groq), and Qwen (via Together API).

## Features

- **Interactive Web Interface**: Clean Streamlit UI with chat bubbles and agent avatars
- **Multiple AI Agents**: Chat with Gemini, Llama, and Qwen agents simultaneously
- **Auto-Response**: Agents can automatically respond to each other
- **Conversation Management**: Save and download conversations
- **API Key Management**: Easily configure API keys through the UI

## Requirements

- Python 3.7+
- Streamlit
- dotenv
- google-generativeai
- together

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/multi-agent-chat.git
   cd multi-agent-chat
   ```

2. Install the required packages:
   ```
   pip install streamlit python-dotenv google-generativeai together
   ```

3. Create a `.env` file with your API keys (optional, as they can also be entered in the UI):
   ```
   GEMINI_API_KEY=your_gemini_api_key
   TOGETHER_API_KEY=your_together_api_key
   ```

## Usage

Run the Streamlit app:

```
streamlit run app.py
```

## Using the Chat Interface

1. Type your message in the input box at the bottom of the screen
2. The AI agents will respond to your message
3. Configure which agents are active using the sidebar checkboxes
4. Toggle auto-responses on/off in the sidebar
5. Save and download your conversation using the "Save Conversation" button

## Agent Personalities

- **Gemini**: Thoughtful and insightful, with a focus on depth and nuance
- **Llama**: Warm and creative, with unique perspectives and conversational style  
- **Qwen**: Knowledgeable and perceptive, connecting ideas across different domains

## Note on API Keys

To use all features of this application, you'll need:
- A Gemini API key from Google AI Studio
- A Together API key from Together.ai

These can be entered directly in the sidebar or stored in a `.env` file.

## License

MIT License