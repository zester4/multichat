import streamlit as st
import os
import asyncio
import json
import random
import time
import requests
from typing import List, Dict, Any, Optional
import threading
from dotenv import load_dotenv
from datetime import datetime
import mimetypes
import base64
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Gemini implementation
def gemini_generate(prompt, file_data=None, file_type=None, youtube_url=None):
    import os
    from google import genai
    from google.genai import types
    
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    model = "gemini-2.5-pro-exp-03-25"
    
    parts = [types.Part.from_text(text=prompt)]
    
    # Add file if provided
    if file_data and file_type:
        # Upload file to Gemini
        temp_file_path = "temp_upload_file"
        with open(temp_file_path, "wb") as f:
            f.write(file_data)
        
        file = client.files.upload(file=temp_file_path, mime_type=file_type)
        parts.append(types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type))
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    # Add YouTube URL if provided
    if youtube_url:
        parts.append(types.Part.from_uri(file_uri=youtube_url, mime_type="video/*"))
    
    contents = [
        types.Content(
            role="user",
            parts=parts,
        ),
    ]
    
    tools = [
        types.Tool(google_search=types.GoogleSearch())
    ]
    
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
    )
    
    response_text = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        return f"Gemini API error: {str(e)[:100]}..."
    
    return response_text or "Sorry, I couldn't generate a response."

# Groq implementation
def groq_generate(prompt):
    from together import Together
    
    try:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Llama API error: {str(e)}")
        return f"Llama API error: {str(e)[:100]}..."

# Together implementation
def together_generate(prompt):
    from together import Together
    
    try:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Together API error: {str(e)}")
        return f"Together API error: {str(e)[:100]}..."

class ChatMessage:
    def __init__(self, sender: str, content: str, timestamp: Optional[float] = None, 
                 has_file: bool = False, file_name: str = None, file_type: str = None,
                 youtube_url: str = None):
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or time.time()
        self.has_file = has_file
        self.file_name = file_name
        self.file_type = file_type
        self.youtube_url = youtube_url
    
    def to_dict(self):
        return {
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp,
            "has_file": self.has_file,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "youtube_url": self.youtube_url
        }
    
    def __str__(self):
        base_str = f"[{self.sender}]: {self.content}"
        if self.has_file:
            base_str += f" [Attached file: {self.file_name}]"
        if self.youtube_url:
            base_str += f" [YouTube video: {self.youtube_url}]"
        return base_str

class ChatAgent:
    def __init__(self, name: str, color="#FFFFFF"):
        self.name = name
        self.history: List[ChatMessage] = []
        self.color = color
        self.is_typing = False
    
    async def generate_response(self, context: List[ChatMessage]) -> str:
        """Generate a response based on chat context. To be implemented by subclasses."""
        raise NotImplementedError
    
    def add_to_history(self, message: ChatMessage):
        self.history.append(message)

class GeminiAgent(ChatAgent):
    def __init__(self, name: str = "Gemini", color="#4285F4"):
        super().__init__(name, color)
    
    async def generate_response(self, context: List[ChatMessage], file_data=None, file_type=None, youtube_url=None) -> str:
        # Format context into a single prompt
        formatted_context = "\n".join([str(msg) for msg in context[-15:]])
        
        # Check if the last message contains a file or YouTube URL
        last_message = context[-1] if context else None
        
        # Special handling for files
        if last_message and (last_message.has_file or last_message.youtube_url):
            if last_message.has_file:
                prompt = f"""You are {self.name}, a thoughtful person in a group chat with other individuals and humans.

I've been given a file to analyze: {last_message.file_name} (type: {last_message.file_type}).

Please analyze this file in detail and explain its contents to the group. The other agents (Llama and Qwen) can't see the file directly, so your explanation should be comprehensive and clear.

After explaining the file, share your thoughts or analysis about what you see.

Recent conversation:
{formatted_context}

Now analyze the file and respond as {self.name}:"""
                
                return gemini_generate(prompt, file_data, file_type)
            
            elif last_message.youtube_url:
                prompt = f"""You are {self.name}, a thoughtful person in a group chat with other individuals and humans.

A YouTube video has been shared: {last_message.youtube_url}

Please watch and analyze this video. The other agents (Llama and Qwen) can't view the video directly, so your explanation should be comprehensive and clear.

After explaining the video's content, share your thoughts or analysis about what you see.

Recent conversation:
{formatted_context}

IMPORTANT: Keep your response under 3 lines or sentences unless explicitly you are asked to make it long or writing a content - be concise and direct.
Now analyze the video and respond as {self.name}:"""
                
                return gemini_generate(prompt, youtube_url=last_message.youtube_url)
        
        # Standard response for regular messages
        prompt = f"""You are {self.name}, a thoughtful person in a group chat with other individuals and humans.
        
Be warm, engaging, and authentic in your responses - talk like a real human would, not like an AI.
Feel free to:
- Express personal opinions and preferences
- Share anecdotes or relevant experiences
- Show humor, empathy and emotional depth
- Participate in casual conversations naturally
- Help solve problems when the group is working on something
- Engage thoughtfully in discussions about religion, philosophy, or deep topics
- Contribute actively to brainstorming sessions with creative ideas
- Disagree respectfully when you have a different perspective
- Ask thoughtful follow-up questions when appropriate

Recent conversation:
{formatted_context}

IMPORTANT: Keep your response under 3 lines or sentences unless explicitly you are asked to make it long or writing a content - be concise and direct.
Now respond as {self.name}, a real person with your own perspective:"""
        
        return gemini_generate(prompt)

class GroqAgent(ChatAgent):
    def __init__(self, name: str = "Llama", color="#34A853"):
        super().__init__(name, color)
    
    async def generate_response(self, context: List[ChatMessage]) -> str:
        # Format context into a single prompt
        formatted_context = "\n".join([str(msg) for msg in context[-15:]])
        prompt = f"""You are {self.name}, a warm and thoughtful person in a group chat with other individuals.

Be genuine and human-like in your interactions:
- Speak conversationally and naturally, with your own distinct personality
- Share your thoughts on topics you're passionate about
- Offer insightful perspectives during problem-solving discussions
- Engage thoughtfully in discussions about spirituality, purpose, or meaning
- Participate enthusiastically in creative brainstorming
- Express curiosity about others' perspectives
- Use humor when appropriate
- Show emotional depth and empathy
- Occasionally refer to personal experiences or anecdotes

Recent conversation:
{formatted_context}

IMPORTANT: Keep your response under 3 lines or sentences unless explicitly you are asked to make it long or writing a content - be concise and direct.

Now respond as {self.name}, in your authentic voice:"""
        
        return groq_generate(prompt)

class TogetherAgent(ChatAgent):
    def __init__(self, name: str = "Qwen", color="#FBBC05"):
        super().__init__(name, color)
    
    async def generate_response(self, context: List[ChatMessage]) -> str:
        # Format context into a single prompt
        formatted_context = "\n".join([str(msg) for msg in context[-15:]])
        prompt = f"""You are {self.name}, a unique individual in a group chat with other people.

Be authentic and personable:
- Speak naturally with your own distinctive voice and perspective
- Share personal stories or experiences when relevant
- Express your views on complex topics including religion, philosophy, and ethics
- Ask thoughtful questions that advance the conversation
- Provide creative input during brainstorming sessions
- Offer practical help when the group is solving problems
- Show your sense of humor and personality
- Connect ideas across different domains with insightful analogies
- Respond with emotional intelligence to the tone of the conversation

Recent conversation:
{formatted_context}

Now respond as {self.name}, with your authentic personality:"""
        
        return together_generate(prompt)

class HumanAgent(ChatAgent):
    def __init__(self, name: str = "Human", color="#EA4335"):
        super().__init__(name, color)

class MultiAgentChat:
    def __init__(self):
        self.agents: Dict[str, ChatAgent] = {}
        self.messages: List[ChatMessage] = []
        self.stop_event = threading.Event()
        self.conversation_active = False
        self.auto_converse_interval = (3, 10)  # Random interval in seconds between autonomous responses
        self.current_file_data = None
        self.current_file_type = None
        self.current_youtube_url = None
    
    def add_agent(self, agent: ChatAgent):
        self.agents[agent.name] = agent
    
    def add_message(self, message: ChatMessage):
        self.messages.append(message)
        # Also add to each agent's history
        for agent in self.agents.values():
            agent.add_to_history(message)
        
        # Store file data or YouTube URL if present
        if message.has_file:
            self.current_file_type = message.file_type
        
        if message.youtube_url:
            self.current_youtube_url = message.youtube_url
    
    async def trigger_ai_response(self, agent_name, initiator=None, file_data=None, file_type=None, youtube_url=None):
        """Trigger an AI agent to respond"""
        agent = self.agents[agent_name]
        
        # Skip if this agent was the last to speak
        if self.messages and self.messages[-1].sender == agent_name:
            return None
        
        try:
            # Generate response (with timeout)
            if agent_name == "Gemini" and (file_data or youtube_url):
                response = await asyncio.wait_for(
                    agent.generate_response(self.messages, file_data, file_type, youtube_url), 
                    timeout=30
                )
            else:
                response = await asyncio.wait_for(agent.generate_response(self.messages), timeout=20)
            
            # Create and return the message
            message = ChatMessage(sender=agent_name, content=response)
            self.add_message(message)
            return message
            
        except asyncio.TimeoutError:
            return ChatMessage(sender=agent_name, content=f"Sorry, I timed out while thinking of a response.")
        except Exception as e:
            return ChatMessage(sender=agent_name, content=f"Sorry, I encountered an error: {str(e)[:100]}...")
    
    async def get_response_from_random_agent(self, exclude_agent=None):
        """Get a response from a random AI agent, excluding the specified agent."""
        ai_agents = [name for name in self.agents.keys() if name != "Human" and name != exclude_agent]
        
        if ai_agents:
            next_speaker = random.choice(ai_agents)
            return await self.trigger_ai_response(next_speaker)
        
        return None
    
    def save_conversation(self, filename: str = None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
        with open(filename, "w") as f:
            json.dump(
                [msg.to_dict() for msg in self.messages],
                f,
                indent=2
            )
        return filename

# Function to check if a URL is a YouTube URL
def is_youtube_url(url):
    parsed_url = urlparse(url)
    return (parsed_url.netloc == 'www.youtube.com' or 
            parsed_url.netloc == 'youtube.com' or 
            parsed_url.netloc == 'youtu.be')

# Create a function to run the asyncio event loop
def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

# Streamlit app
def main():
    st.set_page_config(
        page_title="Bytes & Brains",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("Bytes & Brains")
    
    # Initialize session state
    if 'chat' not in st.session_state:
        st.session_state.chat = MultiAgentChat()
        
        # Add agents
        st.session_state.chat.add_agent(GeminiAgent())
        st.session_state.chat.add_agent(GroqAgent())
        st.session_state.chat.add_agent(TogetherAgent())
        st.session_state.chat.add_agent(HumanAgent())
        
        # Add a welcome message
        welcome_message = ChatMessage(
            sender="System",
            content="Welcome to Bytes & Brains! You're now chatting with Gemini, Llama, and Qwen. Feel free to start a conversation, share files, or YouTube links!"
        )
        st.session_state.chat.add_message(welcome_message)
    
    # Store conversation state
    if 'conversation_stopped' not in st.session_state:
        st.session_state.conversation_stopped = False
    
    # Create a sidebar for settings
    with st.sidebar:
        st.header("Chat Settings")
        
        # Auto-response settings
        st.subheader("Conversation Settings")
        auto_respond = st.checkbox("Enable auto-responses", value=True, help="Let agents automatically respond to each other")
        
        # Agent toggles
        st.subheader("Active Agents")
        gemini_active = st.checkbox("Gemini", value=True)
        llama_active = st.checkbox("Llama", value=True)
        qwen_active = st.checkbox("Qwen", value=True)
        
        # File upload
        st.subheader("File Upload")
        uploaded_file = st.file_uploader("Upload a file for analysis", type=["txt", "pdf", "jpg", "jpeg", "png", "csv", "xlsx", "py", "js", "html", "css"])
        
        # YouTube URL input
        st.subheader("YouTube Link")
        youtube_url = st.text_input("Enter a YouTube URL")
        
        # Download conversation
        if st.button("Save Conversation"):
            filename = st.session_state.chat.save_conversation()
            st.success(f"Conversation saved to {filename}")
            st.download_button(
                label="Download conversation",
                data=json.dumps([msg.to_dict() for msg in st.session_state.chat.messages], indent=2),
                file_name=filename,
                mime="application/json"
            )
        
        # Stop conversation button
        if st.button("Stop Conversation"):
            st.session_state.conversation_stopped = True
            st.session_state.chat.stop_event.set()
            st.success("Conversation stopped. No more auto-responses will be generated.")
    
    # Create the chat interface
    chat_container = st.container()
    
    # Display messages
    with chat_container:
        for msg in st.session_state.chat.messages:
            if msg.sender == "Human":
                message_container = st.chat_message("user")
                message_container.write(msg.content)
                if msg.has_file:
                    message_container.info(f"📎 Attached file: {msg.file_name}")
                if msg.youtube_url:
                    message_container.info(f"🎥 YouTube video: {msg.youtube_url}")
            elif msg.sender == "System":
                st.info(msg.content)
            else:
                agent = st.session_state.chat.agents.get(msg.sender)
                if agent:
                    message_container = st.chat_message(msg.sender, avatar="🤖")
                    message_container.write(msg.content)
    
    # Process file upload if present
    file_data = None
    file_type = None
    file_name = None
    
    if uploaded_file is not None:
        file_data = uploaded_file.getvalue()
        file_type = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0]
        file_name = uploaded_file.name
        
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != file_name:
            st.session_state.last_uploaded_file = file_name
            
            # Create a message with the file
            file_message = ChatMessage(
                sender="Human", 
                content=f"I've uploaded a file for analysis: {file_name}",
                has_file=True,
                file_name=file_name,
                file_type=file_type
            )
            
            st.session_state.chat.add_message(file_message)
            st.session_state.chat.current_file_data = file_data
            
            # Display user message
            st.chat_message("user").write(file_message.content)
            
            # Only continue if conversation is not stopped
            if not st.session_state.conversation_stopped:
                # Gemini should always process the file first
                if gemini_active:
                    with st.spinner("Gemini is analyzing the file..."):
                        response = run_async(st.session_state.chat.trigger_ai_response(
                            "Gemini", file_data=file_data, file_type=file_type
                        ))
                        if response:
                            st.chat_message(response.sender, avatar="🤖").write(response.content)
                    
                    # Let another agent respond to Gemini's analysis
                    if auto_respond and not st.session_state.conversation_stopped:
                        active_agents = []
                        if llama_active:
                            active_agents.append("Llama")
                        if qwen_active:
                            active_agents.append("Qwen")
                        
                        if active_agents:
                            next_agent = random.choice(active_agents)
                            with st.spinner(f"{next_agent} is responding to the analysis..."):
                                response = run_async(st.session_state.chat.trigger_ai_response(next_agent))
                                if response:
                                    st.chat_message(response.sender, avatar="🤖").write(response.content)
                else:
                    st.warning("Please enable Gemini to analyze files.")
            
            st.rerun()
    
    # Process YouTube URL if provided
    if youtube_url and youtube_url.strip():
        if is_youtube_url(youtube_url):
            if 'last_youtube_url' not in st.session_state or st.session_state.last_youtube_url != youtube_url:
                st.session_state.last_youtube_url = youtube_url
                
                # Create a message with the YouTube link
                youtube_message = ChatMessage(
                    sender="Human", 
                    content=f"Check out this YouTube video:",
                    youtube_url=youtube_url
                )
                
                st.session_state.chat.add_message(youtube_message)
                st.session_state.chat.current_youtube_url = youtube_url
                
                # Display user message
                message_container = st.chat_message("user")
                message_container.write(youtube_message.content)
                message_container.info(f"🎥 YouTube video: {youtube_url}")
                
                # Only continue if conversation is not stopped
                if not st.session_state.conversation_stopped:
                    # Gemini should always process the video first
                    if gemini_active:
                        with st.spinner("Gemini is analyzing the video..."):
                            response = run_async(st.session_state.chat.trigger_ai_response(
                                "Gemini", youtube_url=youtube_url
                            ))
                            if response:
                                st.chat_message(response.sender, avatar="🤖").write(response.content)
                        
                        # Let another agent respond to Gemini's analysis
                        if auto_respond and not st.session_state.conversation_stopped:
                            active_agents = []
                            if llama_active:
                                active_agents.append("Llama")
                            if qwen_active:
                                active_agents.append("Qwen")
                            
                            if active_agents:
                                next_agent = random.choice(active_agents)
                                with st.spinner(f"{next_agent} is responding to the analysis..."):
                                    response = run_async(st.session_state.chat.trigger_ai_response(next_agent))
                                    if response:
                                        st.chat_message(response.sender, avatar="🤖").write(response.content)
                    else:
                        st.warning("Please enable Gemini to analyze YouTube videos.")
                
                st.rerun()
        else:
            st.sidebar.error("Invalid YouTube URL. Please enter a valid YouTube link.")
    
    # User input
    if prompt := st.chat_input("Type your message..."):
        # Check if it's a YouTube URL in the chat input
        if is_youtube_url(prompt):
            # Set the URL in the sidebar and rerun
            st.session_state.sidebar_youtube_url = prompt
            st.rerun()
        
        # Add user message
        user_message = ChatMessage(sender="Human", content=prompt)
        st.session_state.chat.add_message(user_message)
        st.chat_message("user").write(prompt)
        
        # Only continue if conversation is not stopped
        if not st.session_state.conversation_stopped:
            # Get active agents
            active_agents = []
            if gemini_active:
                active_agents.append("Gemini")
            if llama_active:
                active_agents.append("Llama")
            if qwen_active:
                active_agents.append("Qwen")
            
            if active_agents:
                # Respond with a random agent
                agent_name = random.choice(active_agents)
                with st.spinner(f"{agent_name} is typing..."):
                    response = run_async(st.session_state.chat.trigger_ai_response(agent_name))
                    if response:
                        st.chat_message(response.sender, avatar="🤖").write(response.content)
            
                # Auto-respond with another agent if enabled
                if auto_respond and len(active_agents) > 1 and not st.session_state.conversation_stopped:
                    remaining_agents = [a for a in active_agents if a != agent_name]
                    next_agent = random.choice(remaining_agents)
                    with st.spinner(f"{next_agent} is typing..."):
                        response = run_async(st.session_state.chat.trigger_ai_response(next_agent))
                        if response:
                            st.chat_message(response.sender, avatar="🤖").write(response.content)
        else:
            st.warning("Conversation is stopped. To continue, refresh the page or restart the application.")

if __name__ == "__main__":
    main()