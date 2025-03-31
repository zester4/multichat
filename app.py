import streamlit as st
import os
import asyncio
import json
import random
import time
from typing import List, Dict, Any, Optional
import threading
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Gemini implementation
def gemini_generate(prompt):
    import os
    from google import genai
    from google.genai import types
    
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    model = "gemini-2.5-pro-exp-03-25"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
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
    def __init__(self, sender: str, content: str, timestamp: Optional[float] = None):
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or time.time()
    
    def to_dict(self):
        return {
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    def __str__(self):
        return f"[{self.sender}]: {self.content}"

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
    
    async def generate_response(self, context: List[ChatMessage]) -> str:
        # Format context into a single prompt
        formatted_context = "\n".join([str(msg) for msg in context[-15:]])
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
    
    def add_agent(self, agent: ChatAgent):
        self.agents[agent.name] = agent
    
    def add_message(self, message: ChatMessage):
        self.messages.append(message)
        # Also add to each agent's history
        for agent in self.agents.values():
            agent.add_to_history(message)
    
    async def trigger_ai_response(self, agent_name, initiator=None):
        """Trigger an AI agent to respond"""
        agent = self.agents[agent_name]
        
        # Skip if this agent was the last to speak
        if self.messages and self.messages[-1].sender == agent_name:
            return None
        
        try:
            # Generate response (with timeout)
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
            content="Welcome to Bytes & Brains! You're now chatting with Gemini, Llama, and Qwen. Feel free to start a conversation!"
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
                st.chat_message("user").write(msg.content)
            elif msg.sender == "System":
                st.info(msg.content)
            else:
                agent = st.session_state.chat.agents.get(msg.sender)
                if agent:
                    st.chat_message(msg.sender, avatar="🤖").write(msg.content)
    
    # User input
    if prompt := st.chat_input("Type your message..."):
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