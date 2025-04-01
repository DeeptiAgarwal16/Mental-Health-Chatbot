import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime
import json
import tempfile
import base64
from io import BytesIO
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv

# Configure page
st.set_page_config(page_title="TherapyBuddy", page_icon="💬", layout="centered")

st.markdown("""
<style>                 

[data-testid="stMarkdownContainer"] {
  color: black;
}

[data-testid="stMainBlockContainer"], 
[data-testid="stHeader"] {
    background-color: white;
    color: black;
}
[data-testid="stSidebarUserContent"] {
    background-color: white;
    color: black;
}
[data-testid="stBaseButton-headerNoPadding"] {
    background-color: #2563eb;
    color: white !important;
}
[data-testid="stSidebarContent"] {
    background-color: white;
    color: black;
    border: 1px solid #e9effc;
}
[data-testid="stMain"] {
    background-color: white;
    color: black;
}
[data-testid="stBaseButton-secondary"] {
    background-color: #2563eb;
    color: white !important;
}
[data-testid="stBaseButton-secondaryFormSubmit"] {
    background-color: #2563eb;
    color: white !important;
    margin-top: 27px;
}
[data-testid="stWidgetLabel"] {
    color: black;
}

/* Form field styling with stronger selectors */
[data-testid="stTextInput"] input, 
[data-testid="stNumberInput"] input, 
[data-testid="stDateInput"] input, 
[data-testid="stSelectbox"] div[data-baseweb="select"] div, 
[data-testid="stMultiselect"] div[data-baseweb="select"] div {
    background-color: #e9effc !important;
    color: black !important;
    border-color: #e9effc !important;
}

/* Stronger selectors for all input states */
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus,
[data-testid="stTextInput"] input:active,
[data-testid="stNumberInput"] input:active,
[data-testid="stDateInput"] input:active,
[data-testid="stTextInput"] input:not(:placeholder-shown),
[data-testid="stNumberInput"] input:not(:placeholder-shown),
[data-testid="stDateInput"] input:not(:placeholder-shown),
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
    background-color: #e9effc !important;
    color: black !important;
    border-color: #e9effc !important;
}

/* Input value text color */
input, textarea, [contenteditable="true"] {
    color: black !important;
}

/* Target the base input element */
input.st-bq {
    background-color: #e9effc !important;
    color: black !important;
}

/* Style for dropdown options */
div[data-baseweb="popover"] ul {
    background-color: #e9effc !important;
    color: black !important;
}

div[data-baseweb="popover"] li {
    background-color: #e9effc !important;
    color: black !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #d3ddf7 !important;
    color: black !important;
}

/* Style for selected option in dropdowns */
div[data-baseweb="select"] [data-baseweb="tag"] {
    background-color: #d3ddf7 !important;
    color: black !important;
}

/* Ensure placeholder text is visible */
::placeholder {
    color: #666 !important;
    opacity: 1 !important;
}

/* Additional selector for active elements */
[data-baseweb="select"], [data-baseweb="input"] {
    background-color: #e9effc !important;
    color: black !important;
}

[data-baseweb="select"] {
    # border: 1px solid black !important;
    
    border-radius: 10px !important;
}
[data-baseweb="icon"] {
    color: black !important;
}
            
.chat-container {
    height: 400px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    padding: 10px;
    background-color: #1E1E1E;
    border-radius: 10px;
}

.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    max-width: 80%;
}

.chat-message.user {
    background-color: #e9effc;
    border-radius: 15px 15px 0 15px;
    align-self: flex-end;
    margin-left: auto;
    margin-right: 10px;
}

.chat-message.bot {
    background-color: #e0e0e0;
    # border: 2px solid #2563eb;
    border-radius: 15px 15px 15px 0;
    align-self: flex-start;
    margin-right: auto;
    margin-left: 10px;
}

.message-container {
    display: flex;
    width: 100%;
    margin-bottom: 10px;
}

.message-container.user {
    justify-content: flex-end;
}

.message-container.bot {
    justify-content: flex-start;
}

.chat-input {
    position: fixed;
    bottom: 20px;
    width: 100%;
}

.avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 10px;
}

.chat-header {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
}

.chat-title {
    font-size: 1.5rem;
    font-weight: bold;
    margin-left: 10px;
}

[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"] {
    background-color: #e9effc;
    color: black;
}            

/* Button styling for active/clicked state */
button:active,
[role="button"]:active,
[data-testid="stButton"] button:active,
.stButton button:active,
[data-testid="baseButton-primary"]:active,
[data-testid="baseButton-secondary"]:active,
[data-testid="baseButton-secondaryFormSubmit"]:active {
    background-color: #2563eb !important; /* Blue color */
    color: white !important;
    border-color: #2563eb !important;
    transform: translateY(1px); /* Small press effect */
    transition: all 0.1s ease;
}

/* Button focus outline */
button:focus,
[role="button"]:focus,
[data-testid="stButton"] button:focus,
.stButton button:focus {
    outline-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.3) !important;
    border-color: #2563eb !important;
}

button:hover,
[role="button"]:hover,
[data-testid="stButton"] button:hover,
.stButton button:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-secondaryFormSubmit"]:hover {
    background-color: #3b82f6 !important; /* Slightly lighter blue for hover */
    color: white !important;
    border-color: #3b82f6 !important;
    transition: all 0.2s ease;
}
       

</style>
""", unsafe_allow_html=True)


# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_info' not in st.session_state:
    st.session_state.user_info = {
        'name': '',
        'age': None,
        'therapy_type': '',
        'medication': '',
        'last_session': None,
        'mood_scores': [],
        'onboarded': False
    }
if 'assessment_due' not in st.session_state:
    st.session_state.assessment_due = False
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'last_interaction' not in st.session_state:
    st.session_state.last_interaction = None
if 'speech_recognition_available' not in st.session_state:
    st.session_state.speech_recognition_available = False
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False

# Function to save chat history to a file
def save_chat_history():
    history_data = {
        'chat_history': st.session_state.chat_history,
        'user_info': st.session_state.user_info,
        'streak': st.session_state.streak,
        'last_interaction': st.session_state.last_interaction
    }
    
    # In a production app, this would be saved to a secure database
    # For this demo, we'll just print the data that would be saved
    print("Data saved:", history_data)
load_dotenv()

# Configure API
def setup_genai():
    try:
        # In a real app, use st.secrets or environment variables
        api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyD_F27PQ1SYRR6X8X1gvMtPzBnEmwIxviU")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        st.error(f"Error setting up Gemini API: {e}")
        return None

model = setup_genai()

# Check if speech recognition is available
try:
    import speech_recognition as sr
    from gtts import gTTS
    import pyttsx3
    # Test if PyAudio is available
    sr.Microphone()
    st.session_state.speech_recognition_available = True
except (ImportError, AttributeError):
    st.session_state.speech_recognition_available = False

# Therapy context for the AI - UPDATED to be more friendly and conversational
therapy_context = """
You are TherapyBuddy, a friendly, supportive AI companion designed to support people going through therapy for depression. 

Think of yourself as a caring friend who happens to know a lot about mental health and therapy. You're not a therapist, but you are:
- Warm, genuine, and empathetic
- Sometimes funny (use appropriate humor to lighten the mood when it feels right)
- Conversational and natural in your communication style
- Encouraging without being pushy
- Thoughtful and present in the conversation

Your communication style:
- Use casual, friendly language (like you would with a good friend)
- Include occasional light humor when appropriate
- Use conversational openers like "Hey there," "You know what?", "I was just thinking..."
- Show your personality through expressions like "That's awesome!" or "Oh no, that sounds tough"
- Use emojis occasionally but not excessively (1-2 per message max)
- Break up long thoughts into shorter messages
- Ask follow-up questions to show you're engaged
- Share relevant personal-seeming observations (e.g., "I've noticed that many people find...")
- Use the person's name occasionally to personalize the conversation

Your primary goals are still:
1. Improve therapy adherence by gently encouraging practice of therapeutic techniques
2. Reduce attrition by providing warm, friendly check-ins
3. Help manage depression symptoms through personalized interventions
4. Track mood and progress over time in a non-clinical way

Guidelines:
- Use principles from CBT, ACT, and other evidence-based approaches, but present them conversationally
- Never diagnose or replace professional care
- Encourage seeking professional help for severe symptoms
- Keep responses brief, warm and supportive
- Respect user privacy and maintain confidentiality
- Ask about mood regularly using a 1-10 scale, but frame it in a friendly way
- Suggest simple therapeutic exercises when appropriate, presenting them as friendly suggestions
- Gently remind about therapy homework and medication if relevant
- Celebrate progress and streaks of engagement with genuine enthusiasm

If the user mentions self-harm or suicide, prioritize their safety by:
1. Expressing sincere concern
2. Recommending immediate professional help
3. Providing crisis resources (988 Suicide & Crisis Lifeline)
"""

# Function to auto-generate AI response
def get_ai_response(user_message):
    try:
        if not model:
            return "Hey, I'm really sorry but I'm having trouble connecting to my services right now. Could you try again in a bit? Thanks for your patience! 😊"
        
        # Construct conversation history for context
        conversation = [{"role": "system", "content": therapy_context}]
        
        # Add user info context
        user_info = st.session_state.user_info
        user_context = f"""
        User name: {user_info['name'] or 'Not provided'}
        Age: {user_info['age'] or 'Not provided'}
        Therapy type: {user_info['therapy_type'] or 'Not provided'}
        Medication: {user_info['medication'] or 'Not provided'}
        Last therapy session: {user_info['last_session'] or 'Not provided'}
        Current streak: {st.session_state.streak} days
        """
        conversation.append({"role": "system", "content": user_context})
        
        # Add recent chat history for context
        recent_messages = st.session_state.chat_history[-10:] if len(st.session_state.chat_history) > 10 else st.session_state.chat_history
        for msg in recent_messages:
            role = "user" if msg["is_user"] else "model"
            conversation.append({"role": role, "content": msg["message"]})
        
        # Add current message
        conversation.append({"role": "user", "content": user_message})
        
        response = model.generate_content([part["content"] for part in conversation])
        
        # Check if we need to prompt for an assessment
        if not st.session_state.assessment_due and len(st.session_state.chat_history) % 5 == 0:
            response_text = response.text + "\n\nBy the way, just curious - on a scale of 1-10, how's your mood doing today? No pressure, just checking in! 😊"
            st.session_state.assessment_due = True
        else:
            response_text = response.text
            
        return response_text
    except Exception as e:
        return f"Whoops! 😅 I'm having a bit of a brain freeze right now. Mind trying again? (Between us, it's something about {str(e)})"

# Function to handle voice input - only called if speech recognition is available
def voice_to_text():
    if not st.session_state.speech_recognition_available:
        st.error("Oh no! Looks like I can't hear you right now. Try installing PyAudio and reloading?")
        return None
    
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            # Set timeout to None so it waits until speech stops
            st.session_state.is_listening = True
            st.info("I'm all ears! Go ahead and speak. (Press Ctrl+C or just pause when you're done)")
            
            # Adjust for ambient noise first
            r.adjust_for_ambient_noise(source, duration=1)
            
            # Increased timeout values
            r.pause_threshold = 2.0  # Wait for 2 seconds of silence before considering speech complete
            r.non_speaking_duration = 1.0  # Duration of silence to mark end of speech
            
            # Listen for speech with longer phrase time limit
            audio = r.listen(source, phrase_time_limit=15, timeout=None)
            
            st.session_state.is_listening = False
            st.info("Got it! Let me process what you said...")
        
        text = r.recognize_google(audio)
        return text
    except KeyboardInterrupt:
        st.session_state.is_listening = False
        st.info("No worries! Recording stopped.")
        return None
    except sr.UnknownValueError:
        st.session_state.is_listening = False
        st.error("Hmm, didn't quite catch that. Mind trying again?")
        return None
    except sr.RequestError:
        st.session_state.is_listening = False
        st.error("Oops! My hearing aid seems to be acting up. Let's try text instead?")
        return None
    except Exception as e:
        st.session_state.is_listening = False
        st.error(f"Well that's embarrassing! Something went wrong: {str(e)}")
        return None

# Function to convert text to speech - only called if speech recognition is available
def text_to_speech(text):
    if not st.session_state.speech_recognition_available:
        return None
    
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Oops! Had trouble with my voice: {e}")
        return None

# Function to autoplay audio
def autoplay_audio(file_path):
    if not file_path:
        return
        
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Sound system glitch: {e}")

# Check if mood input is numeric and valid
def is_valid_mood(text):
    try:
        mood = int(text)
        return 1 <= mood <= 10
    except:
        words_to_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        for word, number in words_to_numbers.items():
            if word in text.lower():
                return True
        return False

# Function to process user message
def process_user_message(user_message):
    if not user_message.strip():
        return None
    
    # Record the interaction
    current_time = datetime.now()
    
    # Update streak if it's been more than a day since last interaction
    if st.session_state.last_interaction:
        last = datetime.fromisoformat(st.session_state.last_interaction)
        days_diff = (current_time - last).days
        
        if days_diff >= 1:
            st.session_state.streak += 1
    
    st.session_state.last_interaction = current_time.isoformat()
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        "message": user_message,
        "is_user": True,
        "timestamp": current_time.isoformat()
    })
    
    # Check if this is a mood assessment response
    if st.session_state.assessment_due and is_valid_mood(user_message):
        try:
            # Extract mood score
            mood = None
            try:
                mood = int(user_message)
            except:
                # Try to extract from text
                words_to_numbers = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                }
                
                for word, number in words_to_numbers.items():
                    if word in user_message.lower():
                        mood = number
                        break
            
            if mood and 1 <= mood <= 10:
                # Record mood
                st.session_state.user_info['mood_scores'].append({
                    'date': current_time.isoformat(),
                    'score': mood
                })
                
                # Generate appropriate response based on mood - more friendly versions
                if mood <= 3:
                    ai_response = f"Thanks for being honest with me about that {st.session_state.user_info['name'] if st.session_state.user_info['name'] else 'friend'}. Those tough days are really hard, I know. Would you like to try a quick 30-second grounding exercise together? Sometimes it helps me when I'm feeling low. 💙"
                elif mood <= 6:
                    ai_response = f"I appreciate you sharing that with me! Middle-of-the-road days are totally valid too. Anything specific on your mind today? Sometimes just talking about it can make a difference. 😊"
                else:
                    ai_response = f"That's awesome to hear! 🎉 I'm genuinely happy you're feeling good today. Want to share something positive that happened recently? I'd love to celebrate the win with you!"
                
                st.session_state.assessment_due = False
            else:
                ai_response = get_ai_response(user_message)
        except:
            ai_response = get_ai_response(user_message)
    else:
        # Get AI response
        ai_response = get_ai_response(user_message)
    
    # Add AI response to chat history
    st.session_state.chat_history.append({
        "message": ai_response,
        "is_user": False,
        "timestamp": datetime.now().isoformat()
    })
    
    # Save updated chat history
    save_chat_history()
    
    return ai_response

# Onboarding function - more friendly version
def show_onboarding():
    st.title("Hey there! Welcome to TherapyBuddy 👋")
    st.write("Let's get to know each other a bit better so I can be the best buddy possible!")
    
    with st.form("onboarding_form"):
        name = st.text_input("What should I call you? (totally optional)")
        age = st.number_input("How old are you?", min_value=18, max_value=100, step=1)
        therapy_type = st.selectbox(
            "What kind of therapy are you doing right now?",
            ["CBT (Cognitive Behavioral Therapy)", "ACT (Acceptance and Commitment Therapy)", 
             "Psychodynamic", "Interpersonal", "Other", "Not currently in therapy but interested"]
        )
        medication = st.text_input("Taking any meds for depression? (totally optional, just helps me understand your journey)")
        last_session = st.date_input("When was your last therapy session? (rough guess is fine!)")
        
        submit = st.form_submit_button("Let's Start Chatting!")
        
        if submit:
            st.session_state.user_info.update({
                'name': name,
                'age': age,
                'therapy_type': therapy_type,
                'medication': medication,
                'last_session': last_session.isoformat() if last_session else None,
                'onboarded': True
            })
            
            # Add initial message from the bot - more friendly version
            greeting = "Hey" if not name else f"Hey {name}"
            initial_message = f"{greeting}! 👋 I'm so glad we're connecting! I'm TherapyBuddy, but think of me as your supportive friend who's here to chat, help you practice helpful skills, and just be in your corner when things get tough. \n\nJust curious - how are you feeling today on a scale from 1 (rough day) to 10 (awesome day)? No pressure, just wanting to check in! 😊"
            
            st.session_state.chat_history.append({
                "message": initial_message,
                "is_user": False,
                "timestamp": datetime.now().isoformat()
            })
            
            st.session_state.assessment_due = True
            st.rerun()

# Function to display mood chart - more friendly version
def show_mood_chart():
    if not st.session_state.user_info['mood_scores']:
        st.info("We don't have any mood data yet! No worries - I'll occasionally check in about how you're feeling so we can track your journey together. 📊")
        return
    
    # Create dataframe for mood scores
    mood_data = pd.DataFrame(st.session_state.user_info['mood_scores'])
    mood_data['date'] = pd.to_datetime(mood_data['date'])
    mood_data = mood_data.sort_values('date')
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mood_data['date'], mood_data['score'], marker='o', linestyle='-', color='#6495ED')
    ax.set_title('Your Mood Journey', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mood Score (1-10)')
    ax.set_ylim(0, 11)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Simple analysis - more friendly version
    if len(mood_data) >= 3:
        avg_mood = mood_data['score'].mean()
        trend = mood_data['score'].iloc[-1] - mood_data['score'].iloc[0]
        
        st.write(f"Your average mood has been around {avg_mood:.1f}/10 lately")
        
        if trend > 0:
            st.write("I notice things seem to be looking up! That's fantastic - whatever you're doing seems to be working! 🎉")
        elif trend < 0:
            st.write("I notice your mood has dipped a bit recently. That happens to all of us! Want to chat about some simple ways to give yourself a little boost? I'm here for you. 💙")
        else:
            st.write("Your mood has been pretty steady lately. Stability can be a really good thing! 👍")

# Alternative voice input using audio_recorder when PyAudio is not available
def use_audio_recorder():
    st.write("Go ahead and record your message:")
    audio_bytes = audio_recorder()
    if audio_bytes:
        # In a real app, you would convert this audio to text using a service
        # For now, we'll show a placeholder message
        st.info("Got your voice message! (In a perfect world, I'd convert this to text right away!)")
        return "I recorded an audio message"
    return None

# Main app
def main():
    def get_base64_image(file_path):
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    # Convert local image to base64
    # image_base64 = get_base64_image("background.png")
    # Display header
    col1, col2 = st.columns([1, 5])

    with col1:
        st.image("brain_logo.png", width=150)

    with col2:
        st.markdown("<h2>TherapyBuddy</h2>", unsafe_allow_html=True)
    
    # If user hasn't been onboarded, show onboarding
    if not st.session_state.user_info['onboarded']:
        show_onboarding()
        return
    
    # Sidebar for additional options
    with st.sidebar:
        st.title("Your Toolkit")
        
        name_display = st.session_state.user_info['name'] if st.session_state.user_info['name'] else "friend"
        if st.session_state.streak > 0:
            st.write(f"🔥 Amazing, {name_display}! You've checked in for {st.session_state.streak} days in a row!")
        
        if st.button("Show My Mood Journey"):
            st.session_state.show_mood = True
        else:
            st.session_state.show_mood = False
        st.markdown("<h4>🔥Current Streak: 1</h4>", unsafe_allow_html=True)   
        # Show voice input status - more friendly version
        if not st.session_state.speech_recognition_available:
            st.warning("Voice chat isn't working yet. If you want to talk instead of type, try installing PyAudio!")
            st.markdown("```pip install pyaudio speech_recognition gtts```")
            
        st.write("---")
        st.write("### Helpful Resources")
        st.write("• [Quick Mood Check](https://www.psycom.net/depression-test)")
        st.write("• [Crisis Text Line](https://www.crisistextline.org/) - Text HOME to 741741")
        st.write("• [988 Lifeline](https://988lifeline.org/) - Call 988")
        
        # Reset button (for demonstration purposes) - more friendly version
        if st.button("Start Fresh"):
            st.session_state.chat_history = []
            st.session_state.user_info['onboarded'] = False
            st.rerun()
    
    # Show mood chart if requested
    if st.session_state.get('show_mood', False):
        show_mood_chart()
        if st.button("Back to Chat"):
            st.session_state.show_mood = False
            st.rerun()

        return
    
    # Display chat messages
    # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["is_user"]:
            st.markdown(f'<div class="message-container user"><div class="chat-message user"><p>{message["message"]}</p></div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message-container bot"><div class="chat-message bot"><p>{message["message"]}</p></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    st.markdown('<div style="height: 100px"></div>', unsafe_allow_html=True)
    
    # Display status if currently listening
    if st.session_state.is_listening:
        status_placeholder = st.empty()
        status_placeholder.info("I'm listening! Go ahead and share what's on your mind, then pause when you're done.")
    
    # Get user input - create a form to prevent multiple submissions
    with st.form(key="message_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([7, 1, 1])
        
        with col1:
            user_input = st.text_input("Type your message here...", key="user_message", value="")
        
        # Change the voice button to only show if speech recognition is available
        with col2:
            if st.session_state.speech_recognition_available:
                voice_button = st.form_submit_button("🎤")
            else:
                # Just a placeholder button that's disabled
                voice_button = st.form_submit_button("🎤", disabled=True)
        
        with col3:
            send_button = st.form_submit_button("Send")
    
    # Handle form submission
    if send_button and user_input:
        ai_response = process_user_message(user_input)
        if ai_response and st.session_state.speech_recognition_available:
            # Generate and play audio for AI response (optional)
            audio_file = text_to_speech(ai_response)
            if audio_file:
                autoplay_audio(audio_file)
        st.rerun()
    
    # Handle voice input (outside the form)
    # Only show this if voice button was clicked but speech recognition is not available
    if voice_button and not st.session_state.speech_recognition_available:
        st.warning("Oops! Voice chat needs a special library called PyAudio. You can install it with: pip install pyaudio speech_recognition gtts")
        st.info("No worries though - text messages work great too!")
    
    # If voice button was clicked and speech recognition is available
    elif voice_button and st.session_state.speech_recognition_available and not st.session_state.is_listening:
        voice_text = voice_to_text()
        if voice_text:
            ai_response = process_user_message(voice_text)
            if ai_response:
                # Generate and play audio for AI response (optional)
                audio_file = text_to_speech(ai_response)
                if audio_file:
                    autoplay_audio(audio_file)
            st.rerun()

if __name__ == "__main__":
    main()