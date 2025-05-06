# streamlit_app.py
import streamlit as sl
import os
import random
import pandas as pd
from typing import TypedDict, List, Dict, Optional, Annotated, Sequence
import sqlite3
import json
import uuid
import time
from pathlib import Path  # Use pathlib for better path handling
from dotenv import load_dotenv

# --- LangChain & LangGraph ---
# Ensure libraries are installed in the environment (e.g., requirements.txt)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# --- Configuration ---
# Assume dataset is in a 'data' subdirectory relative to the script
# OR configure via environment variable
DATASET_DIR_NAME = "stories"  # Keep the folder name consistent
APP_DIR = Path(__file__).parent  # Get the directory where the script is running
DATASET_PATH = Path(
    os.environ.get("DATASET_PATH", APP_DIR / "data" / DATASET_DIR_NAME)
)  # Env var or default location
MANIFEST_FILENAME = "stories_manifest.csv"
DEFAULT_LANGUAGE = "English"
MANIFEST_FULL_PATH = DATASET_PATH / MANIFEST_FILENAME

# --- Validate Dataset Path ---
if not DATASET_PATH.is_dir() or not MANIFEST_FULL_PATH.is_file():
    sl.error(
        f"Error: Dataset path '{DATASET_PATH}' not found or manifest '{MANIFEST_FILENAME}' missing."
    )
    sl.error(
        "Please ensure the dataset folder exists relative to the script or set the DATASET_PATH environment variable."
    )
    sl.stop()  # Stop execution if dataset is missing

# --- Google AI API Key Setup ---
# ** READ FROM ENV FILE **
load_dotenv()
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")

LLM_READY = False
llm = None
if GOOGLE_AI_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=GOOGLE_AI_API_KEY,
            temperature=0.3,
        )
        print(f"Google AI LLM Model initialized ('{llm.model}').")
        LLM_READY = True
    except Exception as e:
        sl.error(f"ERROR during Google AI (Gemini) Setup: {e}")
        LLM_READY = False
else:
    sl.error(
        "GOOGLE_AI_API_KEY environment variable not set. Storyteller functionality disabled."
    )
    # sl.stop() # Optionally stop if LLM is critical


# --- LangGraph State Definition ---
class StorytellerAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    dataset_base_path: str  # Store as string for checkpointer
    stories_manifest: Optional[List[Dict]]
    supported_languages_map: Dict[str, str]
    selected_language: str
    stories_to_play_cycle: List[str]
    played_in_cycle: List[str]
    error_message: Optional[str]
    selected_story_id: Optional[str]
    selected_story_title: Optional[str]
    selected_story_content: Optional[str]
    selected_audio_path: Optional[str]
    user_request_type: Optional[str]
    extracted_language: Optional[str]
    output_audio_path: Optional[str]


# --- Node Function Definitions ---
# (Copy the LATEST versions of load_manifest_and_languages_node,
# determine_intent_llm_node, handle_language_change_node,
# filter_stories_by_language_node, select_story_node,
# fetch_story_data_node, format_response_node here. They remain the same.)
# --- PASTE NODE DEFINITIONS HERE ---
def load_manifest_and_languages_node(state: StorytellerAgentState) -> dict:
    print("--- Node: load_manifest_and_languages ---")
    if (
        state.get("stories_manifest") is not None and state.get("stories_manifest")
    ) and (
        state.get("supported_languages_map") is not None
        and state.get("supported_languages_map")
    ):
        print("Manifest (list) and languages already loaded in state.")
        if (
            state.get("error_message")
            and "Manifest file not found" in state["error_message"]
        ):
            return {"error_message": None}
        return {}
    base_path = state["dataset_base_path"]
    manifest_path = os.path.join(base_path, MANIFEST_FILENAME)
    manifest_list = None
    languages_map = {}
    error = None
    try:
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        manifest_df = pd.read_csv(manifest_path)
        required_cols = [
            "story_id",
            "language",
            "title",
            "file_path",
            "audio_file_path",
        ]
        if not all(col in manifest_df.columns for col in required_cols):
            raise ValueError(f"Manifest missing cols: {required_cols}")
        manifest_df["language"] = manifest_df["language"].astype(str)
        manifest_list = manifest_df.to_dict("records")
        print(f"Loaded manifest list ({len(manifest_list)} entries).")
        unique_languages = set(
            item["language"] for item in manifest_list if "language" in item
        )
        for lang_name in sorted(list(unique_languages)):
            lang_path = os.path.join(base_path, lang_name)
            languages_map[lang_name] = lang_path
            print(f"  Lang support: '{lang_name}'")
        if not languages_map:
            lang_error = "No languages discovered."
            error = error + f"; {lang_error}" if error else lang_error
    except FileNotFoundError as e:
        error = str(e)
        print(error)
    except Exception as e:
        error = f"Error loading manifest: {e}"
        print(error)
        manifest_list = None
    stories_to_play = state.get("stories_to_play_cycle", [])
    played = state.get("played_in_cycle", [])
    update_dict = {
        "stories_manifest": manifest_list,
        "supported_languages_map": languages_map,
        "error_message": error,
    }
    if state.get("stories_to_play_cycle") is None:
        update_dict["stories_to_play_cycle"] = []
    if state.get("played_in_cycle") is None:
        update_dict["played_in_cycle"] = []
    return update_dict


def determine_intent_llm_node(state: StorytellerAgentState) -> dict:
    print("--- Node: determine_intent_llm ---")
    error = state.get("error_message")
    if state.get("stories_manifest") is None or not state.get(
        "supported_languages_map"
    ):
        if not error:
            error = "Manifest/Language data failed to load previously."
        print(f"Critical Error before intent: {error}")
        return {
            "user_request_type": "error_state",
            "error_message": error,
            "selected_story_id": None,
            "selected_story_title": None,
            "selected_story_content": None,
            "selected_audio_path": None,
            "output_audio_path": None,
        }
    if not LLM_READY or not llm:
        print("LLM not available, using basic keyword intent detection.")
        user_message = state["messages"][-1].content.lower()
        request_type = "unknown"
        if "story" in user_message or "tell me" in user_message:
            request_type = "request_story"
        elif "hello" in user_message or "hi" in user_message:
            request_type = "greeting"
        elif "language" in user_message:
            request_type = "ask_language_options"
        return {
            "user_request_type": request_type,
            "extracted_language": None,
            "error_message": error,
            "selected_story_id": None,
            "selected_story_title": None,
            "selected_story_content": None,
            "selected_audio_path": None,
            "output_audio_path": None,
        }
    human_input = state["messages"][-1].content
    history = state["messages"][:-1]
    chat_history_str = "\n".join(
        [
            f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}"
            for m in history[-4:]
        ]
    )
    available_lang_list = list(state.get("supported_languages_map", {}).keys())
    available_lang_str = (
        ", ".join(sorted(available_lang_list))
        if available_lang_list
        else "None specified"
    )
    prompt = f"""Analyze the user's latest message to a storyteller bot. Classify the intent: request_story, \
    greeting, ask_language_options, set_language, inquire_capabilities, unknown.
    Available languages: {available_lang_str}
    If intent is 'set_language', extract the language name.
    History: {chat_history_str}
    User Message: "{human_input}" Output ONLY JSON: {{"intent": "...", "language": "..." or null}}"""
    request_type = "unknown"
    extracted_lang = None
    llm_error = None
    try:
        print("Calling LLM for intent detection...")
        llm_response = llm.invoke(prompt)
        response_text = llm_response.content.strip()
        print(f"LLM Raw Response: {response_text}")
        try:
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json").strip()
            elif response_text.startswith("```"):
                response_text = response_text.strip("```").strip()
            intent_data = json.loads(response_text)
            request_type = intent_data.get("intent", "unknown")
            extracted_lang = intent_data.get("language")
            print(f"LLM Classified Intent: {request_type}, Language: {extracted_lang}")
        except json.JSONDecodeError:
            print(f"LLM response not valid JSON: {response_text}")
            llm_error = "LLM response parse error."
    except Exception as e:
        print(f"Error calling LLM: {e}")
        llm_error = f"LLM API call failed: {e}"
    final_error = error or llm_error
    return {
        "user_request_type": request_type,
        "extracted_language": extracted_lang,
        "error_message": final_error,
        "selected_story_id": None,
        "selected_story_title": None,
        "selected_story_content": None,
        "selected_audio_path": None,
        "output_audio_path": None,
    }


def handle_language_change_node(state: StorytellerAgentState) -> dict:
    print("--- Node: handle_language_change ---")
    intent = state.get("user_request_type")
    requested_lang = state.get("extracted_language")
    supported_map = state.get("supported_languages_map", {})
    available_langs = list(supported_map.keys())
    current_lang = state.get("selected_language")
    response_message = ""
    new_lang_set = False
    updates = {"error_message": None}
    if not supported_map:
        response_message = "Language support info not loaded."
    elif intent == "ask_language_options":
        response_message = (
            f"I can tell stories in: {', '.join(sorted(available_langs))}. Which one?"
        )
    elif intent == "set_language":
        target_lang = next(
            (
                lang
                for lang in available_langs
                if requested_lang and requested_lang.lower() == lang.lower()
            ),
            None,
        )
        if target_lang:
            if target_lang == current_lang:
                response_message = f"Already set to {target_lang}."
            else:
                response_message = f"Okay, language switched to {target_lang}."
                updates["selected_language"] = target_lang
                updates["stories_to_play_cycle"] = []
                updates["played_in_cycle"] = []
                new_lang_set = True
                print(f"Lang changed to {target_lang}")
        elif requested_lang:
            response_message = f"Sorry, '{requested_lang}' not supported. Available: {', '.join(sorted(available_langs))}."
        else:
            response_message = (
                f"Which language? Options: {', '.join(sorted(available_langs))}."
            )
    else:
        response_message = "Language handling error."
    ai_message = AIMessage(content=response_message)
    updates["messages"] = add_messages(state.get("messages", []), [ai_message])
    updates["user_request_type"] = "language_processed"
    return updates


def filter_stories_by_language_node(state: StorytellerAgentState) -> dict:
    print("--- Node: filter_stories_by_language ---")
    manifest_list = state.get("stories_manifest")
    language = state.get("selected_language")
    error = state.get("error_message")
    stories_to_play = state.get("stories_to_play_cycle", [])
    played_in_cycle = state.get("played_in_cycle", [])
    updates = {"error_message": error}
    if error:
        print("Skipping filter due to previous error.")
        return updates
    if not language:
        updates["error_message"] = "Cannot filter: Language not selected."
        print(updates["error_message"])
        return updates
    if manifest_list is None:
        updates["error_message"] = "Cannot filter: Manifest not loaded."
        print(updates["error_message"])
        return updates
    try:
        current_lang_stories = [
            s for s in manifest_list if s.get("language") == language
        ]
        if not current_lang_stories:
            raise ValueError(f"No stories found for '{language}'.")
        current_lang_story_ids = list(set(s["story_id"] for s in current_lang_stories))
        needs_reset = not stories_to_play or set(played_in_cycle) == set(
            current_lang_story_ids
        )
        if needs_reset:
            if set(played_in_cycle) == set(current_lang_story_ids):
                print("Reshuffling cycle...")
            else:
                print("Initializing cycle.")
            stories_to_play = current_lang_story_ids.copy()
            random.shuffle(stories_to_play)
            played_in_cycle = []
            last_played = (
                state.get("played_in_cycle", [])[-1]
                if state.get("played_in_cycle")
                else None
            )
            if (
                len(stories_to_play) > 1
                and last_played
                and stories_to_play[0] == last_played
            ):
                stories_to_play.pop(0)
        updates["stories_to_play_cycle"] = stories_to_play
        updates["played_in_cycle"] = played_in_cycle
        updates["error_message"] = None
        print(
            f"Filtered {len(current_lang_story_ids)} stories for '{language}'. Cycle has {len(stories_to_play)} remaining."
        )
    except Exception as e:
        updates["error_message"] = f"Error filtering: {e}"
        print(updates["error_message"])
    return updates


def select_story_node(state: StorytellerAgentState) -> dict:
    print("--- Node: select_story ---")
    stories_to_play = state.get("stories_to_play_cycle", [])
    played_in_cycle = state.get("played_in_cycle", [])
    error = state.get("error_message")
    updates = {"error_message": error}
    if error:
        print("Skipping selection due to previous error.")
        return updates
    if not stories_to_play:
        language = state.get("selected_language", "the current language")
        updates["error_message"] = (
            f"Cannot select story: Cycle empty/no stories found for {language}."
        )
        print(updates["error_message"])
        return updates
    selected_id = stories_to_play.pop(0)
    played_in_cycle.append(selected_id)
    updates["selected_story_id"] = selected_id
    updates["stories_to_play_cycle"] = stories_to_play
    updates["played_in_cycle"] = played_in_cycle
    updates["error_message"] = None
    print(f"Selected story ID: {selected_id}")
    return updates


def fetch_story_data_node(state: StorytellerAgentState) -> dict:
    print("--- Node: fetch_story_data ---")
    story_id = state.get("selected_story_id")
    manifest_list = state.get("stories_manifest")
    base_path = state.get("dataset_base_path")
    language = state.get("selected_language")
    error = state.get("error_message")
    updates = {
        "error_message": error,
        "selected_story_title": None,
        "selected_story_content": None,
        "selected_audio_path": None,
    }
    if error:
        print("Skipping fetch due to previous error.")
        return updates
    if not story_id or manifest_list is None or not language or not base_path:
        updates["error_message"] = "Cannot fetch: Missing inputs."
        print(updates["error_message"])
        return updates
    try:
        story_entry = next(
            (
                s
                for s in manifest_list
                if s.get("language") == language and s.get("story_id") == story_id
            ),
            None,
        )
        if not story_entry:
            raise FileNotFoundError(
                f"Entry missing for story '{story_id}' / lang '{language}'."
            )
        title = story_entry["title"]
        text_rel_path = story_entry.get("file_path")
        audio_rel_path = story_entry.get("audio_file_path")
        if not text_rel_path or not audio_rel_path:
            raise ValueError("Missing text or audio path in manifest.")
        text_full_path = os.path.join(base_path, text_rel_path.lstrip("/\\"))
        audio_full_path = os.path.join(base_path, audio_rel_path.lstrip("/\\"))
        if not os.path.exists(text_full_path):
            raise FileNotFoundError(f"Text file not found: {text_full_path}")
        with open(text_full_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"Fetched text content for '{title}'.")
        if not os.path.exists(audio_full_path):
            raise FileNotFoundError(f"Audio file not found: {audio_full_path}")
        print(f"Found audio path: {audio_full_path}")
        updates["selected_story_title"] = title
        updates["selected_story_content"] = content
        updates["selected_audio_path"] = str(Path(audio_full_path).resolve())
        updates["error_message"] = None  # Store absolute path
    except Exception as e:
        updates["error_message"] = f"Error fetching data: {e}"
        print(updates["error_message"])
    return updates


def format_response_node(state: StorytellerAgentState) -> dict:
    print("--- Node: format_response ---")
    error = state.get("error_message")
    request_type = state.get("user_request_type", "unknown")
    content = state.get("selected_story_content")
    title = state.get("selected_story_title")
    audio_path = state.get("selected_audio_path")
    final_response = ""
    output_audio = None
    # Clear error state here before potentially setting response based on it
    updates = {"error_message": None}

    if error:
        final_response = f"I encountered an issue: {error}"
        if request_type == "request_story" and (not content or not audio_path):
            final_response += ". I couldn't retrieve story details."
        final_response += ". Please try again."
    elif request_type == "request_story":
        if content and title and audio_path:
            final_response = f"Okay, here is the story '{title}':\n\n---\n\n{content}\n\n---\n\nListen to the audio below. What do you think the moral of the story is?"
            output_audio = audio_path  # Use full path from state
            updates["selected_story_title"] = title
            updates["selected_audio_path"] = audio_path
        else:
            final_response = "I wanted to tell a story, but couldn't retrieve details."
    elif request_type == "greeting":
        final_response = "Hello! Ask 'tell a story' or about 'languages'."
    elif request_type == "inquire_capabilities":
        final_response = "I tell stories! Ask 'tell a story' or about 'languages'."
    elif request_type == "language_processed":
        pass
    elif request_type == "error_state":
        final_response = "Sorry, an error occurred earlier. Can you try again?"
    else:
        final_response = "Not sure how to respond. Ask 'tell a story'."
    updates = {
        "error_message": None,
        "output_audio_path": output_audio,
        "selected_story_id": None,
        "selected_story_title": None,
        "selected_story_content": None,
        "selected_audio_path": None,
        "user_request_type": None,
        "extracted_language": None,
    }
    if final_response:
        ai_message = AIMessage(content=final_response)
        print(f"Formatted Response: {final_response[:100]}...")
        updates["messages"] = add_messages(state.get("messages", []), [ai_message])
    else:
        print("No new message generated by format_response.")
    return updates


# --- End Node Definitions ---


# --- Function to get available languages (for UI) ---
@sl.cache_data
def get_available_languages_from_manifest(path):
    languages = [DEFAULT_LANGUAGE]
    manifest_path = Path(path)
    try:
        if manifest_path.is_file():
            df = pd.read_csv(manifest_path)
            if "language" in df.columns:
                langs = df["language"].astype(str).unique().tolist()
                if DEFAULT_LANGUAGE in langs:
                    langs.remove(DEFAULT_LANGUAGE)
                    languages = [DEFAULT_LANGUAGE] + sorted(langs)
                else:
                    languages = sorted(langs)
        print(f"UI Languages found: {languages}")
        return languages
    except Exception as e:
        print(f"Error reading manifest for UI languages: {e}")
        return languages


# --- Compile Agent ---
@sl.cache_resource  # Cache the compiled agent & memory connection
def get_compiled_agent():
    if not LLM_READY:
        return None, None  # Don't compile if LLM failed
    memory_file = "streamlit_storyteller_node_memory.sqlite"
    conn = None
    try:
        conn = sqlite3.connect(
            memory_file, check_same_thread=False
        )  # Needs check_same_thread=False for Streamlit
        memory = SqliteSaver(conn=conn)
        print(f"Checkpointer initialized (using '{memory_file}')")
    except Exception as e:
        print(f"Failed to initialize checkpointer: {e}")
        return None, None

    # Build graph (ensure all node functions are defined above)
    builder = StateGraph(StorytellerAgentState)
    # Add Nodes (using correct function names)
    builder.add_node("load_data", load_manifest_and_languages_node)
    builder.add_node("determine_intent", determine_intent_llm_node)
    builder.add_node("handle_language", handle_language_change_node)
    builder.add_node("filter_stories", filter_stories_by_language_node)
    builder.add_node("select_story", select_story_node)
    builder.add_node("fetch_data", fetch_story_data_node)
    builder.add_node("format_response", format_response_node)
    
    # Define Edges
    builder.set_entry_point("load_data")
    builder.add_edge("load_data", "determine_intent")

    def route_after_intent(state):
        intent = state.get("user_request_type")
        error = state.get("error_message")
        if intent == "error_state" or (error and "Manifest" in error):
            return "format_response"
        if intent == "request_story":
            return "filter_stories"
        if intent in ["ask_language_options", "set_language"]:
            return "handle_language"
        return "format_response"

    builder.add_conditional_edges(
        "determine_intent",
        route_after_intent,
        {
            "filter_stories": "filter_stories",
            "handle_language": "handle_language",
            "format_response": "format_response",
        },
    )
    builder.add_edge("handle_language", "format_response")
    builder.add_edge("filter_stories", "select_story")
    builder.add_edge("select_story", "fetch_data")
    builder.add_edge("fetch_data", "format_response")
    builder.add_edge("format_response", END)

    try:
        agent = builder.compile(checkpointer=memory)
        print("--- Node-Based Audio+Text Storyteller Agent Compiled ---")
        return (
            agent,
            conn,
        )  # Return agent and connection (connection kept open by checkpointer)
    except Exception as e:
        print(f"--- ERROR Compiling Agent: {e} ---")
        if conn:
            conn.close()
        return None, None


storyteller_agent, db_connection = get_compiled_agent()

# --- Streamlit UI ---
sl.title("🎧 Storyteller Bot 📖")

if not storyteller_agent:
    sl.error(
        "Agent could not be compiled. Please check configuration (API Key, Dataset Path) and logs."
    )
    # Optionally provide more guidance based on LLM_READY flag etc.
    if not LLM_READY:
        sl.warning("LLM Initialization failed. Check API Key.")
    sl.stop()

sl.write("Select a language and ask for a story!")

    # --- Language Selection ---
available_languages = get_available_languages_from_manifest(MANIFEST_FULL_PATH)

# Initialize session state
if "thread_id" not in sl.session_state:
    sl.session_state.thread_id = f"sl_session_{uuid.uuid4()}"
    # Set initial language based on availability
    sl.session_state.selected_language = (
        DEFAULT_LANGUAGE
        if DEFAULT_LANGUAGE in available_languages
        else available_languages[0]
    )
    # sl.session_state.messages = []  # UI message history (simple list of dicts)
    # sl.session_state.last_response_text = None
    sl.session_state.audio_to_play = None
    sl.session_state.error_message = None
    sl.session_state.display_history = []  # For st.chat_message format
    sl.session_state.current_story_title = None # Track if a story is active
    print(f"Initialized Session State for thread: {sl.session_state.thread_id}")

# --- Output Area (Displays Chat and Audio) --- Moved UP
st_output_area = sl.container(height=500, border=False)  # Use container for grouping
with st_output_area:
    # Display chat messages using st.chat_message
    for msg in sl.session_state.display_history:
        with sl.chat_message(msg["role"]):
            # Use markdown to render potential formatting in agent responses
            sl.markdown(msg["content"])

    # Placeholder for the audio player
    audio_placeholder = sl.empty()
    # Display Audio Player if path is set in session state
    if sl.session_state.audio_to_play:
        audio_path = sl.session_state.audio_to_play
        if os.path.exists(audio_path):
            try:
                print(f"Playing audio: {audio_path}")
                audio_placeholder.audio(audio_path)
            except Exception as audio_e:
                sl.error(f"Streamlit error displaying audio: {audio_e}")
                sl.session_state.audio_to_play = None  # Clear on error
        else:
            sl.warning(f"Audio file path set but not found by Streamlit: {audio_path}")
            sl.session_state.audio_to_play = None  # Clear if invalid

# Display error if any occurred (maybe place below controls?)
if sl.session_state.error_message:
    sl.error(f"Agent Error: {sl.session_state.error_message}")
    sl.session_state.error_message = None  # Clear after showing

# --- Controls Area (Language and Buttons) --- Moved BELOW Output
st_controls_area = sl.container(border=True)
with st_controls_area:
    sl.write("**Controls**")  # Add a heading for clarity
    available_languages = get_available_languages_from_manifest(MANIFEST_FULL_PATH)

    selected_lang_widget = sl.selectbox(
        "Select Language:",
        options=available_languages,
        index=available_languages.index(sl.session_state.selected_language),
        key="lang_select_widget",
    )

    # Update agent state if language selection changes via UI
    if selected_lang_widget != sl.session_state.selected_language:
        sl.session_state.selected_language = selected_lang_widget
        sl.session_state.display_history.append(
            {
                "role": "system", 
                "content": f"Language changed to {selected_lang_widget}.",
            }
        )
        sl.session_state.audio_to_play = None
        sl.session_state.error_message = None
        sl.session_state.current_story_title = None
        audio_placeholder.empty()  # Clear audio player immediately
        print(f"UI Language selection changed to: {selected_lang_widget}")
        config = {"configurable": {"thread_id": sl.session_state.thread_id}}
        lang_set_input = {
            "messages": [
                HumanMessage(content=f"Set language to {selected_lang_widget}")
            ],
            "dataset_base_path": str(DATASET_PATH),
        }
        if LLM_READY:
            try:
                with sl.spinner(f"Switching language to {selected_lang_widget}..."):
                    sl_final_state = storyteller_agent.invoke(
                        lang_set_input, config=config
                    )
                if sl_final_state.get("messages"):
                    last_msg = sl_final_state["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        sl.session_state.display_history.append(
                            {"role": "assistant", "content": last_msg.content}
                        )
                sl.session_state.error_message = sl_final_state.get("error_message")
            except Exception as e:
                sl.error(f"Error updating language: {e}")
                sl.session_state.error_message = str(e)
        else:
            sl.warning("LLM not ready, cannot confirm language change.")
        sl.rerun()

    # --- Conditional Button Display ---
    col_btn_1, col_btn_2 = sl.columns(2)  # Place buttons side-by-side

    input_message = None  # Store the message for the agent
    print(f"Session State before button click: {sl.session_state}")
    with col_btn_1:
        # Show EITHER "Tell me a story" OR "Tell me another story"
        if not sl.session_state.current_story_title:  # Show if no story is active
            if sl.button(
                "Tell me a story", key="tell_story_btn", use_container_width=True
            ):
                input_message = "Tell me a story"
        else:  # Show after a story has been played
            if sl.button(
                "Tell me another story",
                key="tell_another_btn",
                use_container_width=True,
            ):
                input_message = "Tell me another story"

    with col_btn_2:
        # Show "Get AI Moral" button only if a story is active
        show_moral_button = sl.session_state.current_story_title is not None
        if sl.button(
            "Get AI Moral (Text)",
            key="get_moral_btn",
            disabled=not show_moral_button,
            use_container_width=True,
        ):
            # Construct a message asking for the moral of the specific story
            story_title_for_prompt = (
                sl.session_state.current_story_title or "the last story"
            )
            input_message = f"What is the moral of the story '{story_title_for_prompt}'?"  # Use this to trigger the correct intent/tool later if using tools
            # For the node-based version, we might need a specific intent type
            # Let's assume the LLM intent detector can handle this phrase

# --- Process Agent Invocation ---
if input_message:
    # Add user action to UI history
    display_user_action = f"(Action: {input_message})" 
    if input_message in ["Tell me a story", "Tell me another story"]:
        display_user_action = f"(Action: {input_message})"
    else:
        display_user_action = input_message
    sl.session_state.display_history.append({"role": "user", "content": display_user_action})

    # --- Clear previous outputs *before* invoking agent ---
    # Clear previous error message
    sl.session_state.error_message = None
    # Clear audio/title specifically when requesting a *new* story
    if input_message in ["Tell me a story", "Tell me another story"]:
        sl.session_state.audio_to_play = None
        sl.session_state.current_story_title = None # Clear title *before* new request
        audio_placeholder.empty()

    # Invoke Agent
    config = {"configurable": {"thread_id": sl.session_state.thread_id}}
    invoke_input = {
        "messages": [HumanMessage(content=input_message)],
        "selected_language": sl.session_state.selected_language,
        "dataset_base_path": str(DATASET_PATH),
    }
    try:
        with sl.spinner("Thinking..."):
            final_state = storyteller_agent.invoke(invoke_input, config=config)

        # Process results
        sl.session_state.error_message = final_state.get("error_message")
        agent_messages = final_state.get("messages", [])
        if agent_messages:
            last_ai_message = next(
                (m for m in reversed(agent_messages) if isinstance(m, AIMessage)), None
            )
            if last_ai_message:
                sl.session_state.display_history.append(
                    {"role": "assistant", "content": last_ai_message.content}
                )
            else: # Agent ended without final message (e.g. error before format)
                if not sl.session_state.error_message:  # Avoid duplicate error display
                    sl.session_state.display_history.append(
                        {
                            "role": "assistant",
                            "content": "(Agent finished without a text response)",
                        }
                    )
        else:
            sl.session_state.display_history.append(
                {"role": "assistant", "content": "(Agent did not return any messages)"}
            )
        # Get audio path signal and story title from final state
        sl.session_state.audio_to_play = final_state.get("output_audio_path")
        sl.session_state.current_story_title = final_state.get(
            "selected_story_title"
        )  # Update active story title
    except Exception as e:
        sl.error(f"An error occurred during agent invocation: {e}")
        sl.session_state.error_message = str(e)
        sl.session_state.display_history.append(
            {"role": "system", "content": f"Error: {e}"}
        )
    # Rerun to display the new message, potential audio, and update buttons
    sl.rerun()
