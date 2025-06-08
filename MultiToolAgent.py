
# MultiToolAgent.py
# A multi-tool LLM agent which can execute python code, search the web, summarize documents, and more.

import os
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv
import tempfile
import time
import logging
from functools import wraps

load_dotenv()

# Setup logging for retry mechanism
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure required API keys are set in environment (OpenAI, Tavily)
if os.getenv("OPENAI_API_KEY") is None:
    raise EnvironmentError("OPENAI_API_KEY not set in environment.")
if os.getenv("TAVILY_API_KEY") is None:
    raise EnvironmentError("TAVILY_API_KEY not set in environment.")

# Imports for Langchain, LangGraph, LangMem, and tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage
import requests
from bs4 import BeautifulSoup
import PyPDF2  
import gradio as gr
import io

# Initialize the language model (OpenAI GPT-4 via LangChain)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Retry mechanism and error handling utilities
def retry_with_backoff(max_retries=3, backoff_factor=1.5, exceptions=(Exception,)):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {e}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
            
            return f"RETRY_FAILED: {func.__name__} failed after {max_retries} attempts. Last error: {last_exception}"
        return wrapper
    return decorator

class PlanRevisionTracker:
    """Tracks failures and suggests plan revisions."""
    def __init__(self):
        self.failures = []
        self.revision_count = 0
        self.max_revisions = 3
    
    def add_failure(self, tool_name: str, error: str, context: str = ""):
        """Record a tool failure."""
        failure = {
            'tool': tool_name,
            'error': error,
            'context': context,
            'timestamp': time.time(),
            'revision_attempt': self.revision_count
        }
        self.failures.append(failure)
        logger.info(f"Recorded failure for {tool_name}: {error}")
    
    def should_revise_plan(self) -> bool:
        """Determine if plan should be revised based on failure patterns."""
        if self.revision_count >= self.max_revisions:
            return False
        
        recent_failures = [f for f in self.failures if time.time() - f['timestamp'] < 60]  # Last minute
        if len(recent_failures) >= 2:
            return True
        
        tool_failures = {}
        for failure in recent_failures:
            tool_failures[failure['tool']] = tool_failures.get(failure['tool'], 0) + 1
        
        return any(count >= 2 for count in tool_failures.values())
    
    def get_revision_suggestion(self) -> str:
        """Generate a plan revision suggestion based on failure patterns."""
        if not self.failures:
            return ""
        
        recent_failures = [f for f in self.failures if time.time() - f['timestamp'] < 120]
        tool_counts = {}
        error_patterns = []
        
        for failure in recent_failures:
            tool_counts[failure['tool']] = tool_counts.get(failure['tool'], 0) + 1
            error_patterns.append(failure['error'])
        
        suggestion = f"PLAN_REVISION_NEEDED (Attempt {self.revision_count + 1}/{self.max_revisions}): "
        
        if 'web_search' in tool_counts or 'TavilySearch' in str(tool_counts):
            suggestion += "Web search is failing - try using alternative information sources or simplify the query. "
        
        if 'execute_python' in tool_counts:
            suggestion += "Code execution is failing - try breaking down the code into smaller parts or using simpler logic. "
        
        if 'summarize_document' in tool_counts:
            suggestion += "Document processing is failing - check if the source is accessible or try a different approach. "
        
        if any('timeout' in error.lower() or 'connection' in error.lower() for error in error_patterns):
            suggestion += "Network issues detected - consider using cached information or alternative sources. "
        
        suggestion += "Consider: 1) Simplifying the approach, 2) Breaking task into smaller steps, 3) Using alternative tools."
        
        self.revision_count += 1
        return suggestion

plan_tracker = PlanRevisionTracker()

# Define the tools with retry mechanisms

# Calculator tool
@retry_with_backoff(max_retries=2, exceptions=(ValueError, SyntaxError, NameError))
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        if not expression or len(expression.strip()) == 0:
            plan_tracker.add_failure("calculate", "Empty expression provided", expression)
            return "Error: Empty expression provided"
        
        dangerous_ops = ['import', 'exec', 'eval', '__', 'open', 'file']
        if any(op in expression.lower() for op in dangerous_ops):
            plan_tracker.add_failure("calculate", "Dangerous operation detected", expression)
            return "Error: Expression contains potentially dangerous operations"
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        error_msg = f"Mathematical evaluation failed: {e}"
        plan_tracker.add_failure("calculate", str(e), expression)
        return f"Error: {error_msg}"

# Python code execution tool

@retry_with_backoff(max_retries=2, exceptions=(SyntaxError, NameError, ImportError))
def execute_python(code: str) -> str:
    """Execute a Python code snippet and return its output or the 'result' variable."""
    try:
        import io
        import sys
        import math
        import random
        import datetime
        from contextlib import redirect_stdout
        
        if not code or len(code.strip()) == 0:
            plan_tracker.add_failure("execute_python", "Empty code provided", code[:100])
            return "Error: No code provided to execute"
        
        dangerous_patterns = ['import os', 'import subprocess', 'open(', 'file(', 'exec(', '__import__']
        if any(pattern in code.lower() for pattern in dangerous_patterns):
            plan_tracker.add_failure("execute_python", "Dangerous operation detected", code[:100])
            return "Error: Code contains potentially dangerous operations"
        
        output_buffer = io.StringIO()
        local_vars: dict[str, Any] = {}
        
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'pow': pow,
                'divmod': divmod,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'any': any,
                'all': all,
                'filter': filter,
                'map': map,
            },
            'math': math,
            'random': random,
            'datetime': datetime,
        }
        
        with redirect_stdout(output_buffer):
            exec(code, safe_globals, local_vars)
        
        printed_output = output_buffer.getvalue()
        
        result_parts = []
        if printed_output.strip():
            result_parts.append(f"Output:\n{printed_output.strip()}")
        
        if "result" in local_vars:
            result_parts.append(f"Result: {local_vars['result']}")
        
        interesting_vars = {}
        for key, value in local_vars.items():
            if not key.startswith('_') and key not in ['math', 'random', 'datetime']:
                interesting_vars[key] = value
        
        if interesting_vars and not "result" in local_vars:
            var_output = []
            for key, value in interesting_vars.items():
                var_output.append(f"{key} = {value}")
            if var_output:
                result_parts.append(f"Variables:\n" + "\n".join(var_output))
        
        if result_parts:
            return "\n".join(result_parts)
        else:
            return "Code executed successfully."
            
    except Exception as e:
        error_msg = f"Python execution failed: {e}"
        plan_tracker.add_failure("execute_python", str(e), code[:100])
        return f"Error during execution: {error_msg}"

# Document summarization tool
@retry_with_backoff(max_retries=3, backoff_factor=2.0, exceptions=(requests.RequestException, ConnectionError, TimeoutError))
def summarize_document(source: str = "") -> str:
    """Summarize the content of a given URL, PDF file, or uploaded document. 
    If no source provided, will look for uploaded documents."""
    global uploaded_files
    
    text_content = ""
    source_description = ""
    
    try:
        # If no source provided, check for uploaded files
        if not source or source.lower() in ["uploaded", "document", "file", "pdf"]:
            if 'current' in uploaded_files:
                source = uploaded_files['current']
                source_description = f"uploaded document: {os.path.basename(source)}"
            else:
                error_msg = "No document found to summarize"
                plan_tracker.add_failure("summarize_document", error_msg, "no_source")
                return "No document found to summarize. Please upload a PDF file first or provide a URL."
        
        if source.lower().startswith("http"):
            source_description = f"URL: {source}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            try:
                resp = requests.get(source, timeout=15, headers=headers, allow_redirects=True)
                resp.raise_for_status()
            except requests.RequestException as e:
                plan_tracker.add_failure("summarize_document", f"Web request failed: {e}", source)
                return f"Error: Could not fetch content from {source}. Network error: {e}"
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.extract()
            
            text_content = soup.get_text()
            
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = ' '.join(chunk for chunk in chunks if chunk)
            
        elif source.lower().endswith(".pdf") or "uploaded_file:" in source:
            if "uploaded_file:" in source:
                file_path = source.replace("uploaded_file:", "")
            else:
                file_path = source
            
            if not source_description:
                source_description = f"PDF file: {os.path.basename(file_path)}"
            
            if not os.path.exists(file_path):
                error_msg = f"PDF file not found: {file_path}"
                plan_tracker.add_failure("summarize_document", error_msg, file_path)
                return f"Error: {error_msg}"
            
            try:
                reader = PyPDF2.PdfReader(file_path)
                for page in reader.pages:
                    if page.extract_text():
                        text_content += page.extract_text() + "\n"
            except Exception as e:
                plan_tracker.add_failure("summarize_document", f"PDF reading failed: {e}", file_path)
                return f"Error: Could not read PDF file. {e}"
        else:
            text_content = source
            source_description = "provided text"
    except Exception as e:
        error_msg = f"Document processing failed: {e}"
        plan_tracker.add_failure("summarize_document", str(e), source[:100] if source else "unknown")
        return f"Error: Could not retrieve content from {source} ({e})"
    
    if not text_content or len(text_content.strip()) < 10:
        error_msg = f"No meaningful content found from {source_description}"
        plan_tracker.add_failure("summarize_document", error_msg, source[:100] if source else "unknown")
        return f"Error: {error_msg}."
    
    if len(text_content) > 8000:
        text_content = text_content[:8000] + "..."
    
    summary_prompt = f"""Please provide a comprehensive summary of the following content from {source_description}. 
    Include the main points, key insights, and important details:

    Content:
    \"\"\"\n{text_content}\n\"\"\"
    
    Summary:"""
    
    try:
        summary = llm.predict(summary_prompt)
        return f"Summary of {source_description}:\n\n{summary.strip()}"
    except Exception as e:
        error_msg = f"LLM summarization failed: {e}"
        plan_tracker.add_failure("summarize_document", error_msg, source_description)
        return f"Error: Summarization failed ({e})"

# Enhanced Tavily search tool with retry mechanism
class RetryTavilySearch(TavilySearch):
    """Enhanced TavilySearch with retry mechanism."""
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0, exceptions=(Exception,))
    def _run(self, query: str) -> str:
        try:
            if not query or len(query.strip()) == 0:
                plan_tracker.add_failure("web_search", "Empty query provided", query)
                return "Error: Empty search query provided"
            
            result = super()._run(query)
            return result
        except Exception as e:
            plan_tracker.add_failure("web_search", str(e), query)
            raise e

tavily_search_tool = RetryTavilySearch(max_results=5, topic="general")

# Setup enhanced memory (LangMem) for different types of memories
store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})

# Create different memory namespaces for different types of information
personal_memory = create_manage_memory_tool(namespace=("personal_info",))
search_personal_memory = create_search_memory_tool(namespace=("personal_info",))

general_memory = create_manage_memory_tool(namespace=("general_knowledge",))
search_general_memory = create_search_memory_tool(namespace=("general_knowledge",))

task_memory = create_manage_memory_tool(namespace=("tasks_and_preferences",))
search_task_memory = create_search_memory_tool(namespace=("tasks_and_preferences",))


#System prompt
system_prompt = """You are an AI assistant equipped with powerful tools, memory capabilities, and autonomous error handling:

TOOLS AVAILABLE:
- WebSearch (TavilySearch): For finding current information online
- Calculator: For mathematical calculations  
- CodeRunner (execute_python): For running Python code
- DocumentSummarizer: For summarizing URLs, PDFs, and uploaded documents
- Memory Tools: For storing and retrieving information across conversations

AUTONOMOUS ERROR HANDLING & PLAN REVISION:
- You have built-in retry mechanisms for all tools (2-3 attempts with exponential backoff)
- If tools fail repeatedly, you receive PLAN_REVISION_NEEDED messages with specific suggestions
- When you see RETRY_FAILED or PLAN_REVISION_NEEDED, adapt your approach immediately:
  * Simplify complex requests into smaller steps
  * Use alternative tools (e.g., if web search fails, use your knowledge + mention limitations)
  * Break down multi-step tasks into individual components
  * If code execution fails, try simpler code or explain the concept instead
  * If document processing fails, ask user to check the source or try alternative formats

ERROR RECOVERY STRATEGIES:
1. **Web Search Failures**: Fall back to existing knowledge, ask for more specific queries, or suggest alternative sources
2. **Code Execution Failures**: Simplify code, break into smaller functions, or provide theoretical explanation
3. **Document Processing Failures**: Suggest alternative formats, check accessibility, or ask for manual text input
4. **Memory Tool Failures**: Use conversation context as backup, explain limitations
5. **Calculator Failures**: Validate expressions, suggest simpler forms, or break down complex calculations

CODE EXECUTION RULES:
- When users ask to "run", "execute", "test", or "check" code, ALWAYS use the execute_python tool
- When users ask for code and then say "can you execute this?" or "run this", use execute_python
- When users want to verify if something works (like "is 5 a perfect square?"), write code and execute it
- The execute_python tool has access to math, random, datetime modules
- Always show both the code and execution results
- If code fails, try simpler approaches or break into smaller parts

DOCUMENT HANDLING:
- When users upload documents (PDFs), they become available for analysis
- If user asks to "summarize", "analyze document", "what's in the document", or similar requests about documents, use the summarize_document tool
- You can call summarize_document() without parameters to automatically find uploaded documents
- You can also summarize specific URLs by passing them to summarize_document(url)
- If document processing fails, suggest alternative approaches or manual text input

MEMORY SYSTEM:
You have THREE types of memory namespaces:
1. PERSONAL INFO (personal_memory/search_personal_memory): Store user's personal details like name, age, DOB, preferences, background
2. GENERAL KNOWLEDGE (general_memory/search_general_memory): Store facts, definitions, concepts learned during conversations  
3. TASKS & PREFERENCES (task_memory/search_task_memory): Store user's work preferences, frequent tasks, coding patterns

MEMORY USAGE RULES:
- ALWAYS search relevant memory namespaces before responding to user questions
- When user shares personal information (name, age, location, preferences), immediately store it in personal_memory
- When learning new facts or concepts, store them in general_memory  
- When user shows task patterns or preferences, store them in task_memory
- Use descriptive memory keys like "user_name", "user_dob", "favorite_programming_language"
- PERSIST MEMORIES across conversation clears - they should survive session resets

CONVERSATION BEHAVIOR:
- Always search your memories first before asking the user to repeat information
- When a user says 'yes' or confirms something, refer back to previous context
- Break complex tasks into clear steps
- Use tools proactively to provide comprehensive answers
- Maintain conversation context while leveraging long-term memory
- When users mention documents, summarization, or analysis, check if they have uploaded files
- When users ask about code execution or want to test something, use execute_python tool
- If you receive error messages about failed retries or need for plan revision, adapt your approach immediately
- Be transparent about failures and explain alternative approaches

EXAMPLE WORKFLOWS:
Memory: "My name is John and I was born on Jan 15, 1990" → Store in personal_memory
Document: "Summarize the document" or "What's in the PDF?" → Use summarize_document()
URL: "Summarize https://example.com" → Use summarize_document("https://example.com")
Code: "Can you execute this code?" or "Is 5 a perfect square?" → Use execute_python tool
Math: "What's 2+2?" → Use calculate tool for simple math, execute_python for complex operations
Error Recovery: If tools fail, simplify approach, use alternatives, or break into smaller steps

CRITICAL: When you see messages starting with "RETRY_FAILED:" or "PLAN_REVISION_NEEDED:", immediately adapt your strategy based on the suggestions provided. Be transparent with users about failures and explain your alternative approach.

Always be proactive with memory, document handling, code execution, and error recovery!"""

# Create the agent with all components
tools_list = [
    tavily_search_tool, 
    calculate, 
    execute_python, 
    summarize_document,
    personal_memory,
    search_personal_memory,
    general_memory, 
    search_general_memory,
    task_memory,
    search_task_memory
]
tool_node = ToolNode(tools_list, handle_tool_errors=True)
agent = create_react_agent(
    model=llm,
    tools=tool_node,
    store=store,
    prompt=system_prompt
)

# Global conversation state (short-term memory)
conversation_messages = []

# Store uploaded file paths
uploaded_files = {}

# Gradio UI setup
with gr.Blocks(title="Multi-Tool AI Agent with Memory", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Multi-Tool AI Agent")
    gr.Markdown("This agent can summarize documents, execute code, and answer questions using the web, use calculator for math, and use memory to remember information.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chat_history = gr.State([])
            chatbot = gr.Chatbot(
                label="AI Agent Chat", 
                height=400,
                show_label=True,
                container=True
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    show_label=False, 
                    placeholder="Type your message here...",
                    scale=4
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("Clear Conversation", variant="secondary")
                memory_demo_btn = gr.Button("Demo Memory", variant="outline")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Document Upload")
            file_upload = gr.File(
                label="Upload Document (PDF)",
                file_types=[".pdf"],
                type="filepath"
            )
            file_status = gr.Textbox(
                label="Upload Status",
                interactive=False,
                value="No file uploaded"
            )
            
            gr.Markdown("### 🧠 Memory Demo")
            gr.Markdown("""
            **Try these commands to test memory:**
            1. "My name is [Your Name] and I was born on [Date]"
            2. Click "Clear Conversation"  
            3. "What's my name and when was I born?"
            
            The agent should remember your info even after clearing!
            """)
            
            gr.Markdown("### 📄 Document Analysis")
            gr.Markdown("""
            **After uploading a PDF:**
            - "Summarize the document"
            - "What's in the PDF?" 
            - "Analyze the uploaded file"
            - "Give me the main points from the document"
            
            **For web content:**
            - "Summarize https://example.com"
            """)
            
            gr.Markdown("### 🐍 Code Execution Demo")
            gr.Markdown("""
            **Try these Python commands:**
            - "Can you check if 25 is a perfect square?"
            - "Execute: print('Hello World!')"
            - "Write code to find prime numbers up to 20 and run it"
            - "Calculate the factorial of 5 using Python"
            """)
            
                        
            gr.Markdown("### 🔧 Available Tools")
            gr.Markdown("""
            - **Web Search**: Current information lookup (with retry)
            - **Calculator**: Mathematical computations (with validation)
            - **Python Code**: Execute code snippets with math, random, datetime (with retry)
            - **Document Summarizer**: Analyze PDFs/URLs (with retry and fallbacks)
            - **Persistent Memory**: Remember across sessions (with backup strategies)
            """)
    
    # Agent response function
    def agent_respond(user_message, history):
        global conversation_messages
        
        if history is None:
            history = []
        
        conversation_messages.append(HumanMessage(content=user_message))
        
        try:
            if plan_tracker.should_revise_plan():
                revision_suggestion = plan_tracker.get_revision_suggestion()
                revision_message = HumanMessage(content=f"SYSTEM_NOTICE: {revision_suggestion}")
                conversation_messages.append(revision_message)
                logger.info(f"Plan revision triggered: {revision_suggestion}")
            
            response = agent.invoke({"messages": conversation_messages})
            answer = response["messages"][-1].content
            
            if "RETRY_FAILED:" in answer or "PLAN_REVISION_NEEDED:" in answer:
                logger.warning("Agent received failure indicators in response")
            
            conversation_messages.append(AIMessage(content=answer))
            
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(error_msg)
            plan_tracker.add_failure("agent", str(e), user_message[:100])
            
            answer = f"I encountered an error while processing your request: {error_msg}. Let me try a simpler approach or different method."
            conversation_messages.append(AIMessage(content=answer))
        
        history.append((user_message, answer))
        return history, history, ""
    
    def clear_conversation():
        global conversation_messages, plan_tracker
        conversation_messages = []  
        plan_tracker = PlanRevisionTracker()  
        return [], [], "Conversation cleared (long-term memory preserved, error tracking reset)"
    
    def demo_memory():
        return [], [], "Try: 'My name is John, I'm 25 years old and I love Python programming'"
    
    def handle_file_upload(file_path):
        if file_path is None:
            uploaded_files.pop('current', None)
            return "No file uploaded"
        
        uploaded_files['current'] = file_path
        file_name = os.path.basename(file_path)
        return f"✅ File uploaded: {file_name}\nNow you can ask me to 'summarize the document' in the chat!"
    
    send_btn.click(
        agent_respond,
        inputs=[user_input, chat_history],
        outputs=[chatbot, chat_history, user_input]
    )
    
    user_input.submit(
        agent_respond,
        inputs=[user_input, chat_history], 
        outputs=[chatbot, chat_history, user_input]
    )
    
    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, chat_history, user_input]
    )
    
    memory_demo_btn.click(
        demo_memory,
        outputs=[chatbot, chat_history, user_input]
    )
    
    file_upload.change(
        handle_file_upload,
        inputs=[file_upload],
        outputs=[file_status]
    )

# Launch the Gradio demo
if __name__ == "__main__":
    demo.launch(share=False, debug=True)

# Evaluation with memory testing and error handling
def run_evaluation():
    global conversation_messages, plan_tracker
    conversation_messages = []
    plan_tracker = PlanRevisionTracker()
    
    test_queries = [
        "My name is Alice and I was born on March 10, 1995. I'm a software engineer.",
        "Remember this fact: Python was created by Guido van Rossum.",
        "I prefer working on machine learning projects using PyTorch.",
        "What's my name and profession?",  
        "Who created Python?",  
        "What are my technical preferences?",  
        "Calculate the square root of 144",
        "Write Python code to find prime numbers up to 20 and execute it",
        "Can you check if 25 is a perfect square using Python code?",  
        "Execute this code: print('Hello, World!')",  
        "Try to summarize this invalid URL: https://this-url-does-not-exist-12345.com",  
        "Execute this broken code: print(undefined_variable)", 
    ]
    
    print("----- Enhanced Benchmark Evaluation with Error Handling -----")
    for i, q in enumerate(test_queries):
        print(f"\n{i+1}. User Query: {q}")
        conversation_messages.append(HumanMessage(content=q))
        try:
            result = agent.invoke({"messages": conversation_messages})
            answer = result["messages"][-1].content
            conversation_messages.append(AIMessage(content=answer))
            print(f"Agent Answer: {answer}")
            
            
            if i == 2:  
                print("\n--- SIMULATING CONVERSATION CLEAR ---")
                conversation_messages = []  # Clear short-term memory
                
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n--- ERROR HANDLING STATISTICS ---")
    print(f"Total failures recorded: {len(plan_tracker.failures)}")
    print(f"Plan revisions attempted: {plan_tracker.revision_count}")
    for failure in plan_tracker.failures:
        print(f"- {failure['tool']}: {failure['error'][:50]}...")

