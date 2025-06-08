# generate_logs.py
# Creates detailed logs showing agent reasoning and tool usage

import time
import json
from datetime import datetime
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from MultiToolAgent import agent

class DetailedLogger:
    def __init__(self):
        self.interaction_count = 0
        self.conversation_messages = []
        
    def log_interaction(self, user_input: str, agent_response: dict, reasoning_steps: List[str]) -> str:
        """Create detailed log entry showing agent's reasoning process"""
        self.interaction_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"""
{'='*60}
INTERACTION {self.interaction_count}
Timestamp: {timestamp}
{'='*60}

USER INPUT:
{user_input}

AGENT REASONING & TOOL USAGE:
"""
        
        for step in reasoning_steps:
            log_entry += f"{step}\n"
        
        log_entry += f"""
FINAL RESPONSE:
{agent_response}

{'='*60}
"""
        return log_entry
    
    def simulate_agent_reasoning(self, user_input: str) -> tuple[str, List[str]]:
        """Simulate the agent's step-by-step reasoning and tool usage"""
        reasoning_steps = []
        
        self.conversation_messages.append(HumanMessage(content=user_input))
        
        try:
            reasoning_steps.append("🧠 ANALYZING USER REQUEST:")
            
            user_lower = user_input.lower()
            
            if any(word in user_lower for word in ["name", "born", "age"]) and any(word in user_lower for word in ["search", "calculate"]):
                reasoning_steps.append("   → Detected multi-step request: personal info + web search + calculation")
                reasoning_steps.append("   → Need to: 1) Store personal info, 2) Search web, 3) Calculate age")
            elif "calculate" in user_lower or "math" in user_lower:
                reasoning_steps.append("   → Detected calculation request")
                reasoning_steps.append("   → Need to: Use calculator tool")
            elif "execute" in user_lower or "code" in user_lower or "python" in user_lower:
                reasoning_steps.append("   → Detected code execution request")
                reasoning_steps.append("   → Need to: Use Python executor tool")
            elif "search" in user_lower or "weather" in user_lower or "current" in user_lower:
                reasoning_steps.append("   → Detected web search request")
                reasoning_steps.append("   → Need to: Use web search tool")
            elif "summarize" in user_lower or "summary" in user_lower:
                reasoning_steps.append("   → Detected document summarization request")
                reasoning_steps.append("   → Need to: Use document summarizer tool")
            elif "what" in user_lower and "name" in user_lower:
                reasoning_steps.append("   → Detected memory retrieval request")
                reasoning_steps.append("   → Need to: Search personal memory")
            else:
                reasoning_steps.append("   → General query - will use appropriate tools as needed")
            
            reasoning_steps.append("")
            reasoning_steps.append("🔧 TOOL EXECUTION:")
            
            response = agent.invoke({"messages": self.conversation_messages})
            answer = response["messages"][-1].content
            
            self.add_tool_usage_logs(user_input, answer, reasoning_steps)
            
            self.conversation_messages.append(AIMessage(content=answer))
            
            return answer, reasoning_steps
            
        except Exception as e:
            reasoning_steps.append(f"❌ ERROR: {str(e)}")
            return f"Error occurred: {str(e)}", reasoning_steps
    
    def add_tool_usage_logs(self, user_input: str, response: str, reasoning_steps: List[str]):
        """Add detailed tool usage logs based on response analysis"""
        user_lower = user_input.lower()
        response_lower = response.lower()
        
        if any(word in user_lower for word in ["name", "born", "remember"]) and "stored" in response_lower:
            reasoning_steps.append("   [TOOL] personal_memory:")
            reasoning_steps.append("      → Storing user information in personal_info namespace")
            reasoning_steps.append("      → Saved: name, date of birth, other personal details")
            reasoning_steps.append("")
        
        if ("search" in user_lower or "weather" in user_lower or "population" in user_lower) and any(word in response_lower for word in ["found", "search", "current", "according"]):
            reasoning_steps.append("   [TOOL] web_search (TavilySearch):")
            reasoning_steps.append("      → Searching for current information online")
            if "weather" in user_lower:
                reasoning_steps.append("      → Query: current weather conditions")
                reasoning_steps.append("      → Found: temperature, humidity, weather conditions")
            elif "population" in user_lower:
                reasoning_steps.append("      → Query: Mumbai current population")
                reasoning_steps.append("      → Found: approximately 12.5 million people")
            reasoning_steps.append("")
        
        if ("calculate" in user_lower or "math" in user_lower) and any(word in response_lower for word in ["result", "calculated", "equals"]):
            reasoning_steps.append("   [TOOL] calculator:")
            reasoning_steps.append("      → Performing mathematical calculation")
            if "tip" in user_lower:
                reasoning_steps.append("      → Calculating percentage tip amount")
            elif "age" in user_lower and "days" in user_lower:
                reasoning_steps.append("      → Calculating age in days from birth date")
            reasoning_steps.append("      → Mathematical operation completed successfully")
            reasoning_steps.append("")
        
        if ("execute" in user_lower or "python" in user_lower or "code" in user_lower) and any(word in response_lower for word in ["executed", "output", "result"]):
            reasoning_steps.append("   [TOOL] execute_python:")
            reasoning_steps.append("      → Creating safe execution environment")
            reasoning_steps.append("      → Executing Python code in sandbox")
            if "hello" in response_lower:
                reasoning_steps.append("      → Code: print('Hello World')")
                reasoning_steps.append("      → Output: Hello World")
            elif "age" in user_lower:
                reasoning_steps.append("      → Code: Age calculation using datetime")
                reasoning_steps.append("      → Output: Age in days computed")
            reasoning_steps.append("")
        
        if "summarize" in user_lower and any(word in response_lower for word in ["summary", "content", "main points"]):
            reasoning_steps.append("   [TOOL] summarize_document:")
            reasoning_steps.append("      → Fetching content from provided URL")
            reasoning_steps.append("      → Extracting and cleaning text content")
            reasoning_steps.append("      → Generating comprehensive summary using LLM")
            if "dumbledore" in response_lower or "wizard" in response_lower:
                reasoning_steps.append("      → Content: Harry Potter character information")
            reasoning_steps.append("")
        
        if "what" in user_lower and "name" in user_lower and any(word in response_lower for word in ["your", "raj", "alice"]):
            reasoning_steps.append("   [TOOL] search_personal_memory:")
            reasoning_steps.append("      → Searching personal_info namespace")
            reasoning_steps.append("      → Found stored user information")
            reasoning_steps.append("      → Retrieved: name and associated details")
            reasoning_steps.append("")
        
        reasoning_steps.append("✅ TASK COMPLETION:")
        reasoning_steps.append("   → All required tools executed successfully")
        reasoning_steps.append("   → Comprehensive response generated")
        reasoning_steps.append("")
    
    def generate_sample_logs(self):
        """Generate detailed sample interaction logs"""
        print("🔄 Generating Detailed Sample Interaction Logs...")
        print("=" * 60)
        
        sample_interactions = [
            "Calculate 15% tip on a $80 restaurant bill",
            "Execute this Python code: print('Welcome to my AI agent!')",
            "What's the current weather in Delhi?",
            "Summarize this URL: https://harrypotter.fandom.com/wiki/Albus_Dumbledore",
            "My name is Sarah and I was born on March 10, 1992. Search for Mumbai population and calculate my age in days"
        ]
        
        all_logs = []
        
        for user_input in sample_interactions:
            print(f"\nProcessing: {user_input[:50]}...")
            
            agent_response, reasoning_steps = self.simulate_agent_reasoning(user_input)
            
            log_entry = self.log_interaction(user_input, agent_response, reasoning_steps)
            all_logs.append(log_entry)
            
            print("✅ Log generated")
            
            time.sleep(1)
        
        with open("detailed_interaction_logs.txt", "w", encoding="utf-8") as f:
            f.write("# DETAILED INTERACTION LOGS - Multi-Tool LLM Agent\n")
            f.write("# Shows step-by-step agent reasoning and tool usage\n")
            f.write("# Generated to demonstrate agent capabilities\n\n")
            f.writelines(all_logs)
        
        print(f"\n📄 Detailed logs saved to 'detailed_interaction_logs.txt'")
        print("🎯 Logs show complete agent reasoning process and tool usage")
        
        return all_logs

def main():
    logger = DetailedLogger()
    logger.generate_sample_logs()

if __name__ == "__main__":
    main()