# evaluation_suite.py
# Simple evaluation suite for MultiToolAgent with easy tasks

import time
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage
from MultiToolAgent import agent, conversation_messages, plan_tracker

class SimpleEvaluator:
    def __init__(self):
        self.results = []
        self.test_messages = []
        
    def evaluate_task_success(self, query: str, response: str) -> float:
        query_lower = query.lower()
        response_lower = response.lower()
        
        if "calculate" in query_lower or "math" in query_lower:
            if any(word in response_lower for word in ["result", "answer", "equals", "=", "calculated", "110"]):
                return 1.0
        
        if "execute" in query_lower or "run" in query_lower or "code" in query_lower:
            if any(word in response_lower for word in ["output", "executed", "result", "hello"]):
                return 1.0
                
        if "search" in query_lower or "current" in query_lower or "weather" in query_lower:
            if any(word in response_lower for word in ["found", "search", "current", "weather", "temperature"]):
                return 1.0
                
        if "summarize" in query_lower:
            if any(word in response_lower for word in ["summary", "dumbledore", "wizard", "hogwarts"]):
                return 1.0
        
        return 0.0
    
    def evaluate_tool_use(self, query: str, response: str) -> float:
        query_lower = query.lower()
        response_lower = response.lower()
        
        if "calculate" in query_lower:
            if any(word in response_lower for word in ["calculated", "result", "equals"]):
                return 1.0
                
        if "execute" in query_lower or "run" in query_lower:
            if any(word in response_lower for word in ["executed", "output", "code"]):
                return 1.0
                
        if "search" in query_lower or "weather" in query_lower:
            if any(word in response_lower for word in ["search", "found", "according"]):
                return 1.0
                
        if "summarize" in query_lower:
            if any(word in response_lower for word in ["summary", "summarize"]):
                return 1.0
        
        return 0.0
    
    def evaluate_coherence(self, query: str, response: str) -> float:
        if len(response.strip()) < 10:
            return 0.0
        
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        relevance = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
        
        if "error" in response.lower() and "successfully" not in response.lower():
            return max(0.3, relevance)
        
        return min(1.0, 0.5 + relevance)
    
    def evaluate_reasoning(self, query: str, response: str) -> float:
        reasoning_words = ["let me", "i will", "first", "then", "because", "so", "here"]
        
        response_lower = response.lower()
        reasoning_count = sum(1 for word in reasoning_words if word in response_lower)
        
        score = min(1.0, reasoning_count / 2.0)
        
        if len(response.split()) > 20:
            score += 0.3
            
        return min(1.0, score)
    
    def run_evaluation(self):
        print("🧪 Simple MultiToolAgent Evaluation")
        print("="*50)
        
        test_cases = [
            {
                "name": "Basic Calculator",
                "query": "Calculate 25 * 4 + 10"
            },
            {
                "name": "Simple Code Execution", 
                "query": "Execute this Python code: print('Hello World')"
            },
            {
                "name": "Weather Search",
                "query": "Search for current weather in Delhi"
            },
            {
                "name": "Document Summarization",
                "query": "Summarize this URL: https://harrypotter.fandom.com/wiki/Albus_Dumbledore"
            }
        ]
        
        for i, test in enumerate(test_cases):
            print(f"\n🔄 Test {i+1}: {test['name']}")
            print(f"Query: {test['query']}")
            
            start_time = time.time()
            
            try:
                self.test_messages.append(HumanMessage(content=test['query']))
                response = agent.invoke({"messages": self.test_messages})
                answer = response["messages"][-1].content
                self.test_messages.append(AIMessage(content=answer))
                
            except Exception as e:
                answer = f"Error: {str(e)}"
            
            end_time = time.time()
            
            task_success = self.evaluate_task_success(test['query'], answer)
            tool_use = self.evaluate_tool_use(test['query'], answer)
            coherence = self.evaluate_coherence(test['query'], answer) 
            reasoning = self.evaluate_reasoning(test['query'], answer)
            
            result = {
                "test_name": test['name'],
                "query": test['query'],
                "response": answer,
                "metrics": {
                    "task_success": task_success,
                    "tool_use": tool_use,
                    "coherence": coherence,
                    "reasoning": reasoning
                },
                "response_time": end_time - start_time
            }
            
            self.results.append(result)
            
            print(f"✅ Task Success: {task_success:.2f}/1.0")
            print(f"✅ Tool Use: {tool_use:.2f}/1.0") 
            print(f"✅ Coherence: {coherence:.2f}/1.0")
            print(f"✅ Reasoning: {reasoning:.2f}/1.0")
            print(f"⏱️  Time: {end_time - start_time:.2f}s")
        
        self.analyze_results()
    
    def analyze_results(self):
        print(f"\n📊 EVALUATION ANALYSIS")
        print("="*50)
        
        avg_task_success = sum(r["metrics"]["task_success"] for r in self.results) / len(self.results)
        avg_tool_use = sum(r["metrics"]["tool_use"] for r in self.results) / len(self.results)
        avg_coherence = sum(r["metrics"]["coherence"] for r in self.results) / len(self.results)
        avg_reasoning = sum(r["metrics"]["reasoning"] for r in self.results) / len(self.results)
        avg_time = sum(r["response_time"] for r in self.results) / len(self.results)
        
        overall_score = (avg_task_success + avg_tool_use + avg_coherence + avg_reasoning) / 4
        
        print(f"📈 OVERALL METRICS:")
        print(f"   Overall Score: {overall_score:.3f}/1.000 ({overall_score*100:.1f}%)")
        print(f"   Task Success: {avg_task_success:.3f}/1.000 ({avg_task_success*100:.1f}%)")
        print(f"   Tool Use: {avg_tool_use:.3f}/1.000 ({avg_tool_use*100:.1f}%)")
        print(f"   Coherence: {avg_coherence:.3f}/1.000 ({avg_coherence*100:.1f}%)")
        print(f"   Reasoning: {avg_reasoning:.3f}/1.000 ({avg_reasoning*100:.1f}%)")
        print(f"   Avg Response Time: {avg_time:.2f} seconds")
        
        print(f"\n🎯 DETAILED RESULTS:")
        for i, result in enumerate(self.results):
            metrics = result["metrics"]
            print(f"Test {i+1}: {result['test_name']}")
            print(f"   Success: {metrics['task_success']:.2f} | Tools: {metrics['tool_use']:.2f} | Coherence: {metrics['coherence']:.2f} | Reasoning: {metrics['reasoning']:.2f}")
        
        print(f"\n🔍 ANALYSIS:")
        
        if overall_score >= 0.8:
            print("   ✅ EXCELLENT: Agent performs very well across all basic tasks")
        elif overall_score >= 0.7:
            print("   ✅ GOOD: Agent handles most basic tasks well")
        elif overall_score >= 0.6:
            print("   ⚠️  FAIR: Agent completes some tasks but needs improvement")
        else:
            print("   ❌ NEEDS WORK: Agent struggles with basic tasks")
        
        strengths = []
        weaknesses = []
        
        if avg_task_success >= 0.7:
            strengths.append("Task completion")
        else:
            weaknesses.append("Task completion")
            
        if avg_tool_use >= 0.7:
            strengths.append("Tool selection")
        else:
            weaknesses.append("Tool selection")
            
        if avg_coherence >= 0.7:
            strengths.append("Response coherence")
        else:
            weaknesses.append("Response coherence")
            
        if avg_reasoning >= 0.7:
            strengths.append("Basic reasoning")
        else:
            weaknesses.append("Basic reasoning")
        
        if strengths:
            print(f"   💪 Strengths: {', '.join(strengths)}")
        if weaknesses:
            print(f"   🎯 Improvement Areas: {', '.join(weaknesses)}")

if __name__ == "__main__":
    evaluator = SimpleEvaluator()
    evaluator.run_evaluation()