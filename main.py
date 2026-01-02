import os
import json
import operator
import time
import ssl
import nltk
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Optional

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Sentiment and text analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Pydantic for data validation
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Download required NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# --- Data Models ---

class InterviewState(TypedDict):
    """Shared state across all interview agents"""
    candidate_name: str
    position: str
    experience_level: str  # junior, mid, senior

    # Conversation history
    messages: Annotated[List[Dict], operator.add]

    # Agent-specific data
    hr_questions: List[str]
    technical_questions: List[str]
    behavioral_questions: List[str]

    hr_responses: List[Dict]
    technical_responses: List[Dict]
    behavioral_responses: List[Dict]

    # Analysis metrics
    stress_level: float  # 0-1 scale
    confidence_score: float  # 0-1 scale
    technical_competency: float  # 0-10 scale
    communication_score: float  # 0-10 scale
    cultural_fit_score: float  # 0-10 scale

    # Dynamic difficulty
    current_difficulty: str  # easy, medium, hard
    questions_asked: int
    correct_answers: int

    # Bias detection
    bias_flags: List[str]

    # Final evaluation
    overall_score: float
    recommendation: str
    detailed_feedback: str

    # Control flow
    current_agent: str
    interview_complete: bool
    has_errors: bool

class QuestionGeneration(BaseModel):
    """Model for generating interview questions"""
    questions: List[str] = Field(description="List of interview questions")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard")
    follow_up_topics: List[str] = Field(description="Potential follow-up topics")

class ResponseEvaluation(BaseModel):
    """Model for evaluating candidate responses"""
    score: float = Field(description="Score from 0-10")
    strengths: List[str] = Field(description="Identified strengths")
    weaknesses: List[str] = Field(description="Areas for improvement")
    confidence_level: str = Field(description="Confidence level: low, medium, high")
    needs_follow_up: bool = Field(description="Whether follow-up is needed")

class FinalEvaluation(BaseModel):
    """Model for final candidate evaluation"""
    overall_score: float = Field(description="Overall score from 0-10")
    recommendation: str = Field(description="hire, maybe, or no_hire")
    technical_rating: float
    behavioral_rating: float
    cultural_fit_rating: float
    key_strengths: List[str]
    key_concerns: List[str]
    detailed_feedback: str

# --- Utilities ---

def get_llm(max_tokens=1000, temperature=0.7):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
    return ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60.0,
        max_retries=2
    )

def safe_llm_call(prompt: str, max_retries: int = 3, max_tokens: int = 2000) -> Optional[str]:
    """Make LLM call with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            llm = get_llm(max_tokens=max_tokens)
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
    return None

def extract_json_safely(text: str) -> Optional[dict]:
    """Extract JSON from LLM response with error handling"""
    if not text: return None
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return None

class ResponseAnalyzer:
    """Advanced analysis utilities for candidate responses"""
    @staticmethod
    def detect_stress_level(text: str) -> float:
        stress_indicators = {
            'high': ['um', 'uh', 'err', 'well...', 'i guess', 'maybe', 'not sure', 'difficult', 'hard', 'struggle', 'confused'],
            'medium': ['think', 'believe', 'suppose', 'probably', 'might'],
            'low': ['confident', 'certain', 'definitely', 'absolutely', 'sure']
        }
        text_lower = text.lower()
        high_count = sum(text_lower.count(word) for word in stress_indicators['high'])
        medium_count = sum(text_lower.count(word) for word in stress_indicators['medium'])
        low_count = sum(text_lower.count(word) for word in stress_indicators['low'])
        stress_score = (high_count * 0.8 + medium_count * 0.4 - low_count * 0.3)
        stress_score = max(0, min(1, stress_score / 5))
        sentiment = sentiment_analyzer.polarity_scores(text)
        if sentiment['compound'] < -0.3: stress_score += 0.2
        return min(1.0, stress_score)

    @staticmethod
    def detect_confidence(text: str) -> float:
        confidence_markers = {
            'high': ['definitely', 'certainly', 'confident', 'absolutely', 'without doubt', 'clearly', 'obviously', 'exactly'],
            'low': ['maybe', 'perhaps', 'might', 'could be', 'not sure', 'i think', 'i guess', 'possibly', 'probably']
        }
        text_lower = text.lower()
        high_count = sum(text_lower.count(word) for word in confidence_markers['high'])
        low_count = sum(text_lower.count(word) for word in confidence_markers['low'])
        word_count = len(text.split())
        length_factor = min(1.0, word_count / 50)
        confidence = (high_count * 0.2 - low_count * 0.15 + length_factor * 0.5)
        return max(0, min(1, confidence))

    @staticmethod
    def detect_bias(question: str, context: str) -> List[str]:
        bias_patterns = {
            'age': ['young', 'old', 'millennial', 'generation', 'age'],
            'gender': ['he', 'she', 'masculine', 'feminine', 'guys', 'girls'],
            'cultural': ['native', 'foreign', 'accent', 'background', 'where are you from'],
            'personal': ['married', 'children', 'family', 'pregnant', 'religion'],
            'disability': ['health', 'disability', 'medical', 'condition']
        }
        detected_bias = []
        text_lower = (question + " " + context).lower()
        for bias_type, keywords in bias_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_bias.append(bias_type)
        return detected_bias

    @staticmethod
    def analyze_communication_quality(text: str) -> float:
        blob = TextBlob(text)
        word_count = len(text.split())
        sentence_count = len(blob.sentences)
        length_score = min(10, (word_count / 20)) if word_count < 200 else 10 - (word_count - 200) / 50
        length_score = max(0, min(10, length_score))
        structure_score = 10 if 2 <= sentence_count <= 4 else max(0, 10 - abs(sentence_count - 3) * 2)
        sentiment = sentiment_analyzer.polarity_scores(text)
        sentiment_score = (sentiment['compound'] + 1) * 5
        communication_score = (length_score * 0.4 + structure_score * 0.3 + sentiment_score * 0.3)
        return round(communication_score, 2)

analyzer = ResponseAnalyzer()

# --- Agents ---

def hr_agent(state: InterviewState) -> InterviewState:
    print("\n" + "="*60 + "\n👔 HR AGENT ACTIVE\n" + "="*60)
    hr_prompt = f"""You are an experienced HR interviewer for a {state['position']} position.
Candidate: {state['candidate_name']}, Level: {state['experience_level']}
Generate {3 if state.get('questions_asked', 0) == 0 else 2} HR questions assessing cultural fit, communication, and motivation.
Return ONLY a JSON object: {{"questions": [], "difficulty": "medium", "follow_up_topics": []}}"""
    
    try:
        response_text = safe_llm_call(hr_prompt)
        questions_data = extract_json_safely(response_text)
        questions = questions_data['questions']
        state['hr_questions'] = questions
        
        hr_responses = []
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            answer = input("Your answer: ").strip()
            
            stress = analyzer.detect_stress_level(answer)
            confidence = analyzer.detect_confidence(answer)
            comm = analyzer.analyze_communication_quality(answer)
            bias = analyzer.detect_bias(question, answer)
            
            hr_responses.append({
                'question': question, 'answer': answer, 'stress_level': stress,
                'confidence': confidence, 'communication_score': comm, 'timestamp': datetime.now().isoformat()
            })
            
            q_count = state.get('questions_asked', 0)
            state['stress_level'] = (state.get('stress_level', 0) * q_count + stress) / (q_count + 1)
            state['confidence_score'] = (state.get('confidence_score', 0) * q_count + confidence) / (q_count + 1)
            state['communication_score'] = (state.get('communication_score', 0) * q_count + comm) / (q_count + 1)
            state['questions_asked'] = q_count + 1
            if bias: state['bias_flags'].extend([f"HR-Q{i}: {b}" for b in bias])
            print(f"   📊 Stress: {stress:.2f} | Confidence: {confidence:.2f} | Communication: {comm:.2f}")

        state['hr_responses'] = hr_responses
        eval_prompt = f"Evaluate cultural fit (0-10) based on: {json.dumps(hr_responses)}"
        eval_res = safe_llm_call(eval_prompt, max_tokens=100)
        import re
        score_match = re.search(r'(\d+(\.\d+)?)', eval_res or "")
        state['cultural_fit_score'] = float(score_match.group(1)) if score_match else 7.0
        
    except Exception as e:
        print(f"❌ HR Agent Error: {e}")
        state['has_errors'] = True

    state['current_agent'] = 'technical'
    return state

def technical_agent(state: InterviewState) -> InterviewState:
    print("\n" + "="*60 + "\n💻 TECHNICAL AGENT ACTIVE\n" + "="*60)
    perf = state.get('correct_answers', 0) / max(1, state.get('questions_asked', 1))
    if perf > 0.8: state['current_difficulty'] = 'hard'
    elif perf < 0.4: state['current_difficulty'] = 'medium'

    tech_prompt = f"""Senior technical interviewer for {state['position']}.
Generate 3 technical questions for {state['experience_level']} level. Difficulty: {state['current_difficulty']}.
Return ONLY JSON: {{"questions": [], "difficulty": "...", "follow_up_topics": []}}"""
    
    try:
        response_text = safe_llm_call(tech_prompt)
        questions_data = extract_json_safely(response_text)
        questions = questions_data['questions']
        state['technical_questions'] = questions
        
        tech_responses = []
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            answer = input("Your answer: ").strip()
            
            eval_prompt = f"Evaluate technical answer (0-10). Question: {question}, Answer: {answer}. Return JSON: {{\"score\": 0.0, \"strengths\": [], \"weaknesses\": []}}"
            eval_data = extract_json_safely(safe_llm_call(eval_prompt)) or {"score": 5.0}
            tech_score = eval_data.get('score', 5.0)
            
            stress = analyzer.detect_stress_level(answer)
            conf = analyzer.detect_confidence(answer)
            
            tech_responses.append({
                'question': question, 'answer': answer, 'technical_score': tech_score,
                'stress_level': stress, 'confidence': conf
            })
            
            q_count = state.get('questions_asked', 0)
            state['stress_level'] = (state.get('stress_level', 0) * q_count + stress) / (q_count + 1)
            state['confidence_score'] = (state.get('confidence_score', 0) * q_count + conf) / (q_count + 1)
            state['questions_asked'] = q_count + 1
            if tech_score >= 7: state['correct_answers'] = state.get('correct_answers', 0) + 1
            print(f"   📊 Technical Score: {tech_score:.1f}/10 | Stress: {stress:.2f}")

        state['technical_responses'] = tech_responses
        state['technical_competency'] = sum(r['technical_score'] for r in tech_responses) / len(tech_responses)
        
    except Exception as e:
        print(f"❌ Technical Agent Error: {e}")

    state['current_agent'] = 'behavioral'
    return state

def behavioral_agent(state: InterviewState) -> InterviewState:
    print("\n" + "="*60 + "\n🎭 BEHAVIORAL AGENT ACTIVE\n" + "="*60)
    beh_prompt = f"""Behavioral interviewer using STAR method for {state['position']}.
Generate 3 questions. Return ONLY JSON: {{"questions": []}}"""
    
    try:
        questions_data = extract_json_safely(safe_llm_call(beh_prompt))
        questions = questions_data['questions']
        state['behavioral_questions'] = questions
        
        beh_responses = []
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}\n(Tip: Use STAR method)")
            answer = input("Your answer: ").strip()
            
            eval_prompt = f"Evaluate STAR answer (0-10). Question: {question}, Answer: {answer}. Return JSON: {{\"score\": 0.0, \"star_complete\": true}}"
            eval_data = extract_json_safely(safe_llm_call(eval_prompt)) or {"score": 5.0}
            score = eval_data.get('score', 5.0)
            
            stress = analyzer.detect_stress_level(answer)
            conf = analyzer.detect_confidence(answer)
            
            beh_responses.append({'question': question, 'answer': answer, 'behavioral_score': score})
            
            q_count = state.get('questions_asked', 0)
            state['stress_level'] = (state.get('stress_level', 0) * q_count + stress) / (q_count + 1)
            state['confidence_score'] = (state.get('confidence_score', 0) * q_count + conf) / (q_count + 1)
            state['questions_asked'] = q_count + 1
            print(f"   📊 Behavioral Score: {score:.1f}/10 | STAR: {'✅' if eval_data.get('star_complete') else '⚠️'}")

        state['behavioral_responses'] = beh_responses
        
    except Exception as e:
        print(f"❌ Behavioral Agent Error: {e}")

    state['current_agent'] = 'evaluator'
    return state

def evaluator_agent(state: InterviewState) -> InterviewState:
    print("\n" + "="*60 + "\n⚖️ EVALUATOR AGENT ACTIVE\n" + "="*60)
    
    summary = {
        'technical': state.get('technical_competency', 0),
        'cultural': state.get('cultural_fit_score', 0),
        'communication': state.get('communication_score', 0),
        'stress': state.get('stress_level', 0),
        'confidence': state.get('confidence_score', 0)
    }
    
    eval_prompt = f"""Final evaluation for {state['candidate_name']}. Data: {json.dumps(summary)}
Return ONLY JSON: {{"overall_score": 0.0, "recommendation": "hire/maybe/no_hire", "detailed_feedback": "..."}}"""
    
    try:
        final_eval = extract_json_safely(safe_llm_call(eval_prompt))
        state['overall_score'] = final_eval['overall_score']
        state['recommendation'] = final_eval['recommendation']
        state['detailed_feedback'] = final_eval['detailed_feedback']
        
        print(f"\n📈 Overall Score: {state['overall_score']:.1f}/10")
        print(f"🎯 Recommendation: {state['recommendation'].upper()}")
        print(f"📝 Feedback: {state['detailed_feedback']}")
        
    except Exception as e:
        print(f"❌ Evaluator Error: {e}")

    state['interview_complete'] = True
    return state

# --- Workflow ---

def create_interview_workflow():
    workflow = StateGraph(InterviewState)
    workflow.add_node("hr_agent", hr_agent)
    workflow.add_node("technical_agent", technical_agent)
    workflow.add_node("behavioral_agent", behavioral_agent)
    workflow.add_node("evaluator_agent", evaluator_agent)
    
    workflow.set_entry_point("hr_agent")
    workflow.add_edge("hr_agent", "technical_agent")
    workflow.add_edge("technical_agent", "behavioral_agent")
    workflow.add_edge("behavioral_agent", "evaluator_agent")
    workflow.add_edge("evaluator_agent", END)
    
    return workflow.compile()

# --- Main ---

def run_interview():
    print("\n🤖 MULTI-AGENT AI INTERVIEWER SYSTEM\n" + "="*60)
    name = input("👤 Candidate Name: ").strip()
    pos = input("💼 Position: ").strip()
    print("📊 Experience: 1. Junior, 2. Mid, 3. Senior")
    exp = {"1": "junior", "2": "mid", "3": "senior"}.get(input("Choice: "), "mid")

    initial_state = {
        'candidate_name': name, 'position': pos, 'experience_level': exp,
        'messages': [], 'hr_questions': [], 'technical_questions': [], 'behavioral_questions': [],
        'hr_responses': [], 'technical_responses': [], 'behavioral_responses': [],
        'stress_level': 0.0, 'confidence_score': 0.0, 'technical_competency': 0.0,
        'communication_score': 0.0, 'cultural_fit_score': 0.0, 'current_difficulty': 'medium',
        'questions_asked': 0, 'correct_answers': 0, 'bias_flags': [],
        'overall_score': 0.0, 'recommendation': '', 'detailed_feedback': '',
        'current_agent': 'hr', 'interview_complete': False, 'has_errors': False
    }

    app = create_interview_workflow()
    try:
        final_state = app.invoke(initial_state)
        filename = f"report_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(final_state, f, indent=2, default=str)
        print(f"\n✅ Interview Complete! Report saved to {filename}")
    except Exception as e:
        print(f"\n❌ Interview failed: {e}")

if __name__ == "__main__":
    run_interview()
