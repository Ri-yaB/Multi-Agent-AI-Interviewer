# 🤖 Multi-Agent AI Interviewer System

An advanced, production-ready AI-powered interview platform using **LangGraph** and **Anthropic Claude 3.5 Sonnet**. This system orchestrates multiple specialized agents to conduct comprehensive candidate evaluations.

## 🌟 Features

### Core Capabilities
- **4 Specialized AI Agents**: HR, Technical, Behavioral, and Evaluator agents with distinct roles.
- **Dynamic Difficulty Adjustment**: Automatically adjusts question difficulty based on candidate performance.
- **Real-time Stress Detection**: Analyzes linguistic markers to detect candidate stress levels.
- **Confidence Analysis**: Evaluates confidence levels from response patterns.
- **STAR Method**: Behavioral questions follow the Situation-Task-Action-Result framework.
- **Bias Detection**: Automatically flags potentially biased questions or responses.
- **Communication Scoring**: Evaluates response quality, clarity, and grammar.

### Advanced Analytics
- **Multi-dimensional Evaluation**: Technical competency, behavioral skills, cultural fit, and communication.
- **Comprehensive Reporting**: Generates a detailed JSON report with full interview history and metrics.
- **Visual Analytics**: (Coming soon in CLI) Radar charts and performance trajectory analysis.

## 📊 System Architecture
mermaid
graph TD
    Start((START)) --> HR[HR Agent]
    HR -->|Cultural Fit & Comm| Tech[Technical Agent]
    Tech -->|Skills & Difficulty| Beh[Behavioral Agent]
    Beh -->|STAR Method| Eval[Evaluator Agent]
    Eval -->|Final Decision| End((END))
    
    subgraph AE["Analysis Engine"]
        Stress[Stress Detection]
        Conf[Confidence Analysis]
        Bias[Bias Detection]
        Sentiment[Sentiment Analysis]
    end
    
    HR -.-> Stress
    Tech -.-> Conf
    Beh -.-> Bias


## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Ri-yaB/Multi-Agent-AI-Interviewer.git
cd Multi-Agent-AI-Interviewer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
ANTHROPIC_API_KEY=your_api_key_here
```

### 4. Run the Interviewer
```bash
python main.py
```

## 🛠️ Configuration

You can customize the agents and evaluation criteria in `main.py`. The system uses Pydantic models for structured data validation and LangGraph for workflow orchestration.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
