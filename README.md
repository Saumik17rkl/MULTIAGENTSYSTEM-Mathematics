# Multi-Agent Math Reasoning System

A robust system for solving mathematical problems using multiple specialized AI agents.

## Features

- **Intent Router**: Classifies and routes math problems to appropriate solvers
- **Solver Agent**: Solves mathematical problems step by step
- **Verifier Agent**: Validates the correctness of solutions
- **Explainer Agent**: Provides detailed explanations of solutions
- **Fallback Mechanism**: Automatically switches between LLM providers (Gemini/Groq)

## Prerequisites

- Python 3.9+
- API keys for at least one LLM provider (Gemini or Groq)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-agent-reasoning.git
   cd multi-agent-reasoning
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```env
   # Required: At least one of these
   GOOGLE_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

## Usage

Run the main application:
```bash
python main.py
```

Or start the FastAPI server:
```bash
uvicorn api:app --reload
```

## Deployment

### Render.com

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following environment variables in the Render dashboard:
   - `PYTHON_VERSION`: 3.9
   - `GOOGLE_API_KEY`: Your Gemini API key
   - `GROQ_API_KEY`: Your Groq API key
4. Set the build command: `pip install -r requirements.txt`
5. Set the start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

## Project Structure

```
.
├── agents/                  # Agent implementations
│   ├── __init__.py
│   ├── explainer.py
│   ├── intent_router.py
│   ├── solver.py
│   └── verifier.py
├── llm/                    # LLM client implementations
│   ├── __init__.py
│   ├── gemini_client.py
│   └── groq_client.py
├── tools/                  # External tools
│   └── python_tool.py
├── .env.example           # Example environment variables
├── main.py                # Main application
├── api.py                 # FastAPI application
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | No* | API key for Google's Gemini models |
| `GROQ_API_KEY` | No* | API key for Groq's LLM services |

*At least one API key is required

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
