### equity-research-mutiagent

#### Equity Research Multi-Agent API

##### A LangGraph-based equity research workflow with:
- market data retrieval
- deterministic recommendation and confidence scoring
- schema verifier
- portfolio construction
- consistency verifier
- LLM explanation layer
- FastAPI web interface

##### Run
```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload
