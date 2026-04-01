from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from graph_workflow import graph

app = FastAPI(title="Equity Research Multi-Agent API")


class AnalyzeRequest(BaseModel):
    tickers: List[str] = Field(
        ...,
        min_length=1,
        json_schema_extra={"example": ["AAPL", "MSFT", "NVDA"]}
    )


@app.get("/")
def root():
    return {
        "message": "Equity Research Multi-Agent API is running.",
        "docs": "/docs"
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    clean_tickers = [t.strip().upper() for t in request.tickers if t.strip()]
    if not clean_tickers:
        raise HTTPException(status_code=400, detail="No valid tickers provided.")

    result = graph.invoke({"tickers": clean_tickers})
    return result