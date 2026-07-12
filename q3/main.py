from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import Optional
import instructor
from openai import OpenAI
import os

app = FastAPI()
client = instructor.from_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))

class InvoiceExtraction(BaseModel):
    invoice_no: Optional[str] = Field(None)
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")
    vendor: Optional[str] = Field(None)
    amount: Optional[float] = Field(None, description="Subtotal before tax")
    tax: Optional[float] = Field(None, description="Tax amount only")
    currency: Optional[str] = Field(None)

@app.post("/extract")
async def extract(request: Request):
    payload = await request.json()
    text = payload.get("invoice_text", "")
    
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=InvoiceExtraction,
        messages=[
            {"role": "system", "content": "Extract invoice fields strictly. Normalize date to YYYY-MM-DD. Use null if not found."},
            {"role": "user", "content": text}
        ]
    )
