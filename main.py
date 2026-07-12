from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import create_model
from typing import Any
import instructor
from openai import OpenAI
import os

app = FastAPI()

# Enable CORS so the grader's Cloudflare Worker can reach your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the client with your API key from Render's environment variables
client = instructor.from_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))

@app.post("/dynamic-extract")
async def extract(request: Request):
    # Receive the text and the dynamic schema from the grader
    payload = await request.json()
    text = payload.get("text", "")
    schema_dict = payload.get("schema", {})
    
    # Dynamically map the schema types to Pydantic types
    type_map = {
        "string": str,
        "integer": int,
        "float": float,
        "date": str,  # LLM handles the format, we validate as string
    }
    
    # Create the Pydantic model at runtime
    fields = {k: (type_map.get(v, Any), ...) for k, v in schema_dict.items()}
    DynamicModel = create_model("DynamicModel", **fields)
    
    # Use instructor to enforce the structure
    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=DynamicModel,
        messages=[
            {"role": "system", "content": "Extract the data exactly matching the requested schema. Return null for missing fields."},
            {"role": "user", "content": text}
        ]
    )
    
    return response.model_dump()
