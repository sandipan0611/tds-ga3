from fastapi import FastAPI, Request
from pydantic import create_model
from typing import Any
import instructor
from openai import OpenAI
import os

app = FastAPI()
# Ensure you set the environment variable in Render dashboard
client = instructor.from_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))

@app.post("/dynamic-extract")
async def extract(request: Request):
    payload = await request.json()
    text = payload["text"]
    schema_dict = payload["schema"]
    
    fields = {k: (str if v == "string" else int if v == "integer" else float if v == "float" else Any, ...) for k, v in schema_dict.items()}
    DynamicModel = create_model("DynamicModel", **fields)
    
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=DynamicModel,
        messages=[{"role": "user", "content": f"Extract the following: {text}"}]
    )
