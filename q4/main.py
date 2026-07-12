from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import create_model, Field
from typing import Any, Optional, List
import instructor
from openai import OpenAI
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = instructor.from_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))

@app.post("/dynamic-extract")
async def extract(request: Request):
    try:
        payload = await request.json()
        text = payload.get("text", "")
        schema_dict = payload.get("schema", {})
        
        # Mapping types from your provided assignment file
        type_map = {
            "string": str,
            "integer": int,
            "float": float,
            "boolean": bool,
            "date": str,
            "array[string]": List[str],
            "array[integer]": List[int]
        }
        
        # Create fields dynamically; use Optional/Field to allow nulls as required
        fields = {k: (Optional[type_map.get(v, Any)], Field(default=None)) for k, v in schema_dict.items()}
        DynamicModel = create_model("DynamicModel", **fields)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=DynamicModel,
            messages=[
                {"role": "system", "content": "Extract data based on the schema. Dates must be YYYY-MM-DD. Use null for missing fields."},
                {"role": "user", "content": text}
            ]
        )
        return response.model_dump()
    except Exception as e:
        print(f"Extraction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
