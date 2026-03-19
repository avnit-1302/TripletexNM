import base64
from pathlib import Path
 
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
 
app = FastAPI()
 
@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]
 
    base_url = creds["base_url"]
    token = creds["session_token"]
    auth = ("0", token)
 
    for f in files:
        data = base64.b64decode(f["content_base64"])
        Path(f["filename"]).write_bytes(data)
 
    # TODO: Use an LLM to interpret the prompt and execute
    # the appropriate Tripletex API calls
 
    return JSONResponse({"status": "completed"})
