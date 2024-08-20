import time
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from openai import OpenAI
from pprint import pprint

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Pydantic model for character data
class Character(BaseModel):
    name: str
    gender: List[str] = Field(default_factory=list)
    age: int
    sexual_orientation: List[str] = Field(default_factory=list)
    occupation: List[str] = Field(default_factory=list)
    family: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    personality: List[str] = Field(default_factory=list)
    appearance: List[str] = Field(default_factory=list)
    speech: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    backstory: List[str] = Field(default_factory=list)
    body_language: List[str] = Field(default_factory=list)
    physical_health: List[str] = Field(default_factory=list)
    mental_health: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)

# Initialize the OpenAI client
client = OpenAI()

def generate_character_from_description(description: str) -> dict:
    start_time = time.time()
    print(f"Generating character profile based on the following description: {description}")
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates detailed character profiles."},
            {"role": "user", "content": f"Create a character profile based on the following description: {description}"}
        ]
    )

    end_time = time.time()
    duration = end_time - start_time

    character_data = completion.choices[0].message.content
    print("----------------------------------------")
    print(completion)
    print("----------------------------------------")
    
    token_usage = completion.usage
    tokens_input = token_usage.prompt_tokens
    tokens_output = token_usage.completion_tokens
    tokens_total = tokens_input + tokens_output
    
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Total tokens used: {tokens_total} (Prompt tokens: {tokens_input}, Completion tokens: {tokens_output})")
    pprint(completion)

    return eval(character_data)  # Be cautious with eval()

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form_01.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    name: str = Form(...),
    gender: List[str] = Form(...),
    age: int = Form(...),
    sexual_orientation: List[str] = Form(...),
    occupation: List[str] = Form(...),
    family: List[str] = Form(...),
    skills: List[str] = Form(...),
    goals: List[str] = Form(...),
    personality: List[str] = Form(...),
    appearance: List[str] = Form(...),
    speech: List[str] = Form(...),
    likes: List[str] = Form(...),
    dislikes: List[str] = Form(...),
    backstory: List[str] = Form(...),
    body_language: List[str] = Form(...),
    physical_health: List[str] = Form(...),
    mental_health: List[str] = Form(...),
    languages: List[str] = Form(...),
):
    character = Character(
        name=name,
        gender=gender,
        age=age,
        sexual_orientation=sexual_orientation,
        occupation=occupation,
        family=family,
        skills=skills,
        goals=goals,
        personality=personality,
        appearance=appearance,
        speech=speech,
        likes=likes,
        dislikes=dislikes,
        backstory=backstory,
        body_language=body_language,
        physical_health=physical_health,
        mental_health=mental_health,
        languages=languages,
    )

    return templates.TemplateResponse("result.html", {"request": request, "character": character.dict()})

@app.post("/generate", response_class=HTMLResponse)
async def generate_character(request: Request, description: str = Form(...)):
    character_data = generate_character_from_description(description)
    return templates.TemplateResponse("form_01.html", {"request": request, **character_data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
