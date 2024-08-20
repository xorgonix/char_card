import time
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from openai import OpenAI
from pprint import pprint
import  json

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Pydantic model for character data
class Character(BaseModel):
    name: str
    gender: Union[str, None]
    custom_gender: Optional[str] = None
    age: int
    sexual_orientation: Union[str, None]
    custom_sexual_orientation: Optional[str] = None
    occupation: str
    family: Optional[str] = None
    skills: Optional[str] = None
    goals: Optional[str] = None
    personality: List[str] = Field(default_factory=list)
    custom_personality: Optional[str] = None
    appearance: Optional[str] = None
    speech: Optional[str] = None
    likes: List[str] = Field(default_factory=list)
    custom_likes: Optional[str] = None
    dislikes: List[str] = Field(default_factory=list)
    custom_dislikes: Optional[str] = None
    backstory: Optional[str] = None
    body_language: Optional[str] = None
    physical_health: Optional[str] = None
    mental_health: Optional[str] = None

# Initialize the OpenAI client
client = OpenAI()

def generate_character_from_description(description: str) -> Character:
    start_time = time.time()
    print(f"Generating character profile based on the following description: {description}")

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "gender": {"type": "string"},
            "custom_gender": {"type": "string"},
            "age": {"type": "integer"},
            "sexual_orientation": {"type": "string"},
            "custom_sexual_orientation": {"type": "string"},
            "occupation": {"type": "string"},
            "family": {"type": "string"},
            "skills": {"type": "string"},
            "goals": {"type": "string"},
            "personality": {"type": "array", "items": {"type": "string"}},
            "custom_personality": {"type": "string"},
            "appearance": {"type": "string"},
            "speech": {"type": "string"},
            "likes": {"type": "array", "items": {"type": "string"}},
            "custom_likes": {"type": "string"},
            "dislikes": {"type": "array", "items": {"type": "string"}},
            "custom_dislikes": {"type": "string"},
            "backstory": {"type": "string"},
            "body_language": {"type": "string"},
            "physical_health": {"type": "string"},
            "mental_health": {"type": "string"}
        },
        "required": ["name", "gender", "age", "occupation"]
    }

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates detailed character profiles in JSON format based on a given description."},
            {"role": "user", "content": f"Create a character profile based on the following description: {description}"}
        ],
        functions=[
            {
                "name": "generate_character_profile",
                "parameters": schema
            }
        ],
        function_call="auto"
    )

    end_time = time.time()
    duration = end_time - start_time

    character_data = completion.choices[0].message.content
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Total tokens used: {completion.usage.total_tokens}")

    # Parse the character data into the Pydantic model
    character = Character(**character_data)
    return character

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    name: str = Form(...),
    gender: str = Form(...),
    custom_gender: Optional[str] = Form(None),
    age: int = Form(...),
    sexual_orientation: str = Form(...),
    custom_sexual_orientation: Optional[str] = Form(None),
    occupation: str = Form(...),
    family: Optional[str] = Form(None),
    skills: Optional[str] = Form(None),
    goals: Optional[str] = Form(None),
    personality: List[str] = Form(...),
    custom_personality: Optional[str] = Form(None),
    appearance: Optional[str] = Form(None),
    speech: Optional[str] = Form(None),
    likes: List[str] = Form(...),
    custom_likes: Optional[str] = Form(None),
    dislikes: List[str] = Form(...),
    custom_dislikes: Optional[str] = Form(None),
    backstory: Optional[str] = Form(None),
    body_language: Optional[str] = Form(None),
    physical_health: Optional[str] = Form(None),
    mental_health: Optional[str] = Form(None),
):
    character = Character(
        name=name,
        gender=gender,
        custom_gender=custom_gender,
        age=age,
        sexual_orientation=sexual_orientation,
        custom_sexual_orientation=custom_sexual_orientation,
        occupation=occupation,
        family=family,
        skills=skills,
        goals=goals,
        personality=personality,
        custom_personality=custom_personality,
        appearance=appearance,
        speech=speech,
        likes=likes,
        custom_likes=custom_likes,
        dislikes=dislikes,
        custom_dislikes=custom_dislikes,
        backstory=backstory,
        body_language=body_language,
        physical_health=physical_health,
        mental_health=mental_health,
    )

    return templates.TemplateResponse("result.html", {"request": request, "character": character.dict()})

@app.post("/generate", response_class=HTMLResponse)
async def generate_character(request: Request, description: str = Form(...)):
    character_data = generate_character_from_description(description)
    return templates.TemplateResponse("form.html", {"request": request, **character_data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
