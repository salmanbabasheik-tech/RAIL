from typing import List, Optional, Dict, Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field
from enum import Enum


class ModelProvider(Enum):
    OLLAMA = "ollama"
    GEMINI = "gemini"
    BEDROCK = "bedrock"


@runtime_checkable
class LLMProvider(Protocol):
    def chat(self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]: ...


class Location(BaseModel):
    address: Optional[str] = None
    postalCode: Optional[str] = None
    city: Optional[str] = None
    countryCode: Optional[str] = None
    region: Optional[str] = None

class Profile(BaseModel):
    network: Optional[str] = None
    username: Optional[str] = None
    url: str

class Basics(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[Location] = None
    profiles: Optional[List[Profile]] = None

class Work(BaseModel):
    name: Optional[str] = None
    position: Optional[str] = None
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[List[str]] = None

class Volunteer(BaseModel):
    organization: Optional[str] = None
    position: Optional[str] = None
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[List[str]] = None

class Education(BaseModel):
    institution: Optional[str] = None
    url: Optional[str] = None
    area: Optional[str] = None
    studyType: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    score: Optional[str] = None
    courses: Optional[List[str]] = None

class Award(BaseModel):
    title: Optional[str] = None
    date: Optional[str] = None
    awarder: Optional[str] = None
    summary: Optional[str] = None

class Certificate(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None
    issuer: Optional[str] = None
    url: Optional[str] = None

class Publication(BaseModel):
    name: Optional[str] = None
    publisher: Optional[str] = None
    releaseDate: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None

class Skill(BaseModel):
    name: Optional[str] = None
    level: Optional[str] = None
    keywords: Optional[List[str]] = None

class Language(BaseModel):
    language: Optional[str] = None
    fluency: Optional[str] = None

class Interest(BaseModel):
    name: Optional[str] = None
    keywords: Optional[List[str]] = None

class Reference(BaseModel):
    name: Optional[str] = None
    reference: Optional[str] = None

class Project(BaseModel):
    name: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: Optional[str] = None
    highlights: Optional[List[str]] = None
    url: Optional[str] = None
    technologies: Optional[List[str]] = None
    skills: Optional[List[str]] = None

class BasicsSection(BaseModel):
    basics: Optional[Basics] = None

class WorkSection(BaseModel):
    work: Optional[List[Work]] = None

class EducationSection(BaseModel):
    education: Optional[List[Education]] = None

class SkillsSection(BaseModel):
    skills: Optional[List[Skill]] = None

class ProjectsSection(BaseModel):
    projects: Optional[List[Project]] = None

class AwardsSection(BaseModel):
    awards: Optional[List[Award]] = None

class JSONResume(BaseModel):
    basics: Optional[Basics] = None
    work: Optional[List[Work]] = None
    volunteer: Optional[List[Volunteer]] = None
    education: Optional[List[Education]] = None
    awards: Optional[List[Award]] = None
    certificates: Optional[List[Certificate]] = None
    publications: Optional[List[Publication]] = None
    skills: Optional[List[Skill]] = None
    languages: Optional[List[Language]] = None
    interests: Optional[List[Interest]] = None
    references: Optional[List[Reference]] = None
    projects: Optional[List[Project]] = None

class CategoryScore(BaseModel):
    score: float = Field(ge=0)
    max: int = Field(gt=0)
    evidence: str = Field(min_length=1)

class Scores(BaseModel):
    open_source: CategoryScore
    self_projects: CategoryScore
    production: CategoryScore
    technical_skills: CategoryScore

class BonusPoints(BaseModel):
    total: float = Field(ge=0, le=20)
    breakdown: str

class Deductions(BaseModel):
    total: float = Field(ge=0)
    reasons: str

class EvaluationData(BaseModel):
    scores: Scores
    bonus_points: BonusPoints
    deductions: Deductions
    key_strengths: List[str] = Field(min_items=1, max_items=5)
    areas_for_improvement: List[str] = Field(min_items=1, max_items=5)

class GitHubProfile(BaseModel):
    username: str
    name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    company: Optional[str] = None
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    avatar_url: Optional[str] = None
    blog: Optional[str] = None
    twitter_username: Optional[str] = None
    hireable: Optional[bool] = None


class BedrockProvider:
    def __init__(self, access_key: str, secret_key: str, region: str):
        import boto3
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def chat(self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        import time, random
        system_blocks = []
        converse_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_blocks.append({"text": msg["content"]})
            else:
                converse_messages.append({"role": msg["role"], "content": [{"text": msg["content"]}]})

        # Only temperature — no top_p for Claude Opus 4.6
        inference_config = {"maxTokens": 4096}
        if options and "temperature" in options:
            inference_config["temperature"] = float(options["temperature"])

        for attempt in range(5):
            try:
                params = {"modelId": model, "messages": converse_messages, "inferenceConfig": inference_config}
                if system_blocks:
                    params["system"] = system_blocks
                response = self.client.converse(**params)
                text = response["output"]["message"]["content"][0]["text"]
                return {"message": {"role": "assistant", "content": text}}
            except Exception as e:
                err = str(e)
                if "ThrottlingException" in err or "ServiceUnavailable" in err:
                    if attempt == 4:
                        raise
                    time.sleep(min(5 * (2 ** attempt), 60) * random.uniform(0.8, 1.2))
                else:
                    raise


class OllamaProvider:
    def __init__(self):
        import ollama
        self.client = ollama

    def chat(self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        ollama_options = options.copy() if options else {}
        ollama_options.pop("stream", None)
        ollama_options["num_ctx"] = 32768
        chat_params = {"model": model, "messages": messages, "options": ollama_options}
        if "stream" in kwargs:
            chat_params["stream"] = kwargs["stream"]
        if "format" in kwargs:
            chat_params["format"] = kwargs["format"]
        return self.client.chat(**chat_params)


class GeminiProvider:
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.client = genai

    def chat(self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        import re, time, random
        from google.api_core.exceptions import ResourceExhausted
        generation_config = {}
        if options:
            if "temperature" in options:
                generation_config["temperature"] = options["temperature"]
            if "top_p" in options:
                generation_config["top_p"] = options["top_p"]
        gemini_model = self.client.GenerativeModel(model_name=model, generation_config=generation_config)
        gemini_messages = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in messages]
        for attempt in range(5):
            try:
                response = gemini_model.generate_content(gemini_messages)
                return {"message": {"role": "assistant", "content": response.text}}
            except ResourceExhausted as e:
                if attempt == 4:
                    raise
                time.sleep(min(10 * (2 ** attempt), 120) * random.uniform(0.8, 1.2))
