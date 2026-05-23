import os
from crewai import LLM
from src.agents_src.config.agent_settings import AgentSettings

def get_llm_for_agent(agent_name: str):
    settings = AgentSettings()
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
    
    # CrewAI natively supports LiteLLM format, so prefix with provider if needed.
    model_name = settings.MODEL_NAME
    if not model_name.startswith("groq/"):
        model_name = f"groq/{model_name}"
        
    return LLM(
        model=model_name,
        temperature=settings.MODEL_TEMPERATURE
    )
