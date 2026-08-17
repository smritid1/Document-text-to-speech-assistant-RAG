from dynaconf import Dynaconf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

settings = Dynaconf(
    settings_files=["config/settings.yaml"],
    envvar_prefix="RAG",   # e.g. RAG_LLM_MODEL=llama3 overrides settings.llm_model
    load_dotenv=True,
)