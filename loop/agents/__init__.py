"""loop.agents — every LLM-facing component (D9). Code computes evidence; these modules
turn it into prompts and turn prose into validated records. Nothing here imports the root
``config`` (``SdkLLMClient`` imports ``llm`` lazily, D0.4)."""

from loop.agents.analyst import LLMAnalyst
from loop.agents.briefs import BRIEF_KINDS, brief
from loop.agents.llm import FakeLLMClient, SdkLLMClient
from loop.agents.researcher import LLMResearcher

__all__ = ["LLMAnalyst", "LLMResearcher", "FakeLLMClient", "SdkLLMClient",
           "brief", "BRIEF_KINDS"]
