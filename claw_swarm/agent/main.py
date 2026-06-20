from __future__ import annotations

import os
import re
import traceback

from swarms import Agent, HierarchicalSwarm
from claw_swarm.agent.prompts import (
    CLAUDE_HELPER_DESCRIPTION,
    CLAUDE_HELPER_NAME,
    CLAUDE_TOOL_SYSTEM,
    TELEGRAM_SUMMARY_SYSTEM,
    build_director_system_prompt,
)
from claw_swarm.config import AgentConfig
from claw_swarm.tools import run_claude_agent
from claw_swarm.agent.model_config import resolve_llm, resolve_model
from claw_swarm.agent.worker_agents import (
    create_developer_agent,
    create_response_agent,
    create_search_agent,
    create_token_launch_agent,
)


def _build_worker_agents(
    worker_model: str | None = None,
    agent_config: AgentConfig | None = None,
) -> list:
    """
    Create the four ClawSwarm worker agents.

    Args:
        worker_model: Optional model spec for workers. Supports cloud model
            names (``"gpt-5.4"``), or local prefixes
            (``"vllm/<model>"``, ``"hf/<model>"``).  When None the
            WORKER_MODEL_NAME / AGENT_MODEL env vars are used, then the
            built-in default.
        agent_config: Optional agent config with per-worker overrides for
            name, description, system_prompt, model, max_tokens, temperature.

    Returns:
        List of four worker Agent instances.
    """
    cfg = agent_config or AgentConfig()
    search_cfg = cfg.workers.get("search")
    response_cfg = cfg.workers.get("response")
    developer_cfg = cfg.workers.get("developer")
    token_launch_cfg = cfg.workers.get("token_launch")

    return [
        create_response_agent(
            model_name=worker_model or (response_cfg.model if response_cfg else None),
            agent_name=response_cfg.name if response_cfg and response_cfg.name else None,
            agent_description=response_cfg.description if response_cfg else None,
            system_prompt=response_cfg.system_prompt if response_cfg else None,
            max_tokens=response_cfg.max_tokens if response_cfg else 0,
            temperature=response_cfg.temperature if response_cfg else 0.0,
        ),
        create_developer_agent(
            model_name=worker_model or (developer_cfg.model if developer_cfg else None),
            agent_name=developer_cfg.name if developer_cfg and developer_cfg.name else None,
            agent_description=developer_cfg.description if developer_cfg else None,
            system_prompt=developer_cfg.system_prompt if developer_cfg else None,
            max_tokens=developer_cfg.max_tokens if developer_cfg else 0,
            temperature=developer_cfg.temperature if developer_cfg else 0.0,
        ),
        create_search_agent(
            model_name=worker_model or (search_cfg.model if search_cfg else None),
            agent_name=search_cfg.name if search_cfg and search_cfg.name else None,
            agent_description=search_cfg.description if search_cfg else None,
            system_prompt=search_cfg.system_prompt if search_cfg else None,
            max_tokens=search_cfg.max_tokens if search_cfg else 0,
            temperature=search_cfg.temperature if search_cfg else 0.0,
        ),
        create_token_launch_agent(
            model_name=worker_model or (token_launch_cfg.model if token_launch_cfg else None),
            agent_name=token_launch_cfg.name if token_launch_cfg and token_launch_cfg.name else None,
            agent_description=token_launch_cfg.description if token_launch_cfg else None,
            system_prompt=token_launch_cfg.system_prompt if token_launch_cfg else None,
            max_tokens=token_launch_cfg.max_tokens if token_launch_cfg else 0,
            temperature=token_launch_cfg.temperature if token_launch_cfg else 0.0,
        ),
    ]


def call_claude(task: str) -> str:
    """
    Run a specified task using Claude as the reasoning and coding engine.

    Args:
        task (str): The task or question for Claude to address. This can be
            long-form analysis, code generation, explanation, or complex multi-step reasoning.

    Returns:
        str: Claude's response(s), joined into a single string. Returns an empty string on failure.

    Example:
        >>> call_claude("Write a summary of the Python standard library.")
        'The Python standard library ...'
    """
    responses = run_claude_agent(
        name=CLAUDE_HELPER_NAME,
        description=CLAUDE_HELPER_DESCRIPTION,
        prompt=CLAUDE_TOOL_SYSTEM,
        tasks=task,
    )
    return (
        "\n\n".join(r for r in responses if r).strip()
        if responses
        else ""
    )


def _agent_name(default: str = "ClawSwarm") -> str:
    return os.environ.get("CLAWSWARM_AGENT_NAME", default)


def _agent_description(
    default: str = "A hierarchical swarm of agents that can "
    "handle complex tasks",
) -> str:
    return os.environ.get("CLAWSWARM_AGENT_DESCRIPTION", default)


def create_agent(
    *,
    agent_name: str | None = None,
    system_prompt: str | None = None,
    description: str | None = None,
    director_model: str | None = None,
    worker_model: str | None = None,
    agent_config: AgentConfig | None = None,
) -> HierarchicalSwarm:
    """
    Create the ClawSwarm hierarchical swarm: a director agent plus worker agents
    (search, token launch, developer). Use for enterprise chat, technical, and
    research use-cases with delegation to specialists.

    The director uses the ClawSwarm system prompt and the swarm's built-in
    director (SwarmSpec output) so plan/orders are parsed correctly.

    Model selection (director and workers are configured independently):

    *Cloud models (default):*
    Pass any OpenAI / Anthropic model name, e.g. ``"gpt-5.4"`` or set the
    ``AGENT_MODEL`` / ``WORKER_MODEL_NAME`` environment variables.

    *Local models via vLLM:*
    Use the ``vllm/<model>`` prefix — requires ``pip install vllm`` and a GPU::

        create_agent(director_model="vllm/mistralai/Mistral-7B-Instruct-v0.1")

    Or connect to a running vLLM HTTP server::

        create_agent(director_model="vllm-server/meta-llama/Llama-2-7b-chat-hf")

    *Local models via HuggingFace Transformers:*
    Use the ``hf/<model>`` prefix — requires ``pip install transformers accelerate``::

        create_agent(
            director_model="hf/microsoft/phi-2",
            worker_model="hf/microsoft/phi-2",
        )

    These prefixes can also be set via environment variables::

        AGENT_MODEL=vllm/mistralai/Mistral-7B-Instruct-v0.1
        WORKER_MODEL_NAME=hf/microsoft/phi-2

    Args:
        agent_name:     Name for the swarm and director (shown in logs/UI).
        system_prompt:  Override the default ClawSwarm prompt for the director.
        description:    Override the default swarm description.
        director_model: Model spec for the director agent. Supports cloud names
            or local prefixes (``vllm/``, ``vllm-server/``, ``hf/``).
            Falls back to AGENT_MODEL env var, then ``"gpt-5.4"``.
        worker_model:   Model spec for worker agents. Falls back to
            WORKER_MODEL_NAME env var, AGENT_MODEL, then ``"gpt-5.4"``.
        agent_config:   Optional AgentConfig with per-agent overrides loaded from
            ``claw_swarm_agents.yaml`` (or AGENT_CONFIG_PATH env var). Supports
            ``director`` (name, description, system_prompt, model, max_tokens,
            temperature) and ``workers`` dict with keys: search, response,
            developer, token_launch.

    Returns:
        HierarchicalSwarm: Swarm ready for `.run(task)` calls.

    Example:
        >>> swarm = create_agent()
        >>> reply = swarm.run("What's new in Python 3.12?")
        >>> print(reply)
        'Python 3.12 introduces ...'
    """
    cfg = agent_config or AgentConfig()
    director_cfg = cfg.director

    name = agent_name or (director_cfg.name if director_cfg.name else _agent_name())
    desc = description or (director_cfg.description if director_cfg.description else _agent_description())

    final_system_prompt = system_prompt or (director_cfg.system_prompt if director_cfg.system_prompt else None)

    director_system_prompt = build_director_system_prompt(
        agent_name=name,
        system_prompt=final_system_prompt,
    )

    workers = _build_worker_agents(worker_model, agent_config=cfg)

    final_director_model = director_model or (director_cfg.model if director_cfg.model else None)

    director_spec = (
        final_director_model
        or os.environ.get("AGENT_MODEL", "").strip()
        or "gpt-5.4"
    )
    cloud_model, llm_obj = resolve_llm(
        director_spec, default="gpt-5.4"
    )

    director_kwargs: dict[str, object] = {
        "agent_name": name,
        "agent_description": desc,
        "system_prompt": director_system_prompt,
        "max_loops": 1,
    }
    if llm_obj is not None:
        director_kwargs["llm"] = llm_obj
    else:
        director_kwargs["model_name"] = cloud_model
    if director_cfg.max_tokens > 0:
        director_kwargs["max_tokens"] = director_cfg.max_tokens
    if director_cfg.temperature > 0.0:
        director_kwargs["temperature"] = director_cfg.temperature

    if llm_obj is not None:
        director_agent = Agent(**director_kwargs)
        return HierarchicalSwarm(
            name=name,
            description=desc,
            agents=workers,
            director_name=name,
            director_system_prompt=director_system_prompt,
            director_feedback_on=False,
            director=director_agent,
        )

    if director_cfg.max_tokens > 0 or director_cfg.temperature > 0.0:
        director_agent = Agent(**director_kwargs)
        return HierarchicalSwarm(
            name=name,
            description=desc,
            agents=workers,
            director_name=name,
            director_system_prompt=director_system_prompt,
            director_feedback_on=False,
            director=director_agent,
        )

    return HierarchicalSwarm(
        name=name,
        description=desc,
        agents=workers,
        director_name=name,
        director_model_name=cloud_model,
        director_system_prompt=director_system_prompt,
        director_feedback_on=False,
        director=None,
    )


def hierarchical_swarm(task: str):
    """
    Execute a task using the ClawSwarm hierarchical swarm (convenience wrapper).

    Args:
        task (str): The main task or instruction to be performed by the swarm.

    Returns:
        Any: The result returned by the swarm, or None on exception.
    """
    try:
        return create_agent().run(task)
    except Exception as e:
        print(
            f"Error running hierarchical_swarm: {e}\n{traceback.format_exc()}"
        )
        return None


# Emoji pattern for stripping any that slip through the summarizer
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)


def _create_summarizer_agent() -> Agent:
    """Create an agent that summarizes long output (no emojis)."""
    return Agent(
        agent_name="ClawSwarm-Summarizer",
        agent_description="Summarizes swarm output into concise messages; no emojis.",
        system_prompt=TELEGRAM_SUMMARY_SYSTEM,
        model_name=resolve_model(None, default="gpt-4.1"),
        max_loops=1,
    )


def summarize_for_telegram(swarm_output: str) -> str:
    """
    Take the raw output from the hierarchical swarm and return a concise
    summary suitable for Telegram, with no emojis.

    Args:
        swarm_output: Raw string output from the swarm (may be long or
            multi-part). If it's not a string (e.g. list from feedback_director),
            pass str(swarm_output).

    Returns:
        Summarized text for Telegram, with emojis stripped. Returns the
        original string (with emojis stripped) if summarization fails.
    """
    if not swarm_output or not str(swarm_output).strip():
        return ""

    summarizer = _create_summarizer_agent()

    out = summarizer.run(
        f"Summarize the following output for a Telegram message. No emojis.\n\n{swarm_output}"
    )

    return out
