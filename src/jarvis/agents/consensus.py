"""ConsensusAgent — runs a prompt through N ResearcherAgents in parallel and returns
the highest-scoring response as judged by CriticAgent.

Use for high-stakes queries where hallucination risk matters. The standard single-agent
path is the default; consensus is opt-in.

Usage:
    agent = ConsensusAgent(model, max_tokens, tool_schemas, tool_registry, n_agents=3)
    best_response = agent.run(prompt)
"""
from __future__ import annotations

import structlog
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

log = structlog.get_logger()


class ConsensusAgent:
    def __init__(
        self,
        model: str,
        max_tokens: int,
        tool_schemas: list[dict],
        tool_registry: dict[str, Callable[[dict], str]],
        n_agents: int = 3,
        session_id: str = "",
        user_id: str | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._tool_schemas = tool_schemas
        self._tool_registry = tool_registry
        self._n_agents = max(2, n_agents)
        self._session_id = session_id
        self._user_id = user_id

    def run(self, prompt: str) -> str:
        """Run prompt through N agents in parallel; return the best-scoring response."""
        from jarvis.agents.researcher import ResearcherAgent
        from jarvis.agents.critic import build_critic

        def _run_one(_: int) -> str:
            agent = ResearcherAgent(
                model=self._model,
                max_tokens=self._max_tokens,
                tool_schemas=self._tool_schemas,
                tool_registry=self._tool_registry,
                session_id=self._session_id,
                user_id=self._user_id,
            )
            try:
                response, _ = agent.run_turn([{"role": "user", "content": prompt}])
                return response
            except Exception as exc:
                log.warning("consensus_agent_failed", error=str(exc))
                return ""

        with ThreadPoolExecutor(max_workers=self._n_agents) as pool:
            futures = [pool.submit(_run_one, i) for i in range(self._n_agents)]
            responses = [f.result() for f in as_completed(futures)]

        responses = [r for r in responses if r]
        if not responses:
            return ""
        if len(responses) == 1:
            return responses[0]

        # Score each with CriticAgent; return the highest-scoring candidate
        critic = build_critic(self._model, self._max_tokens)
        scored = []
        for resp in responses:
            try:
                critique = critic.critique(prompt, resp)
                scored.append((critique.score, resp))
            except Exception:
                scored.append((0, resp))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_response = scored[0]
        log.info("consensus_complete",
                 n_agents=self._n_agents,
                 scores=[s for s, _ in scored],
                 best_score=best_score)
        return best_response
