# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""An explicit dependency exit must not consume successive source groups."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from relax.agentic.runner import AgentDependencyError, AgentExecutionError
from relax.agentic.runner.ipc import ManagedAgentProcess, ManagedCommandAppSpec, SessionInput


@pytest.mark.parametrize(
    ("exit_code", "output", "error_type"),
    [
        (75, None, AgentDependencyError),
        (2, None, AgentExecutionError),
        (75, '{"reward": 1}', AgentExecutionError),
        (75, "invalid output", AgentExecutionError),
        (0, '{"reward": 1}', None),
    ],
)
async def test_dependency_exit_requires_no_output_and_preserves_other_exit_semantics(
    tmp_path, monkeypatch, exit_code, output, error_type
) -> None:
    monkeypatch.setenv("RELAX_AGENT_DEPENDENCY_EXIT_SUPPORTED", "inherited-value")
    launched_env = {}
    child = SimpleNamespace(wait=AsyncMock(return_value=exit_code), close=AsyncMock())

    async def launch(**kwargs):
        launched_env.update(kwargs["env"])
        Path(kwargs["log_path"]).write_text("fixture agent diagnostic\n")
        if output is not None:
            Path(kwargs["env"]["RELAX_OUTPUT_JSON"]).write_text(output)
        return child

    process = ManagedAgentProcess(
        SimpleNamespace(launch=launch),
        ManagedCommandAppSpec(
            command="fixture-agent",
            cwd=str(tmp_path),
            env={"RELAX_AGENT_DEPENDENCY_EXIT_SUPPORTED": "app-value"},
        ),
        SessionInput(session_id="session", rollout_mode="train", group_id="group"),
    )
    if error_type is None:
        await process.wait()
        assert process.output.reward == 1
    else:
        with pytest.raises(error_type) as caught:
            await process.wait()
        if error_type is AgentDependencyError:
            assert not isinstance(caught.value, AgentExecutionError)
    assert launched_env["RELAX_AGENT_DEPENDENCY_EXIT_SUPPORTED"] == "1"
    child.close.assert_awaited_once()
