"""Concrete planner providers.

The **only** place in the agent package where a language-model SDK may be
imported. Everything above this package - schemas, registry, executor,
grounding, planner - depends solely on the provider-neutral
:class:`~app.services.agent.planner.AgentPlanner` abstraction, and an AST test
asserts the SDK appears in exactly one file here.
"""
