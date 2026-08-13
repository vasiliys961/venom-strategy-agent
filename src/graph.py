"""
LangGraph-граф VENOM: пять узлов-этапов + reflection-роутер + сборка.
Каждый узел вызывает LLM с соответствующим промптом и валидирует,
что обязательные поля этапа заполнены, прежде чем перейти дальше.

Если книга подключена (см. rag.py и ingest_book.py), перед каждым
вызовом LLM подтягивается релевантный контекст из её текста — так
ответы становятся точнее и опираются на реальный текст, а не только
на общую структуру метода.
"""
from __future__ import annotations

import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from .state import VenomCanvas, SmartObjective
from . import prompts
from .rag import retrieve


class VenomGraph:
    def __init__(self, llm):
        self.llm = llm
        self.graph = self._build()

    def _ask(self, system_prompt: str, user_input: str, schema_hint: str) -> dict:
        book_context = retrieve(user_input) if user_input else ""
        full_system_prompt = system_prompt
        if book_context:
            full_system_prompt = system_prompt + "\n\n" + book_context

        messages = [
            SystemMessage(content=full_system_prompt + "\n\nОтветь СтрОгО в формате JSON: " + schema_hint),
            HumanMessage(content=user_input),
        ]
        resp = self.llm.invoke(messages)
        try:
            return json.loads(resp.content)
        except (json.JSONDecodeError, TypeError):
            return {"_raw": resp.content}

    def vision_node(self, state: VenomCanvas, user_input: str = "") -> VenomCanvas:
        data = self._ask(
            prompts.VISION_PROMPT, user_input,
            '{"vision_10y": str, "desired_future": str, "core_values": [str], "question": str}'
        )
        state.vision_10y = data.get("vision_10y", state.vision_10y)
        state.desired_future = data.get("desired_future", state.desired_future)
        state.core_values = data.get("core_values", state.core_values)
        if state.is_stage_complete("vision"):
            state.stage = "evaluation"
        return state

    def evaluation_node(self, state: VenomCanvas, user_input: str = "") -> VenomCanvas:
        prompt = prompts.EVALUATION_PROMPT.format(desired_future=state.desired_future or "")
        data = self._ask(
            prompt, user_input,
            '{"life_spheres": {}, "strengths": [str], "weaknesses": [str], "environment_notes": str, "question": str}'
        )
        state.life_spheres.update(data.get("life_spheres", {}))
        state.strengths.extend(data.get("strengths", []))
        state.weaknesses.extend(data.get("weaknesses", []))
        state.environment_notes = data.get("environment_notes", state.environment_notes)
        if state.is_stage_complete("evaluation"):
            state.stage = "gaps"
        return state

    def gaps_node(self, state: VenomCanvas, user_input: str = "") -> VenomCanvas:
        data = self._ask(
            prompts.GAPS_PROMPT, user_input,
            '{"stability_gaps": [str], "growth_gaps": [str], "root_causes": {}, "question": str}'
        )
        state.stability_gaps.extend(data.get("stability_gaps", []))
        state.growth_gaps.extend(data.get("growth_gaps", []))
        state.root_causes.update(data.get("root_causes", {}))
        if state.is_stage_complete("gaps"):
            state.stage = "objectives"
        return state

    def objectives_node(self, state: VenomCanvas, user_input: str = "") -> VenomCanvas:
        data = self._ask(
            prompts.OBJECTIVES_PROMPT, user_input,
            '{"strategic_goals": [str], "smart_objectives": [{"title": str, "horizon": str, "metric": str, "first_step": str}], "question": str}'
        )
        state.strategic_goals.extend(data.get("strategic_goals", []))
        for o in data.get("smart_objectives", []):
            state.smart_objectives.append(SmartObjective(**o))
        if state.is_stage_complete("objectives"):
            state.stage = "management"
        return state

    def management_node(self, state: VenomCanvas, user_input: str = "") -> VenomCanvas:
        data = self._ask(
            prompts.MANAGEMENT_PROMPT, user_input,
            '{"habits_to_build": [str], "retrospective_cadence": str, "management_system": str, "question": str}'
        )
        state.habits_to_build.extend(data.get("habits_to_build", []))
        state.retrospective_cadence = data.get("retrospective_cadence", state.retrospective_cadence)
        state.management_system = data.get("management_system", state.management_system)
        if state.is_stage_complete("management"):
            state.stage = "assembly"
        return state

    def reflection_node(self, state: VenomCanvas) -> VenomCanvas:
        prompt = prompts.REFLECTION_PROMPT.format(canvas_json=state.model_dump_json())
        data = self._ask(prompt, "", '{"needs_revision_of": str|null, "reason": str}')
        state.needs_revision_of = data.get("needs_revision_of")
        return state

    def assembly_node(self, state: VenomCanvas) -> tuple[VenomCanvas, str]:
        prompt = prompts.ASSEMBLY_PROMPT.format(canvas_json=state.model_dump_json())
        book_context = retrieve(state.desired_future or "")
        if book_context:
            prompt = prompt + "\n\n" + book_context
        resp = self.llm.invoke([SystemMessage(content=prompt)])
        state.stage = "done"
        return state, resp.content

    def _build(self):
        g = StateGraph(VenomCanvas)
        g.add_node("vision", self.vision_node)
        g.add_node("evaluation", self.evaluation_node)
        g.add_node("gaps", self.gaps_node)
        g.add_node("objectives", self.objectives_node)
        g.add_node("management", self.management_node)
        g.set_entry_point("vision")
        g.add_edge("vision", "evaluation")
        g.add_edge("evaluation", "gaps")
        g.add_edge("gaps", "objectives")
        g.add_edge("objectives", "management")
        g.add_edge("management", END)
        return g.compile()
