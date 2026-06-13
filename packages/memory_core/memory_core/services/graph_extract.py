import os
import re
import json
import logging
from memory_core.services.llm import get_client, raise_ai_service_error
from memory_core.services.memory_policy import (
    STORE_PATH_NODE_TYPES,
    CONSOLIDATION_NODE_TYPES,
    ALLOWED_RELATIONS,
)

logger = logging.getLogger(__name__)

GRAPH_EXTRACT_PROMPT = """\
你是一个知识图谱抽取助手。给定一段用户记忆文本，从中抽取实体节点和关系边。

严格按以下 JSON 格式返回：
{
  "nodes": [
    {"label": "实体名称", "type": "person|project|task|idea"}
  ],
  "edges": [
    {"source": "源节点 label", "target": "目标节点 label", "relation": "related_to|belongs_to|mentioned_with"}
  ]
}

节点类型只允许以下四种：
- person：人名、合作对象、领导、同事
- project：项目、计划、方案
- task：明确的动作或待办
- idea：想法、概念、策略

关系类型只允许以下三种：
- related_to：一般关联
- belongs_to：从属关系
- mentioned_with：同时被提及

规则：
- 只抽取文本中明确提及的实体，不要推测
- 不要把无意义的口语词、语气词变成节点
- label 应简洁准确，不要包含冗余修饰
- 如果文本中没有可抽取的实体，返回 {"nodes": [], "edges": []}
- 只返回 JSON，不要有其它文字"""

CONSOLIDATION_GRAPH_EXTRACT_PROMPT = """\
你是一个离线记忆整合助手。给定一段用户记忆文本，只在它明显值得进入长期关系图谱时，抽取高价值节点和关系。

严格按以下 JSON 格式返回：
{
  "nodes": [
    {"label": "实体名称", "type": "person|project|event"}
  ],
  "edges": [
    {"source": "源节点 label", "target": "目标节点 label", "relation": "related_to|belongs_to|mentioned_with"}
  ]
}

节点类型只允许以下三种：
- person：明确的人名、合作对象、稳定关系对象
- project：明确的项目、长期计划、持续事项
- event：高价值事件，例如关键会面、重要里程碑、重大活动

规则：
- 只抽取高价值、可复用的长期关系信息
- 不要抽取 task、idea、临时碎片、情绪、闲聊
- event 只在事件本身足够重要且具名/可指代时才建立
- 只抽取文本中明确提及的实体，不要推测
- 不要把无意义的口语词、语气词变成节点
- label 应简洁准确，不要包含冗余修饰
- 如果文本不值得进入图谱，返回 {"nodes": [], "edges": []}
- 只返回 JSON，不要有其它文字"""


class GraphExtractionError(Exception):
    pass


def _normalize_label(label: str) -> str:
    text = label.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _validate_and_clean(
    raw: dict,
    *,
    valid_node_types: set[str],
    valid_relations: set[str],
) -> dict:
    """Validate and normalize extracted nodes/edges, dropping invalid entries."""
    clean_nodes: list[dict] = []
    seen_labels: set[tuple[str, str]] = set()

    for n in raw.get("nodes", []):
        label = _normalize_label(str(n.get("label", "")))
        ntype = str(n.get("type", "")).strip().lower()
        if not label or ntype not in valid_node_types:
            continue
        key = (ntype, label)
        if key in seen_labels:
            continue
        seen_labels.add(key)
        clean_nodes.append({"label": label, "type": ntype})

    node_labels = {n["label"] for n in clean_nodes}
    clean_edges: list[dict] = []
    for e in raw.get("edges", []):
        src = _normalize_label(str(e.get("source", "")))
        tgt = _normalize_label(str(e.get("target", "")))
        rel = str(e.get("relation", "")).strip().lower()
        if rel not in valid_relations:
            continue
        if src not in node_labels or tgt not in node_labels:
            continue
        if src == tgt:
            continue
        clean_edges.append({"source": src, "target": tgt, "relation": rel})

    return {"nodes": clean_nodes, "edges": clean_edges}


async def _extract_graph_with_prompt(
    content: str,
    *,
    system_prompt: str,
    valid_node_types: set[str],
    valid_relations: set[str],
) -> dict:
    # Reuse the shared LLM client so OPENAI_BASE_URL applies here too.
    c = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    try:
        resp = await c.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise_ai_service_error(exc, "LLM 图谱抽取服务")
    text = resp.choices[0].message.content or "{}"
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GraphExtractionError(f"LLM 返回了无效 JSON: {text[:200]}") from exc

    return _validate_and_clean(
        raw,
        valid_node_types=valid_node_types,
        valid_relations=valid_relations,
    )


async def extract_graph(content: str) -> dict:
    """Extract nodes and edges from memory content via LLM.

    Returns {"nodes": [...], "edges": [...]}.
    Raises GraphExtractionError on unrecoverable failure.
    """
    return await _extract_graph_with_prompt(
        content,
        system_prompt=GRAPH_EXTRACT_PROMPT,
        valid_node_types=STORE_PATH_NODE_TYPES,
        valid_relations=ALLOWED_RELATIONS,
    )


async def extract_consolidation_graph(content: str) -> dict:
    """Extract high-value graph nodes for offline consolidation only."""
    return await _extract_graph_with_prompt(
        content,
        system_prompt=CONSOLIDATION_GRAPH_EXTRACT_PROMPT,
        valid_node_types=CONSOLIDATION_NODE_TYPES,
        valid_relations=ALLOWED_RELATIONS,
    )
