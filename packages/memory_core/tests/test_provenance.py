"""证据链服务测试：种子数据 → explain_belief 必须给出可追溯的完整链条。"""

import json

import pytest

from memory_core import database
from memory_core.database import get_db, init_db
from memory_core.services.provenance import explain_belief


@pytest.fixture
async def seeded_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "prov.db")
    await init_db()
    db = await get_db()

    # 两条来源记忆
    await db.execute(
        """INSERT INTO memory_items (id, content, kind, importance, weight, admission_tier)
           VALUES ('m1', '这周末又去爬山了，黄山真不错', 'preference', 7.0, 1.0, 'standard')"""
    )
    await db.execute(
        """INSERT INTO memory_items (id, content, kind, importance, weight, admission_tier)
           VALUES ('m2', '买了一双新的登山鞋准备下次爬山', 'preference', 6.0, 1.0, 'standard')"""
    )
    # 准入打分
    await db.execute(
        """INSERT INTO memory_admission_log
           (id, memory_id, raw_content, total_score, admitted, tier, score_utility)
           VALUES ('a1', 'm1', '这周末又去爬山了，黄山真不错', 0.82, 1, 'standard', 0.9)"""
    )
    # 人设特征（含来源）
    await db.execute(
        """INSERT INTO user_persona
           (id, trait_key, trait_value, confidence, evidence_count, source_memory_ids)
           VALUES ('p1', 'hobby', '喜欢爬山等户外运动', 0.87, 2, ?)""",
        (json.dumps(["m1", "m2"]),),
    )
    # 修正日志
    await db.execute(
        """INSERT INTO preference_revision_log (id, persona_id, old_value, new_value, trigger)
           VALUES ('r1', 'p1', '偶尔散步', '喜欢爬山等户外运动', 'user_correction')"""
    )
    # 结构化事实
    await db.execute(
        """INSERT INTO structured_facts (id, memory_id, kind, subject, predicate, object)
           VALUES ('f1', 'm1', 'preference', '我', '喜欢', '爬山')"""
    )
    # 无关干扰记忆
    await db.execute(
        """INSERT INTO memory_items (id, content, kind) VALUES ('m9', '明天要交季度报告', 'task')"""
    )
    await db.commit()
    yield db
    await db.close()


def _fake_retriever(*items):
    """构造确定性 retriever，模拟真实 retrieve_context 的返回（含 _source 通道标签）。"""

    async def _retrieve(query, db):
        return list(items)

    return _retrieve


async def test_full_evidence_chain_for_persona(seeded_db):
    retriever = _fake_retriever(
        {"id": "m1", "_source": "hybrid", "content": "这周末又去爬山了，黄山真不错"}
    )
    result = await explain_belief("爬山", seeded_db, retriever=retriever)

    persona = [b for b in result["beliefs"] if b["type"] == "persona"]
    assert len(persona) == 1
    belief = persona[0]
    assert "爬山" in belief["statement"]
    assert belief["confidence"] == 0.87
    # 证据链回到两条原始记忆
    contents = [e["content"] for e in belief["evidence"]]
    assert any("黄山" in c for c in contents)
    assert any("登山鞋" in c for c in contents)
    # 准入打分随证据带出
    m1_ev = next(e for e in belief["evidence"] if e["memory_id"] == "m1")
    assert m1_ev["admission"]["total_score"] == 0.82
    # 修正日志在场
    assert belief["revisions"][0]["trigger"] == "user_correction"


async def test_fact_belief_without_llm(seeded_db):
    result = await explain_belief("爬山", seeded_db, retriever=_fake_retriever())
    facts = [b for b in result["beliefs"] if b["type"] == "fact"]
    assert len(facts) == 1
    assert facts[0]["statement"] == "我 喜欢 爬山"
    assert facts[0]["evidence"][0]["memory_id"] == "m1"


async def test_supporting_memories_reflect_retrieval_channels(seeded_db):
    retriever = _fake_retriever(
        {"id": "m1", "_source": "hybrid", "content": "黄山"},
        {"id": "m2", "_source": "vector", "content": "登山鞋"},
    )
    result = await explain_belief("爬山", seeded_db, retriever=retriever)
    ids = [m["memory_id"] for m in result["supporting_memories"]]
    assert ids == ["m1", "m2"]
    # 召回通道随证据带出（可对账：每条证据是经哪条通道找到的）
    assert result["retrieval_channels"] == {"hybrid": 1, "vector": 1}
    assert result["supporting_memories"][1]["retrieval_channel"] == "vector"


async def test_explains_memory_recalled_without_keyword_match(seeded_db):
    """护城河核心：向量/同义召回到的记忆，即便正文不含查询关键词，也能被解释。

    旧实现用 content LIKE '%爬山%' 找证据 → 漏掉这条同义记忆；
    新实现以真实检索结果为证据来源 → 能解释，并标注是 'vector' 通道找到的。
    """
    await seeded_db.execute(
        """INSERT INTO memory_items (id, content, kind, admission_tier)
           VALUES ('m_syn', '我热爱户外活动，尤其是登高远眺', 'preference', 'standard')"""
    )
    await seeded_db.commit()

    # 证明旧的子串匹配会漏：该记忆正文不含「爬山」
    cur = await seeded_db.execute(
        "SELECT count(*) AS c FROM memory_items WHERE id='m_syn' AND content LIKE '%爬山%'"
    )
    assert (await cur.fetchone())["c"] == 0

    # 新实现：模拟向量通道召回到它
    retriever = _fake_retriever(
        {"id": "m_syn", "_source": "vector", "content": "我热爱户外活动，尤其是登高远眺"}
    )
    result = await explain_belief("爬山", seeded_db, retriever=retriever)
    ids = [m["memory_id"] for m in result["supporting_memories"]]
    assert "m_syn" in ids, "向量召回的同义记忆必须能被 /why 解释（堵住致命洞）"
    syn = next(m for m in result["supporting_memories"] if m["memory_id"] == "m_syn")
    assert syn["retrieval_channel"] == "vector"


async def test_no_match_returns_empty(seeded_db):
    result = await explain_belief(
        "量子物理", seeded_db, retriever=_fake_retriever()
    )
    assert result["beliefs"] == []
    assert result["supporting_memories"] == []
    assert result["retrieval_channels"] == {}


async def test_blank_query(seeded_db):
    result = await explain_belief("   ", seeded_db)
    assert result["beliefs"] == [] and result["supporting_memories"] == []
