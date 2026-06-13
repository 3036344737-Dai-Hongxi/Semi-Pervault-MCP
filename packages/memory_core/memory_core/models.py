from pydantic import BaseModel, Field


class VoiceUploadResponse(BaseModel):
    id: str
    transcript: str
    confidence: float


class ClarifyRequest(BaseModel):
    voice_record_id: str
    raw_transcript: str


class ClarifyResponse(BaseModel):
    status: str  # "clear" | "unclear"
    normalized_text: str | None = None
    question: str | None = None


class MemoryStoreRequest(BaseModel):
    content: str = Field(..., min_length=1)
    voice_record_id: str | None = None
    tags: list[str] = []


class MemoryUpdateRequest(BaseModel):
    content: str = Field(..., min_length=1)


class MemoryQueuedJob(BaseModel):
    job_id: str
    job_type: str
    reused_existing: bool = False


class MemoryReprocessResponse(BaseModel):
    memory_id: str
    content_version: int
    origin: str
    origin_run_id: str
    jobs: list[MemoryQueuedJob]


class MemoryItem(BaseModel):
    id: str
    voice_record_id: str | None
    content: str
    tags: list[str]
    kind: str
    task_status: str | None = None
    emotion_score: float = 0.0
    consolidated: bool = False
    importance: float = 5.0
    admission_score: float | None = None
    admission_tier: str = "standard"
    weight: float
    last_referenced_at: str | None
    created_at: str


class MemorySearchResult(BaseModel):
    items: list[MemoryItem]
    total: int


class LongTermOverviewResponse(BaseModel):
    persona_count: int = 0
    reflection_count: int = 0
    pending_graph_node_count: int = 0
    low_value_memory_count: int = 0


class PersonaItem(BaseModel):
    id: str
    trait_key: str
    trait_value: str
    confidence: float = 0.0
    evidence_count: int = 0
    source_memory_ids: list[str] = []
    last_updated: str | None = None


class ReflectionListItem(BaseModel):
    id: str
    insight: str
    source_memory_ids: list[str] = []
    source_memory_count: int = 0
    importance: float = 8.0
    created_at: str | None = None


class LongTermLayersResponse(BaseModel):
    persona_items: list[PersonaItem] = []
    reflection_items: list[ReflectionListItem] = []


class MemoryAdmissionExplanation(BaseModel):
    memory_id: str
    utility: float
    confidence: float
    novelty: float
    recency: float
    type_prior: float
    total_score: float
    tier: str
    created_at: str | None = None


class MemoryAdmissionExplanationResponse(BaseModel):
    memory_id: str
    explanation: MemoryAdmissionExplanation | None = None


class MemoryPipelineTraceJob(BaseModel):
    job_type: str
    status: str
    origin: str
    origin_run_id: str | None = None
    attempt_count: int = 0
    subject_version: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    terminal_reason: str | None = None
    last_error: str | None = None


class MemoryPipelineTraceRun(BaseModel):
    origin_run_id: str | None = None
    origin: str
    subject_version: int | None = None
    job_count: int = 0
    status_counts: dict[str, int] = {}
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    is_current_version: bool = True
    jobs: list[MemoryPipelineTraceJob] = []


class MemoryPipelineTraceResponse(BaseModel):
    memory_id: str
    content_version: int | None = None
    hidden_job_count: int = 0
    runs: list[MemoryPipelineTraceRun] = []
    jobs: list[MemoryPipelineTraceJob] = []


class MemoryAIStageCounts(BaseModel):
    total: int = 0
    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    dead: int = 0
    obsolete: int = 0
    succeeded: int = 0


class MemoryAIStageHealth(BaseModel):
    stage_id: str
    label: str
    category: str
    provider: str
    configured: bool = True
    health: str
    counts: MemoryAIStageCounts = MemoryAIStageCounts()
    latest_status: str | None = None
    latest_terminal_reason: str | None = None
    last_started_at: str | None = None
    last_updated_at: str | None = None
    last_finished_at: str | None = None
    last_error: str | None = None
    checkpoint_created_at: str | None = None
    recent_memory_count: int | None = None
    recent_output_count: int | None = None


class MemoryAIHealthResponse(BaseModel):
    openai_configured: bool
    embedding_configured: bool
    sleep_agent_enabled: bool
    worker_running: bool
    sleep_agent_last_run_status: str | None = None
    sleep_agent_last_started_at: str | None = None
    sleep_agent_last_finished_at: str | None = None
    sleep_agent_last_error_count: int = 0
    stages: list[MemoryAIStageHealth] = []


class MemoryExportRequest(BaseModel):
    confirm_export: bool


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatSource(BaseModel):
    id: str
    content: str
    created_at: str


class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    needs_clarification: bool = False
    clarification_question: str | None = None


class ChatSession(BaseModel):
    id: str
    title: str
    last_message: str
    timestamp: str
    message_count: int
    messages: list[ChatMessage]


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSession]


class ChatResponse(BaseModel):
    reply: str
    sources: list[ChatSource]
    needs_clarification: bool = False
    clarification_question: str | None = None


class MemoryReflection(BaseModel):
    id: str
    insight: str
    source_memory_ids: list[str] = []
    importance: float = 8.0
    created_at: str | None = None


class PreferenceRevisionLog(BaseModel):
    id: str
    persona_id: str | None = None
    old_value: str | None = None
    new_value: str | None = None
    trigger: str | None = None
    created_at: str | None = None


# ── Stage 3: Graph models ──────────────────────────────


class GraphNode(BaseModel):
    id: str
    type: str
    label: str
    properties: dict = {}
    weight: float = 1.0
    source_memory_count: int = 0
    created_at: str | None = None
    last_seen_at: str | None = None
    status: str = "confirmed"
    possible_duplicate_of: str | None = None


class GraphEdge(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    source_memory_id: str | None = None
    created_at: str | None = None


class GraphEdgeWithLabels(GraphEdge):
    """GraphEdge extended with resolved node labels, used only in node-detail responses."""
    source_label: str | None = None
    target_label: str | None = None


class GraphExtractRequest(BaseModel):
    memory_item_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class GraphExtractResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class GraphSubgraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class MemoryBrief(BaseModel):
    id: str
    content: str
    created_at: str


class GraphNodeDetailResponse(BaseModel):
    node: GraphNode
    edges: list[GraphEdgeWithLabels]
    source_memories: list[MemoryBrief]


class GraphPendingResponse(BaseModel):
    nodes: list[GraphNode]
    candidates: list[GraphNode]
