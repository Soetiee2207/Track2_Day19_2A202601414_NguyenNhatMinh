### 1. Chunking Strategy — Semantic Chunking

I choose **semantic chunking** for episodic memory.

Instead of splitting every message independently or storing an entire conversation as one vector, the system groups consecutive content that is semantically related into the same chunk. A new chunk is created when the semantic similarity between the current content and the existing chunk drops below a configured threshold.

For the POC, the chunking process is:

```text
Vietnamese / English text
        ↓
Embedding Model
        ↓
Vector
        ↓
Cosine Similarity
        ↓
Semantic Chunking
        ↓
Qdrant
```

Example:

```text
Conversation
│
├── Chunk 1: Kubernetes fundamentals
│   ├── Pods
│   ├── Deployments
│   └── Services
│
├── Chunk 2: Kubernetes autoscaling
│   ├── HPA
│   ├── CPU utilization
│   └── Replica scaling
│
└── Chunk 3: Cloud security
    ├── IAM
    ├── Network isolation
    └── Encryption
```

I considered two alternatives.

**Per-message chunking** provides fine-grained retrieval and small context units, but related information can become fragmented across multiple vectors. A query about Kubernetes HPA may retrieve only one message while missing important context from nearby messages.

**Per-conversation chunking** preserves the full conversation context and reduces the number of vectors, but it can retrieve a large amount of unrelated information when a conversation contains multiple topics. This increases context-window usage and can reduce retrieval precision.

Semantic chunking provides a balance:

| Strategy              | Retrieval Quality                   | Storage Cost | Context Window |
| --------------------- | ----------------------------------- | ------------ | -------------- |
| Per-message           | High precision but fragmented       | High         | Low per result |
| Per-conversation      | Lower for multi-topic conversations | Low          | High           |
| **Semantic chunking** | **Balanced**                        | **Moderate** | **Moderate**   |

Therefore, semantic chunking is selected because it keeps semantically related information together while avoiding the unnecessary storage and context-window cost of indexing an entire conversation as a single memory.

For the POC, chunks should also have a soft size limit of approximately **300–500 tokens** to prevent a semantically coherent topic from becoming excessively large. This limit is secondary to semantic boundaries and is mainly used to control context size.

Each chunk stored in Qdrant contains:

```text
{
    user_id,
    chunk_id,
    text,
    source,
    topic,
    created_at
}
```

The `user_id` payload is used as a retrieval filter so that episodic memories are isolated between users.
