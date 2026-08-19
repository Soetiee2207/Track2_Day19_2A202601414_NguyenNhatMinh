
from agent import HybridMemoryAgent


def seed_memories(agent: HybridMemoryAgent) -> None:
    """Add sample episodic memories for the demo."""

    memories = [
        """
        Tôi đã đọc về Kubernetes fundamentals.
        Kubernetes sử dụng Pods để chạy workload.
        Deployments quản lý việc triển khai và cập nhật Pods.
        Services cung cấp network access ổn định cho các Pods.
        """,

        """
        Tôi đang tìm hiểu Kubernetes autoscaling.
        Horizontal Pod Autoscaler (HPA) có thể tự động tăng
        hoặc giảm số lượng replicas dựa trên CPU utilization.
        Resource requests và limits ảnh hưởng đến việc scheduling
        và autoscaling của Kubernetes.
        """,

        """
        Tôi đã đọc về AWS Auto Scaling.
        Auto Scaling có thể điều chỉnh compute capacity
        dựa trên workload và nhu cầu của hệ thống.
        Đây là một cách phổ biến để tự động mở rộng hạ tầng cloud.
        """,

        """
        Tôi đang tìm hiểu cloud security.
        Các chủ đề quan trọng gồm IAM, encryption,
        network isolation và access control.
        IAM giúp kiểm soát user và service được phép
        truy cập tài nguyên cloud nào.
        """,

        """
        Gần đây tôi quan tâm nhiều đến cloud infrastructure,
        Kubernetes và AWS.
        Tôi muốn học thêm về autoscaling và cloud security.
        """,
    ]

    for memory in memories:
        agent.remember(
            memory,
            user_id="u_001",
        )


def run_demo() -> None:
    agent = HybridMemoryAgent(
        similarity_threshold=0.70,
        max_tokens=500,
    )

    seed_memories(agent)

    queries = [
        # 1. Vector-only retrieval
        "Tôi đã đọc gì về Kubernetes?",

        # 2. Profile context
        "Recommend đọc gì tiếp",

        # 3. Recent activity
        "Tôi đang quan tâm gì gần đây?",

        # 4. Paraphrase / vector retrieval
        "Tài liệu về tự động mở rộng hạ tầng?",

        # 5. Hybrid retrieval
        "Cho tôi summary cloud security",
    ]

    print("=" * 80)
    print("HYBRID MEMORY AGENT — DEMO")
    print("=" * 80)

    for index, query in enumerate(queries, start=1):
        print(f"\n{'=' * 80}")
        print(f"QUERY {index}")
        print(f"{'=' * 80}")

        print(f"\nUser: {query}\n")

        context = agent.recall(
            query=query,
            user_id="u_001",
        )

        print("Assembled Context:")
        print("-" * 80)
        print(context)


if __name__ == "__main__":
    run_demo()