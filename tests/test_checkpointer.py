"""
The bounded checkpointer (agent/graph.py TrimmedMemorySaver): a thread keeps only its last
MAX_CHECKPOINTS checkpoints — every turn resumes from the latest one — and the writes and
channel blobs of the dropped ones go with them.
"""

from agent.graph import MAX_CHECKPOINTS, TrimmedMemorySaver

CONFIG = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}


def put_one(saver: TrimmedMemorySaver, i: int) -> None:
    checkpoint = {"v": 1, "id": f"{i:032d}", "ts": str(i),
                  "channel_values":   {"messages": list(range(i + 1))},
                  "channel_versions": {"messages": f"{i:032d}"},
                  "versions_seen":    {}}
    saver.put(CONFIG, checkpoint, {"step": i}, {"messages": f"{i:032d}"})
    saver.put_writes({"configurable": {**CONFIG["configurable"], "checkpoint_id": f"{i:032d}"}},
                     [("messages", i)], task_id=str(i))


def test_a_thread_keeps_only_its_last_checkpoints():
    saver = TrimmedMemorySaver()
    for i in range(MAX_CHECKPOINTS + 10):
        put_one(saver, i)

    checkpoints = saver.storage["t"][""]
    assert len(checkpoints) == MAX_CHECKPOINTS
    assert min(checkpoints) == f"{10:032d}"   # the 10 oldest are gone

    # The latest checkpoint still loads whole, its values included: the next turn resumes from it
    latest = saver.get_tuple(CONFIG)
    assert latest.checkpoint["id"] == f"{MAX_CHECKPOINTS + 9:032d}"
    assert latest.checkpoint["channel_values"]["messages"] == list(range(MAX_CHECKPOINTS + 10))

    # The writes and blobs of the dropped checkpoints are gone, the referenced blobs kept
    assert {key[2] for key in saver.writes} == {f"{i:032d}" for i in range(10, MAX_CHECKPOINTS + 10)}
    assert {key[3] for key in saver.blobs} == {f"{i:032d}" for i in range(10, MAX_CHECKPOINTS + 10)}


def test_a_blob_still_referenced_by_a_kept_checkpoint_survives():
    # A channel written once keeps its old version in every later checkpoint: its blob must stay
    saver = TrimmedMemorySaver()
    first = {"v": 1, "id": f"{0:032d}", "ts": "0",
             "channel_values":   {"messages": ["q"], "selected_db": "bank"},
             "channel_versions": {"messages": f"{0:032d}", "selected_db": "v-db"},
             "versions_seen":    {}}
    saver.put(CONFIG, first, {"step": 0}, {"messages": f"{0:032d}", "selected_db": "v-db"})
    for i in range(1, MAX_CHECKPOINTS + 5):
        checkpoint = {"v": 1, "id": f"{i:032d}", "ts": str(i),
                      "channel_values":   {"messages": list(range(i + 1))},
                      "channel_versions": {"messages": f"{i:032d}", "selected_db": "v-db"},
                      "versions_seen":    {}}
        saver.put(CONFIG, checkpoint, {"step": i}, {"messages": f"{i:032d}"})

    assert ("t", "", "selected_db", "v-db") in saver.blobs
    latest = saver.get_tuple(CONFIG)
    assert latest.checkpoint["channel_values"]["selected_db"] == "bank"
