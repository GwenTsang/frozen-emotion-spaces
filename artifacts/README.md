# Golden replication artifacts

Large numerical artifacts are not stored in Git. Their logical paths, byte
sizes, SHA-256 digests, and content-addressed Bucket paths are locked in
`artifacts.lock.json`.

The Bucket is a mutable transport layer, not a version-control system. The Git
lock file is authoritative. Downloads always go through a temporary file and
are installed only after their size and SHA-256 have been verified.

Roman Klinger authorized publication of the crowd-enVENT-derived replication
artifacts. The Bucket is therefore public, while the raw corpus ZIP remains at
its official source. Fetching requires no Hugging Face account:

```bash
python scripts/fetch_artifacts.py --destination artifacts/downloads/replication-20260816
```

The Bucket is optional: every experiment represented by the golden outputs can
be recomputed from the official corpora, pinned checkpoints, checked-in splits,
and the commands in the root `Makefile`.
