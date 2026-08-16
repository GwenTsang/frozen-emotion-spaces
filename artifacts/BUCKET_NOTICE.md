# Frozen Emotion Spaces replication artifacts

This public Bucket stores content-addressed numerical artifacts for release
`replication-r1`. It is not a versioned or immutable archive.

The authoritative logical-path, byte-size, and SHA-256 registry is the
`artifacts/artifacts.lock.json` file committed with the corresponding source
release. Never consume an object by a `latest` alias and never trust a Bucket
sync without verifying the Git-locked SHA-256.

The raw corpus ZIP files are not stored here. Publication of the
crowd-enVENT-derived artifacts was authorized by Roman Klinger; downstream
users must still cite the dataset and follow its source terms.
