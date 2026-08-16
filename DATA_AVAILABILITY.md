# Data availability and terms

The raw corpora are intentionally not committed to this repository. Download
them from their official distributors, review their terms, and place them at
the paths below (or override the Make variables).

| Corpus | Official source | Expected local path | SHA-256 |
|---|---|---|---|
| crowd-enVENT 2023 | <https://www.ims.uni-stuttgart.de/data/appraisalemotion> | `datasets/crowd-enVent2023.zip` | `8e5b8379aa137124d985f817661fcff5fcede537363798e4e2824f06bd2b746b` |
| EmoTwiCS v1 | <https://github.com/SofieLabat/EmoTwiCS-data> | `datasets/EmoTwiCS_v1.zip` | `4b458b7d17e8124dc94ff677b4d2517c44bb1d4d5e063b944e6210b68825c081` |

EmoTwiCS is distributed under CC BY-NC-SA 4.0. The crowd-enVENT release asks
users to cite Troiano, Oberländer, and Klinger (2023).

The eight preserved split tables under [`splits/`](splits/) contain only item,
group, and fold identifiers—no corpus text. They are included because they are
the confirmatory split lock and because the reconstruction regenerates them
cell for cell from the two official archives. Their file hashes are recorded in
[`splits/SHA256SUMS`](splits/SHA256SUMS).

Verify downloaded inputs before computing:

```bash
sha256sum datasets/crowd-enVent2023.zip datasets/EmoTwiCS_v1.zip
sha256sum --check splits/SHA256SUMS
```

Pinned model revisions are declared in
[`src/frozen_emotion_spaces/config.py`](src/frozen_emotion_spaces/config.py).
The Hugging Face model licenses and terms continue to apply.
