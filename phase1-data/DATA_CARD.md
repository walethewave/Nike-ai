# Data Card — Yorùbá Speech Corpus (Naija Voice AI, Phase 1)

**Status:** template — fill in the blanks after running `scripts/inspect_datasets.py` at full scale and making the licensing decision in `PLAN.md`. Do not present this filled-in version as final until real numbers replace every `TBD`.

## Composition

| Source | Config | License | Utterances used | Hours used | Notes |
|---|---|---|---|---|---|
| NaijaVoices | `yoruba-batch-?` | CC BY-NC-SA 4.0 | TBD | TBD | Gated — registration required |
| Google FLEURS | `yo_ng` | CC-BY-4.0 | TBD | TBD | Read speech, news-domain |
| Mozilla Common Voice | `yo` | CC0-1.0 | TBD | TBD | Crowdsourced, variable quality |

## Splits

- **Train:** TBD
- **Validation:** TBD
- **Test (held out, speaker-disjoint):** TBD speakers / TBD utterances — list in `manifests/test_speakers_yoruba.txt`

State explicitly here whether test-set speakers appear in train for *any* source (they must not).

## Speaker demographics (from NaijaVoices `gender`/`age_range` fields where available)

- Gender distribution: TBD
- Age range distribution: TBD
- Number of unique speakers: TBD

## Domain / register

NaijaVoices carries a `phase` field whose meaning wasn't confirmed during initial research — check whether it distinguishes scripted vs. spontaneous speech, or something else, before writing this section. FLEURS is read/news-domain. Common Voice is crowdsourced read sentences, mixed quality.

## Known gaps / limitations

- Dialect coverage: TBD (check against "Voices Unheard: Yorùbá Regional Dialects", arXiv 2406.19564 — cited in `RESEARCH_OPEN_WEIGHT_ASR.md`)
- Domain skew: TBD (is this corpus mostly read speech? conversational speech is what Phase 3 actually needs at inference time — a train/inference domain mismatch is worth stating plainly)
- Tone/diacritic annotation consistency across sources: TBD (check a sample manually — inconsistent diacritic use across sources will hurt fine-tuning; see `phase2-asr/scripts/normalize.py`)

## Licensing summary

State the Phase 1 decision from `PLAN.md` here in one sentence, e.g.: *"This corpus mixes CC BY-NC-SA (NaijaVoices) and CC0/CC-BY data; models trained on the full mix are released non-commercially; a second permissive-only variant is trained on FLEURS+Common Voice for commercial use."*
