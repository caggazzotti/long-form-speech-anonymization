# Content Anonymization for Privacy in Long-form Audio

This is the repo for the 2025 paper ["Content Anonymization for Privacy in Long-form Audio"](https://doi.org/10.48550/arXiv.2510.12780). This paper calls attention to the previously overlooked privacy risk of identifying a speaker by the stylistic content of their speech, which builds up over multiple utterances.

Most voice anonymization techniques focus on single utterances and most content anonymization approaches focus on obscuring personally identifiable information (PII). However, throughout a conversation, even if the voice is anonymized and the PII removed, a speaker can still be identified by their speaking style and patterns. Thus, we present a new pipeline for anonymizing both a speaker's voice and the content of their speech for more comprehensive protection of a speaker's identity.

---

## Getting the Fisher data

The data comes from the Fisher English Training Speech and Transcripts corpora, which require a Linguistic Data Consortium license.

- Fisher English Training Speech audio files: [LDC2004S13](https://catalog.ldc.upenn.edu/LDC2004S13), [LDC2005S13](https://catalog.ldc.upenn.edu/LDC2005S13)
- Fisher English Training Transcript text files: [LDC2004T19](https://catalog.ldc.upenn.edu/LDC2004T19), [LDC2005T19](https://catalog.ldc.upenn.edu/LDC2005T19)

The exact calls used for verification trials are obtained from the [speech-attribution](https://github.com/caggazzotti/speech-attribution) repository. Follow **[DATA.md](DATA.md)** to build trials by difficulty level (`base`, `hard`, `harder`) with full transcripts (`trunc_style: 'none'`).

This produces two file types used in this repo:

- `ldc_{dataset}_{difficulty}_trials.npy`  
  Used for **no-anonymization content baseline** (both sides are original LDC/Fisher transcript text).
- `{dataset}_{trialtype}_trials_info_final.json` (e.g. `test_hardpos_trials_info_final.json`, `test_hardneg_trials_info_final.json`)  
  Used for **matched anonymization experiments** to map call/pin pairs when constructing call1/call2 trial sides from utterance JSON.

Point this repo to those outputs via `speech_attribution_dir` (or `ldc_trials_dir` + `trials_info_dir`) in `config.yaml`.

---

## Anonymization + Experiments

The table below shows the data used for each setting:

| | **No anonymization** | **Anonymization** |
|---|----------------------|---------------------|
| **Voice Attack** | Original **LDC/Fisher** audio on **both** sides of each trial. | **Call 1**: original Fisher audio.<br>**Call 2**: voice-anonymized audio (e.g. ASR-TTS) |
| **Content Attack** | Original **LDC/Fisher** transcripts on **both** sides of each trial. | **Call 1**: ASR transcript of original Fisher audio (e.g. Whisper).<br>**Call 2**: content-anonymized text (LLM paraphrase) |
| **Content+Voice Attack** | Original **LDC/Fisher** transcripts on **both** sides of each trial. | **Call 1**: ASR transcript (e.g. Whisper).<br>**Call 2**: voice-anonymized then transcribed (Whisper) then content-anonymized (LLM paraphrase) |

### No anonymization

1. Get Fisher outputs from **[DATA.md](DATA.md)** (`ldc_{dataset}_{difficulty}_trials.npy`).
2. Set `ldc_trials_dir` (or `speech_attribution_dir`) in config.
3. Embed and evaluate LDC baseline:

   ```bash
   python scripts/embed_trials_sluar.py config.yaml --system ldc --varyutts
   python scripts/evaluate_ldc_sluar.py config.yaml
   ```

### Voice anonymization

Use the same trial definitions and labels from **[DATA.md](DATA.md)**.

1. Use Fisher **audio** for the same calls as in the trials.
2. Run your **voice anonymization** method on that audio.
3. Transcribe outputs as needed for content-attack evaluation.
4. *Detailed voice-anonymization scripts/commands will be added here.*


### Content anonymization

Scripts live under `scripts/`. Configure paths and systems in **`config.yaml`**.

**Setup**

```bash
pip install -r requirements.txt
```

- **Python** 3.8+
- **LDC** Fisher access for trial construction ([DATA.md](DATA.md))
- **SLUAR** checkpoint on Hugging Face: [noandrews/sluar](https://huggingface.co/noandrews/sluar); set `HF_TOKEN` / `HUGGINGFACE_TOKEN` if required
- Optional: if the checkpoint needs a custom model class, set `SLUAR_PATH` to the package root so `from sluar.models import LUAR` resolves; otherwise loading uses `trust_remote_code=True`

**Content anonymization pipeline**

You can run all stages with one command:

```bash
bash scripts/run_content_pipeline.sh --all
```

Or run selected stages:

```bash
bash scripts/run_content_pipeline.sh --match --embed-matched --embed-ldc --eval
```

### Details on each stage of the pipeline:

1. **ASR transcribe Fisher audio calls**

(See Step 3 under Voice anonymization for directions) Automatically transcribe **every Fisher audio call** that appears in your trials at the chosen difficulty level (all call IDs used as call 1 or call 2). Store per-call, per-speaker utterances as JSON of the format `{call_id: {speaker_id: {"text": [str, ...], "gender": "m"|"f"}}}` in a file called `whisper_medium_test_trials_utts.json`. These will be used for the first side (call 1) of each trial.

2. **Generate paraphrase prompts**

Create prompt files from call 2 utterances according to your prompt recipe/template.

   ```bash
   python scripts/generate_paraphrase_prompts.py
   ```

3. **Run paraphrasing model/API**

Produce anonymized text for the **second side** of each trial (call 2).

Run either:
- `scripts/run_batch_paraphrase.py` for batch/API paraphrasing workflows, or
- `scripts/run_local_gemma_paraphrase.py` for local Gemma-based paraphrasing.

Use a stable `custom_id` per utterance (e.g., `callId-speakerId-time`) so outputs can be aligned.

4. **Convert paraphrase responses to utterance JSON**

Convert response files into `data/paraphrased_*_test_trials_utts.json`, where * stands for each LLM paraphrasing model name.


   ```bash
   python scripts/paraphrase_responses_to_utterances.py \
    --responses data/paraphrased_gpt4omini_responses.jsonl \
    --output data/paraphrased_gpt4omini_test_trials_utts.json \
    --normalize
   ```

5. **Match trials directly from utterances + trial definitions**

Create properly matched trials so that **call 1** uses the Whisper ASR transcript and **call 2** uses the anonymized/paraphrased text (same trial order and labels as the Fisher info files).

   ```bash
   python scripts/match_trials.py config.yaml
   ```

   Matched text-trial `.npy` files are written under **`trials/matched/`** (e.g. `whisper-gemma3-4b_test_hard_trials.npy`). Per-system and LDC trial `.npy` files stay under **`trials/`** (and **`trials/varyutts/`** when used).

   This reads:
   - `data/whisper_medium_test_trials_utts.json` (call 1 side),
   - `data/paraphrased_*_test_trials_utts.json` (call 2 side), and
   - `trials_info_dir` Fisher trial JSONs.

6. **Embed matched trials with SLUAR**

Use the content attack model, SLUAR, to create embeddings for the now matched anonymization trials.

   ```bash
   python scripts/embed_trials_sluar.py config.yaml --matched
   ```

7. **Evaluate**
  
Evaluate the matched (content anonymization) baseline:

   ```bash
   python scripts/evaluate_matched_trials.py config.yaml
   ```

   Matched evaluation outputs go under **`output/matched/`** (e.g. `SLUAR_whisper-gemma3-4b_varyuttsall_test_results.txt`). LDC baseline results are written under **`output/`**.

8. **Optional**: Calculate aligned similarity (greedy + DTW)

   ```bash
   python scripts/calculate_similarity_aligned.py config.yaml
   ```

Optional utility: `scripts/build_trials_from_utterances.py` can still generate per-system trial `.npy` files if you want them for debugging or custom analyses.


### Voice + Content Anonymization

Use the same trial definitions and labels from **[DATA.md](DATA.md)**.

1. Run the **voice anonymization** pipeline on the Fisher audio calls used in the trials.
2. Transcribe the resulting anonymized audio (same ASR setup used elsewhere for consistency).
3. Run **content anonymization** on the call 2 side (e.g., LLM paraphrasing) using the transcribed text.
4. Match call 1/call 2 sides as before and evaluate with the same embedding/evaluation steps used for content anonymization.


---



## Citation

If you use this pipeline, please cite:

```bibtex
@misc{aggazzotti2026,
  title     = {Content Anonymization for Privacy in Long-form Audio},
  author    = {Cristina Aggazzotti and Ashi Garg and Zexin Cai and Nicholas Andrews},
  year={2025},
  eprint={2510.12780},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2510.12780}, 
}
```

## License

The Fisher data is subject to the terms of the Linguistic Data Consortium.

---

**Disclaimer:** This codebase was originally written by hand for research experimentation. LLM tools available in Cursor were used later to help organize, tidy, and streamline the code for easier public use.

If you find bugs or inconsistencies, please open an issue or submit a pull request.
