# Data setup: Fisher trials by difficulty

Content and voice anonymization experiments use **Fisher English** trials by difficulty (i.e., `base`, `hard`, `harder`). The trial construction pipeline lives in the [speech-attribution](https://github.com/caggazzotti/speech-attribution) repository and requires an **LDC license** for Fisher ([LDC2004T19](https://catalog.ldc.upenn.edu/LDC2004T19), [LDC2005T19](https://catalog.ldc.upenn.edu/LDC2005T19)).

## Important difference from speech-attribution default

In the original speech-attribution setup, the first **5 utterances** of each speaker in each call are removed to reduce topic/speaker cues. For **long-form speech anonymization** we do **not** remove those utterances: we use the **full transcript** for each speaker so that content and voice anonymization are evaluated on the same data.

## How to get the correct Fisher trials (full transcript)

1. **Clone and configure speech-attribution**
   - Clone the [speech-attribution](https://github.com/caggazzotti/speech-attribution) repo.
   - Install its dependencies.

2. **Set paths and full-transcript behavior in `config.yaml`**
   - `fisher_dir1`, `fisher_dir2`: paths to LDC Fisher Part 1 and Part 2 (e.g. `.../LDC2004T19`, `.../LDC2005T19`).
   - `work_dir`: directory where speech-attribution will write outputs (e.g. `./speech-attribution` or an absolute path).
   - **Set full transcript (no removal of first 5 utterances):**
     ```yaml
     trunc_style: 'none'   # use 'none' (default in speech-attribution is 'beginning')
     trunc_size: 5         # ignored when trunc_style is 'none'
     ```
   - Keep `encodings: ['ldc']` so that LDC transcripts are produced, which are used here.

3. **Run the speech-attribution pipeline in order**
   - **Step 1 – Split datasets**
     ```bash
     python scripts/split_datasets.py config.yaml
     ```
   - **Step 2 – Create trials**
     ```bash
     python scripts/create_trials.py config.yaml
     ```
   - **Step 3 – Add transcripts to trials**
     ```bash
     python scripts/add_transcripts_to_trials.py config.yaml
     ```
     This reads LDC transcripts and, because `trunc_style: 'none'`, keeps **all** utterances (no first-5 removal). Outputs are written under `work_dir/trials_data/`.

4. **Use the trial and info files in this repo**
  - After the run you will have, for each `{dataset}` (i.e., `train`, `val`, `test`) and `{difficulty}` (`base`, `hard`, `harder`), transcript trial files:
    - `{work_dir}/trials_data/ldc_{dataset}_{difficulty}_trials.npy`
  - You also need the trial-info JSON files (produced in `trials_data/`) for building matched Whisper-anonymized pairs:
    - `{work_dir}/trials_data/{dataset}_basepos_trials_info_final.json`
    - `{work_dir}/trials_data/{dataset}_baseneg_trials_info_final.json`
    - `{work_dir}/trials_data/{dataset}_hardpos_trials_info_final.json`
    - `{work_dir}/trials_data/{dataset}_hardneg_trials_info_final.json`
    - `{work_dir}/trials_data/{dataset}_harderneg_trials_info_final.json`
    - (for `harder`, the pipeline uses `hardpos` + `harderneg`)
   - Each `.npy` file contains a list of trials; each trial is a dict:
     - `'label'`: 1 (same speaker) or 0 (different speakers)
     - `'call 1'`: list of utterance strings for the first call’s speaker
     - `'call 2'`: list of utterance strings for the second call’s speaker

  In this repo you can either:
   - Set **`speech_attribution_dir`** in config to the `work_dir` used above (scripts load from `{speech_attribution_dir}/trials_data/`), or
   - Copy needed files locally and set:
     - **`ldc_trials_dir`** for `ldc_*_trials.npy` files
     - **`trials_info_dir`** for `*_trials_info_final.json` files

## Difficulty levels

- **base**: positive = same speaker, different calls, any topic; negative = different speakers, different calls, any topic.
- **hard**: positive = same speaker, different calls, different topics; negative = different speakers, different calls, same topic.
- **harder**: positive = same speaker, different calls, different topics (same as hard); negative = different speakers, same topic, **same call** (conversation partner).

The paper reports results mainly on the **hard** test set, but the pipeline supports `base`, `hard`, and `harder`.
