# Data setup: Fisher trials by difficulty

Content and voice anonymization experiments use **Fisher English** trials by difficulty (i.e., `base`, `hard`, `harder`). The trial construction pipeline lives in the [speech-attribution](https://github.com/caggazzotti/speech-attribution) repository and requires an **LDC license** for Fisher ([LDC2004T19](https://catalog.ldc.upenn.edu/LDC2004T19), [LDC2005T19](https://catalog.ldc.upenn.edu/LDC2005T19)).

In addition to the transcript trial files from `speech-attribution`, the ASR pipeline
in this repo expects pre-segmented speaker utterance wavs prepared from the Fisher
speech corpora ([LDC2004S13](https://catalog.ldc.upenn.edu/LDC2004S13),
[LDC2005S13](https://catalog.ldc.upenn.edu/LDC2005S13)).

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

## Prepare utterance audio for Whisper transcription

For ASR in this repo, first prepare speaker utterance wavs from the Fisher call audio,
then run `scripts/whisper_transcribe.py`. Create them with:

```bash
python scripts/prepare_utterance_audio.py \
  --audio-root /path/to/LDC2004S13/audio \
  --audio-root /path/to/LDC2005S13/audio \
  --transcript-root /path/to/LDC2004T19/fe_03_p1_tran/data/trans \
  --transcript-root /path/to/LDC2005T19/data/trans \
  --speaker-map /path/to/LDC2004T19/fe_03_p1_tran/doc/fe_03_pindata.tbl \
  --speaker-map /path/to/LDC2005T19/doc/fe_03_pindata.tbl \
  --sph2pipe /path/to/sph2pipe \
  --output-dir data/utterance_audio
```

Pass each Fisher corpus part separately; use one `--audio-root`, one
`--transcript-root`, and one `--speaker-map` for Part 1 and another set for Part 2.
`sph2pipe` is required to decode Fisher `.sph` audio before the utterance-level cuts are
made. Install or build it separately, then pass the full path to the executable, for
example `/path/to/sph2pipe_v2.5/sph2pipe`.

### `sph2pipe` installation (required for Fisher `.sph` audio)

`sph2pipe` is not bundled with this repo and must be installed manually.

**Option 1: Build from source (recommended)**

```bash
wget https://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz
tar -xvzf sph2pipe_v2.5.tar.gz
cd sph2pipe_v2.5
make
```

This produces the `sph2pipe` executable in the current directory.

**Option 2: Using conda (if available)**

```bash
conda install -c conda-forge sph2pipe
```

**Verify installation**

```bash
./sph2pipe -h
```

**Usage in this project**

Pass the full path to the executable:

```bash
--sph2pipe /path/to/sph2pipe_v2.5/sph2pipe
```

Without `sph2pipe`, Fisher `.sph` audio cannot be decoded and preprocessing will fail.

Expected output layout:

- `{output_dir}/{call_id}/fe_03_{call_id}_{A|B}_{utt_index}_{speaker_id}.wav`
- Optional metadata TSV at `{output_dir}/metadata.tsv`

This script:

- converts each Fisher `.sph` call into channel-specific wavs using `sph2pipe`
- uses Fisher transcript timestamps to cut one wav per utterance
- maps `(call_id, channel)` to `speaker_id` using the Fisher `pindata` tables

Then set:

```yaml
utterance_audio_dir: data/utterance_audio
```

in this repo’s `config.yaml`.

## Difficulty levels

- **base**: positive = same speaker, different calls, any topic; negative = different speakers, different calls, any topic.
- **hard**: positive = same speaker, different calls, different topics; negative = different speakers, different calls, same topic.
- **harder**: positive = same speaker, different calls, different topics (same as hard); negative = different speakers, same topic, **same call** (conversation partner).

The paper reports results mainly on the **hard** test set, but the pipeline supports `base`, `hard`, and `harder`.
