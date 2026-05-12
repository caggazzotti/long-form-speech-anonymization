#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Run content-anonymization pipeline stages in order.

Usage:
  bash scripts/content_anonymization/run_content_pipeline.sh [options]

Options:
  -c, --config PATH          Config YAML (default: config.yaml)
  --systems CSV              Systems to build (default: from config.systems, excluding ldc)
  --gen-prompts              Generate batch paraphrase prompts JSONL
  --run-batch                Submit + monitor paraphrase batch job
  --run-gemma-local          Run local Gemma paraphrasing from prompt JSONL
  --retry-failed             Build retry prompt JSONL from failed batch responses
  --paraphrase-to-utts       Convert paraphrase responses JSONL to utterance JSON
  --all                      Run all main stages (match -> embed -> eval; plus LDC embed)
  --build                    Optional: build per-system text trials from utterance JSON
  --match                    Build matched text trials (Whisper call1 + anonymized call2)
  --embed-matched            Embed matched text trials with SLUAR
  --embed-ldc                Embed LDC baseline trials with SLUAR (--system ldc --varyutts)
  --eval                     Run matched and LDC evaluation
  -h, --help                 Show this help

Examples:
  # LLM paraphrase prep/run
  # GPT-4o-mini utterance-by-utterance
  PARAPHRASE_RECIPE=gpt4o-mini PARAPHRASE_MODEL=gpt-4o-mini \
  PARAPHRASE_UTTS=data/whisper_medium_test_trials_utts.json \
  PARAPHRASE_PROMPTS=data/paraphrase_gpt4omini_prompts.jsonl \
  bash scripts/content_anonymization/run_content_pipeline.sh --gen-prompts

  # GPT-5 segment-based (default recipe: ~300-token segments, no previous-utterance context)
  PARAPHRASE_RECIPE=gpt5 PARAPHRASE_MODEL=gpt-5 \
  PARAPHRASE_PROMPTS=data/paraphrase_gpt5_prompts.jsonl \
  bash scripts/content_anonymization/run_content_pipeline.sh --gen-prompts

  # Gemma segment-based (default recipe: 16 utterances with previous N=8 utterance context)
  PARAPHRASE_RECIPE=gemma PARAPHRASE_MODEL=google/gemma-3-4b-it \
  PARAPHRASE_PROMPTS=data/paraphrase_gemma_prompts.jsonl \
  bash scripts/content_anonymization/run_content_pipeline.sh --gen-prompts --run-gemma-local

  # Gemma conservative variant (same segmentation/context, conservative prompt template)
  PARAPHRASE_RECIPE=gemma-conservative PARAPHRASE_MODEL=google/gemma-3-4b-it \
  PARAPHRASE_PROMPTS=data/paraphrase_gemma4b_conservative_prompts.jsonl \
  bash scripts/content_anonymization/run_content_pipeline.sh --gen-prompts --run-gemma-local

  PARAPHRASE_UTTS=data/whisper_medium_test_trials_utts.json \
  PARAPHRASE_PROMPTS=data/paraphrase_gpt4omini_prompts.jsonl \
  bash scripts/content_anonymization/run_content_pipeline.sh --gen-prompts
  PARAPHRASE_PROMPTS=data/paraphrase_gpt4omini_prompts.jsonl \
  PARAPHRASE_RESPONSES=data/paraphrased_gpt4omini_responses.jsonl \
  PARAPHRASE_ERRORS=data/paraphrased_gpt4omini_errors.jsonl \
  bash scripts/content_anonymization/run_content_pipeline.sh --run-batch
  PARAPHRASE_PROMPTS=data/paraphrase_gpt4omini_prompts.jsonl \
  PARAPHRASE_ERRORS=data/paraphrased_gpt4omini_errors.jsonl \
  PARAPHRASE_RETRY_PROMPTS=data/paraphrase_gpt4omini_retry.jsonl \
  bash scripts/content_anonymization/run_content_pipeline.sh --retry-failed
  PARAPHRASE_RESPONSES=data/paraphrased_gpt4omini_responses.jsonl \
  PARAPHRASE_UTTS_OUT=data/paraphrased_gpt4omini_test_trials_utts.json \
  bash scripts/content_anonymization/run_content_pipeline.sh --paraphrase-to-utts

  # Core content attack pipeline
  bash scripts/content_anonymization/run_content_pipeline.sh --all
  bash scripts/content_anonymization/run_content_pipeline.sh --match --embed-matched
  bash scripts/content_anonymization/run_content_pipeline.sh --systems "whisper_medium,paraphrased_gpt4omini" --all
EOF
}

CONFIG="${ROOT_DIR}/config.yaml"
SYSTEMS_CSV=""
DO_BUILD=0
DO_GEN_PROMPTS=0
DO_RUN_BATCH=0
DO_RUN_GEMMA_LOCAL=0
DO_RETRY_FAILED=0
DO_PARAPHRASE_TO_UTTS=0
DO_MATCH=0
DO_EMBED_MATCHED=0
DO_EMBED_LDC=0
DO_EVAL=0
ANY_STAGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG="$2"; shift 2;;
    --systems)
      SYSTEMS_CSV="$2"; shift 2;;
    --all)
      DO_MATCH=1; DO_EMBED_MATCHED=1; DO_EMBED_LDC=1; DO_EVAL=1; ANY_STAGE=1; shift;;
    --build)
      DO_BUILD=1; ANY_STAGE=1; shift;;
    --gen-prompts)
      DO_GEN_PROMPTS=1; ANY_STAGE=1; shift;;
    --run-batch)
      DO_RUN_BATCH=1; ANY_STAGE=1; shift;;
    --run-gemma-local)
      DO_RUN_GEMMA_LOCAL=1; ANY_STAGE=1; shift;;
    --retry-failed)
      DO_RETRY_FAILED=1; ANY_STAGE=1; shift;;
    --paraphrase-to-utts)
      DO_PARAPHRASE_TO_UTTS=1; ANY_STAGE=1; shift;;
    --match)
      DO_MATCH=1; ANY_STAGE=1; shift;;
    --embed-matched)
      DO_EMBED_MATCHED=1; ANY_STAGE=1; shift;;
    --embed-ldc)
      DO_EMBED_LDC=1; ANY_STAGE=1; shift;;
    --eval)
      DO_EVAL=1; ANY_STAGE=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1;;
  esac
done

if [[ $ANY_STAGE -eq 0 ]]; then
  DO_MATCH=1; DO_EMBED_MATCHED=1; DO_EMBED_LDC=1; DO_EVAL=1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

if [[ -z "$SYSTEMS_CSV" ]]; then
  SYSTEMS_CSV="$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
systems = [s for s in cfg.get("systems", []) if s != "ldc"]
print(",".join(systems))
PY
)"
fi

IFS=',' read -r -a SYSTEMS <<< "$SYSTEMS_CSV"

log_stage() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "${ROOT_DIR}/${p}"
  fi
}

# Paraphrasing stage defaults (overridable via env vars)
PARAPHRASE_UTTS="${PARAPHRASE_UTTS:-data/whisper_medium_test_trials_utts.json}"
PARAPHRASE_PROMPTS="${PARAPHRASE_PROMPTS:-data/paraphrase_gpt4omini_prompts.jsonl}"
PARAPHRASE_RESPONSES="${PARAPHRASE_RESPONSES:-data/paraphrased_gpt4omini_responses.jsonl}"
PARAPHRASE_ERRORS="${PARAPHRASE_ERRORS:-data/paraphrased_gpt4omini_errors.jsonl}"
PARAPHRASE_RETRY_PROMPTS="${PARAPHRASE_RETRY_PROMPTS:-data/paraphrase_gpt4omini_retry_prompts.jsonl}"
PARAPHRASE_MODEL="${PARAPHRASE_MODEL:-gpt-4o-mini}"
PARAPHRASE_RECIPE="${PARAPHRASE_RECIPE:-custom}"  # custom|gpt4o-mini|gpt5|gemma|gemma-conservative
PARAPHRASE_ENDPOINT_URL="${PARAPHRASE_ENDPOINT_URL:-/v1/chat/completions}"
PARAPHRASE_BATCH_ENDPOINT="${PARAPHRASE_BATCH_ENDPOINT:-/v1/chat/completions}"
PARAPHRASE_POLL_SECONDS="${PARAPHRASE_POLL_SECONDS:-60}"
PARAPHRASE_BATCH_ID_FILE="${PARAPHRASE_BATCH_ID_FILE:-}"
PARAPHRASE_SEGMENT_SEPARATOR="${PARAPHRASE_SEGMENT_SEPARATOR:-##}"
PARAPHRASE_UTTS_OUT="${PARAPHRASE_UTTS_OUT:-data/paraphrased_gpt4omini_test_trials_utts.json}"
PARAPHRASE_NORMALIZE="${PARAPHRASE_NORMALIZE:-1}"
GEMMA_MODEL_ID="${GEMMA_MODEL_ID:-google/gemma-3-4b-it}"
GEMMA_MAX_NEW_TOKENS="${GEMMA_MAX_NEW_TOKENS:-512}"
GEMMA_TEMPERATURE="${GEMMA_TEMPERATURE:-0.0}"
GEMMA_TOP_P="${GEMMA_TOP_P:-1.0}"

if [[ $DO_GEN_PROMPTS -eq 1 ]]; then
  log_stage "Stage: Generate batch paraphrase prompts"
  python "${ROOT_DIR}/scripts/content_anonymization/generate_paraphrase_prompts.py" \
    --utterances "$(resolve_path "$PARAPHRASE_UTTS")" \
    --output "$(resolve_path "$PARAPHRASE_PROMPTS")" \
    --model "$PARAPHRASE_MODEL" \
    --recipe "$PARAPHRASE_RECIPE" \
    --endpoint-url "$PARAPHRASE_ENDPOINT_URL"
fi

if [[ $DO_RUN_BATCH -eq 1 ]]; then
  log_stage "Stage: Run paraphrase batch job"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required for --run-batch" >&2
    exit 1
  fi

  BATCH_ID_ARGS=()
  if [[ -n "$PARAPHRASE_BATCH_ID_FILE" ]]; then
    BATCH_ID_ARGS+=(--batch-id-out "$(resolve_path "$PARAPHRASE_BATCH_ID_FILE")")
  fi

  python "${ROOT_DIR}/scripts/content_anonymization/run_batch_paraphrase.py" \
    --prompts "$(resolve_path "$PARAPHRASE_PROMPTS")" \
    --endpoint "$PARAPHRASE_BATCH_ENDPOINT" \
    --poll-seconds "$PARAPHRASE_POLL_SECONDS" \
    --responses-out "$(resolve_path "$PARAPHRASE_RESPONSES")" \
    --errors-out "$(resolve_path "$PARAPHRASE_ERRORS")" \
    "${BATCH_ID_ARGS[@]}"
fi

if [[ $DO_RUN_GEMMA_LOCAL -eq 1 ]]; then
  log_stage "Stage: Run local Gemma paraphrase job"
  python "${ROOT_DIR}/scripts/content_anonymization/run_local_gemma_paraphrase.py" \
    --prompts "$(resolve_path "$PARAPHRASE_PROMPTS")" \
    --output "$(resolve_path "$PARAPHRASE_RESPONSES")" \
    --model-id "$GEMMA_MODEL_ID" \
    --max-new-tokens "$GEMMA_MAX_NEW_TOKENS" \
    --temperature "$GEMMA_TEMPERATURE" \
    --top-p "$GEMMA_TOP_P"
fi

if [[ $DO_RETRY_FAILED -eq 1 ]]; then
  log_stage "Stage: Build retry prompt JSONL from failed rows"
  python "${ROOT_DIR}/scripts/content_anonymization/retry_failed_batch_rows.py" \
    --prompts "$(resolve_path "$PARAPHRASE_PROMPTS")" \
    --responses "$(resolve_path "$PARAPHRASE_ERRORS")" \
    --output "$(resolve_path "$PARAPHRASE_RETRY_PROMPTS")"
fi

if [[ $DO_PARAPHRASE_TO_UTTS -eq 1 ]]; then
  log_stage "Stage: Convert paraphrase responses to utterance JSON"
  PARSE_ARGS=(
    --responses "$(resolve_path "$PARAPHRASE_RESPONSES")"
    --output "$(resolve_path "$PARAPHRASE_UTTS_OUT")"
    --segment-separator "$PARAPHRASE_SEGMENT_SEPARATOR"
  )
  if [[ "$PARAPHRASE_NORMALIZE" == "1" ]]; then
    PARSE_ARGS+=(--normalize)
  fi
  python "${ROOT_DIR}/scripts/content_anonymization/paraphrase_responses_to_utterances.py" "${PARSE_ARGS[@]}"
fi

if [[ $DO_BUILD -eq 1 ]]; then
  log_stage "Stage: Build text trials from utterance JSON"
  for system in "${SYSTEMS[@]}"; do
    if [[ -n "$system" ]]; then
      echo "Building trials for system: $system"
      python "${ROOT_DIR}/scripts/content_anonymization/build_trials_from_utterances.py" "$CONFIG" --system "$system"
    fi
  done
fi

if [[ $DO_MATCH -eq 1 ]]; then
  log_stage "Stage: Match text trials (Whisper + anonymized)"
  python "${ROOT_DIR}/scripts/content_anonymization/match_trials.py" "$CONFIG"
fi

if [[ $DO_EMBED_MATCHED -eq 1 ]]; then
  log_stage "Stage: Embed matched text trials with SLUAR"
  python "${ROOT_DIR}/scripts/content_anonymization/embed_trials_sluar.py" "$CONFIG" --matched
fi

if [[ $DO_EMBED_LDC -eq 1 ]]; then
  log_stage "Stage: Embed LDC baseline trials with SLUAR"
  python "${ROOT_DIR}/scripts/content_anonymization/embed_trials_sluar.py" "$CONFIG" --system ldc --varyutts
fi

if [[ $DO_EVAL -eq 1 ]]; then
  log_stage "Stage: Evaluate matched and LDC baselines"
  python "${ROOT_DIR}/scripts/content_anonymization/evaluate_matched_trials.py" "$CONFIG"
  python "${ROOT_DIR}/scripts/content_anonymization/evaluate_ldc_sluar.py" "$CONFIG"
fi

echo
echo "Pipeline complete."
