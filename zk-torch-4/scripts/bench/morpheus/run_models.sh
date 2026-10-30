#!/usr/bin/env bash
# Morpheus (S&P'27) end-to-end runs of the paper's eleven models at the
# configurations of scripts/bench/run_paper_e2e.py (the previous evaluation).
#
#   ./run_models.sh [model ...]      (default: all)
#
# Per model: N_MONO (3) monolithic requests (weights opened per request)
# and deferred batches of 2 and 8 requests, each with its measured
# finalization; every proof is serialized, decoded and verified. Logs go
# to $OUT (default scripts/bench/morpheus/out). Transformers run through
# morpheus_e2e; the rest through their benchmark binaries with
# MORPHEUS_AT_RUN (the benchmark's own forward pass is proven).
set -u
# Fail closed: debug and deterministic/private-coin overrides are not paper runs.
for flag in NO_AUTH NO_VERIFY LAYOUT_ONLY LINK_ROWS MORPHEUS_UNCLAIMED MORPHEUS_COIN_SEED; do
  if [ "${!flag+x}" ]; then echo "Refusing benchmark override: $flag" >&2; exit 2; fi
done
cd "$(dirname "$0")/../../.."
OUT=${OUT:-scripts/bench/morpheus/out}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,0,1}
export MORPHEUS_GPUS=${MORPHEUS_GPUS:-4} MORPHEUS_AUTH_GPUS=${MORPHEUS_AUTH_GPUS:-4}
export PROFILE=ternary B0=11 SLOTS_LOG=${SLOTS_LOG:-17} ZK4_EXP_BITS=8 MORPHEUS_TIMING=1
# Transformers: quantized-model-like dense weights (the previous evaluation
# used zero matrices) and the exact RMS reciprocal gadget.
TW=(WEIGHTS=${WEIGHTS:-dense} ZK4_EXACT_RMS=1)
MODES=${MODES:-"mono defer"}

# The server is shared: wait until no process holds GPU memory, so a foreign
# job cannot push a run out of memory or perturb its timing.
wait_gpus() {
  [ -n "${NO_GPU_WAIT:-}" ] && return
  local said=
  # GPU_LOCK (next to this script) marks a timed foreign run, e.g. a baseline.
  while [ -e "$(dirname "$0")/GPU_LOCK" ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)" -gt 0 ]; do
    [ -z "$said" ] && echo "[$(date +%T)] waiting for free GPUs" && said=1
    sleep 60
  done
}
run() {   # name kind(e2e|hook:bin:cfg:units) env...
  local name=$1 kind=$2; shift 2
  for mode in $MODES; do
    local log="$OUT/${name}_${mode}.log"
    # Monolithic: N_MONO independent openings (fresh private randomness, so
    # the number of authentication attempts varies); the tables report the
    # mean. Deferred: measured finalizations of batches DEFER_BATCHES.
    local expected=${N_MONO:-3} batch_spec=-
    local m=(DEFER=0 N_REQUESTS=$expected)
    if [ "$mode" = defer ]; then
      batch_spec=${DEFER_BATCHES:-2,8}; expected=0
      [[ "$batch_spec" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || return 2
      local sizes j; IFS=, read -ra sizes <<< "$batch_spec"
      for j in "${sizes[@]}"; do expected=$((expected+j)); done
      m=(DEFER=1 DEFER_BATCHES=$batch_spec)
    elif [ "$mode" = mono ]; then
      # An inherited DEFER_BATCHES otherwise overrides N_REQUESTS in the driver.
      [ "${DEFER_BATCHES+x}" ] && { echo "Unset DEFER_BATCHES for mono runs" >&2; return 2; }
    else
      echo "Unknown benchmark mode: $mode" >&2; return 2
    fi
    [[ "$expected" =~ ^[1-9][0-9]*$ ]] && [ "$expected" -le 10000 ] || return 2
    # A run that fails for GPU memory while another user's job held a GPU
    # is retried (up to RETRIES times) once the GPUs are free again.
    local try=0
    while true; do
    wait_gpus
    echo "[$(date +%T)] $name $mode -> $log"
    rm -f "$log.foreign"
    # Peak summed GPU memory (MiB) over the run, sampled every 0.5 s; GPU
    # processes of other users are recorded in $log.foreign.
    ( peak=0; while true; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}'); [ "$m" -gt "$peak" ] && peak=$m && echo "$peak" > "$log.peak"
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do u=$(ps -o user= -p "$p" 2>/dev/null); [ -n "$u" ] && [ "$u" != "$USER" ] && echo "$(date +%T) $u $p" >> "$log.foreign"; done
      sleep 0.5; done ) &
    local sampler=$!
    printf "MORPHEUS_EXPECT protocol=2 profile=morpheus-ternary-v1 requests=%s batches=%s\n" "$expected" "$batch_spec" > "$log"
    case $kind in
      e2e) env "$@" "${m[@]}" timeout ${TMO:-14400} ./target/release/morpheus_e2e >> "$log" 2>&1 ;;
      hook:*) IFS=: read -r _ bin cfg units <<< "$kind"
        env "$@" "${m[@]}" MORPHEUS_AT_RUN=1 MORPHEUS_NAME="$name" MORPHEUS_UNITS="$units" timeout ${TMO:-14400} ./target/release/$bin "$cfg" >> "$log" 2>&1 ;;
    esac
    local run_status=$?
    printf "MORPHEUS_EXIT status=%s\n" "$run_status" >> "$log"
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
    echo "peak GPU memory (sum over devices) $(cat "$log.peak" 2>/dev/null) MiB" >> "$log"
    grep -E "^(prove|request stage|batch J|verify per|proof per|peak GPU)|panicked" "$log" | sed 's/^/    /'
    if [ -s "$log.foreign" ] && grep -qiE "AllocationFailed|out of memory" "$log" && [ $try -lt ${RETRIES:-2} ]; then
      try=$((try + 1))
      echo "[$(date +%T)] $name $mode: GPU memory failure while another user's job was on the GPUs ($(head -1 "$log.foreign")); retry $try"
      mv "$log" "$log.contended$try"
      continue
    fi
    break
    done
  done
}

want() { [ $# -eq 0 ] && return 0; for w in $SEL; do [ "$w" = "$1" ] && return 0; done; return 1; }
SEL="$*"
sel() { [ -z "${SEL// /}" ] && return 0; for w in $SEL; do [ "$w" = "$1" ] && return 0; done; return 1; }

sel gpt2   && run GPT-2 e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=24 MODEL=gpt2 NUM_LAYERS=12 SEQ_LEN=128 VOCAB_SIZE=50257
sel bert   && run BERT-Large e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=10 MODEL=bert NUM_LAYERS=24 SEQ_LEN=128
sel llama2 && run LLaMA2-7B e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=24 MODEL=llama2 NUM_LAYERS=32 SEQ_LEN=128 NUM_HEADS=32 HEAD_DIM=128 FFN_DIM=11008 FFN_SHARDS=1 VOCAB_SIZE=32000
sel llama3 && run LLaMA3-8B e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=24 MODEL=llama3 NUM_LAYERS=32 SEQ_LEN=128 NUM_HEADS=32 NUM_KV_HEADS=8 HEAD_DIM=128 FFN_DIM=14336 VOCAB_SIZE=128256
sel gptj   && run GPT-J-6B e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=24 MODEL=gptj NUM_LAYERS=28 SEQ_LEN=128 NUM_HEADS=16 HEAD_DIM=256 FFN_DIM=16384 VOCAB_SIZE=50400
sel resnet && run ResNet-50 hook:bench_streaming_resnet:cv_config.yaml:1 NUM_LAYERS=53 INPUT_SIZE=224 ZK4_TABLE_SIZE_LOG=12
sel vgg    && run VGG-16 hook:bench_streaming_vgg:cv_config.yaml:1 VGG_VARIANT=16 NUM_LAYERS=13 INPUT_SIZE=32 ZK4_TABLE_SIZE_LOG=12
sel yolo   && run YOLOv11n hook:bench_streaming_yolo:cv_config.yaml:1 NUM_STAGES=8 INPUT_SIZE=640 ZK4_TABLE_SIZE_LOG=12
sel unet3d && run 3D-UNet hook:bench_streaming_unet3d:cv_config.yaml:1 NUM_LAYERS=6 INPUT_D=64 INPUT_H=64 INPUT_W=64 ZK4_TABLE_SIZE_LOG=12
sel pointpillars && run PointPillars hook:bench_streaming_pointpillar:cv_config.yaml:1 NX=432 NY=496 N_PILLARS=12000 MAX_POINTS=32 NUM_ANCHORS=2 NUM_CLASSES=3 ZK4_TABLE_SIZE_LOG=12
sel whisper && run Whisper hook:bench_streaming_whisper:llama2_config.yaml:1 NUM_ENC_LAYERS=4 NUM_DEC_LAYERS=4 N_MELS=80 N_AUDIO_CTX=1500 N_TEXT_CTX=448 N_STATE=384 N_HEAD=6 ZK4_TABLE_SIZE_LOG=24

# ---- Comparisons (selected explicitly, never part of the default set) ----
# DeepProve-matched configurations (scripts/bench/deepprove): GPT-2 and
# LLaMA2-7B truncated to L layers, 16 prompt tokens, sequence 32 (16
# generated) or 128 (112 generated).
L2=(MODEL=llama2 NUM_HEADS=32 HEAD_DIM=128 FFN_DIM=11008 FFN_SHARDS=1 VOCAB_SIZE=32000)
want_x() { for w in $SEL; do [ "$w" = "$1" ] && return 0; done; return 1; }
# Monolithic only: the comparison reports one accepted proof per request.
want_x dp && MODES=mono && for s in 32 128; do
  [ $s = 32 ] && run GPT-2_s32 e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=24 MODEL=gpt2 NUM_LAYERS=12 SEQ_LEN=32 VOCAB_SIZE=50257
  for l in 2 16 32; do
    [ $l = 32 ] && [ $s = 128 ] && continue   # the main LLaMA2-7B run
    run LLaMA2-7B_L${l}_s$s e2e "${TW[@]}" ZK4_TABLE_SIZE_LOG=24 "${L2[@]}" NUM_LAYERS=$l SEQ_LEN=$s
  done
done
# DeepProve's GPU build uses one GPU, so the comparison runs Morpheus on one
# GPU, for GPT-2 only.
G1=(MORPHEUS_GPUS=1 MORPHEUS_AUTH_GPUS=1)
want_x dp1 && MODES=mono && for s in 32 128; do
  run GPT-2_s${s}_g1 e2e "${TW[@]}" "${G1[@]}" ZK4_TABLE_SIZE_LOG=24 MODEL=gpt2 NUM_LAYERS=12 SEQ_LEN=$s VOCAB_SIZE=50257
done
# EZKL's two reference circuits at their exported shapes, one GPU.
want_x ezkl && MORPHEUS_GPUS=1 MORPHEUS_AUTH_GPUS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} \
  run LeNet-5 hook:bench_streaming_lenet:llama2_config.yaml:1 MORPHEUS_GPUS=1 MORPHEUS_AUTH_GPUS=1
want_x ezkl && run nanoGPT hook:bench_streaming_nanogpt:llama2_config.yaml:64 MORPHEUS_GPUS=1 MORPHEUS_AUTH_GPUS=1 SEQ_LEN=64
# The EZKL GPU build proves nanoGPT only at sequence length 8 (seq 64 runs out
# of GPU memory in proving-key generation); the identical inference here.
want_x ezkl8 && MODES=mono run nanoGPT_s8 hook:bench_streaming_nanogpt:llama2_config.yaml:8 MORPHEUS_GPUS=1 MORPHEUS_AUTH_GPUS=1 SEQ_LEN=8
# Convolution ablation: the polynomial-product reduction with a committed
# full-convolution advice (MORPHEUS_CONV=pp) against the default.
want_x convpp && MODES=mono run VGG-16_pp hook:bench_streaming_vgg:cv_config.yaml:1 MORPHEUS_CONV=pp VGG_VARIANT=16 NUM_LAYERS=13 INPUT_SIZE=32 ZK4_TABLE_SIZE_LOG=12
want_x convpp && MODES=mono run ResNet-50_pp hook:bench_streaming_resnet:cv_config.yaml:1 MORPHEUS_CONV=pp NUM_LAYERS=53 INPUT_SIZE=224 ZK4_TABLE_SIZE_LOG=12
