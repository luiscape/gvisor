#!/usr/bin/env bash
# One checkpoint/restore cycle of an inference engine under runsc.
#
# Usage: bench.sh --engine vllm|sglang --gpus 0,1 --tp 2 [--restore-gpus 4,5]
#                 [--runsc DIR] [--no-shim] [--no-sleep] [--name NAME]
#                 [-- extra engine args]
#
# Boots the engine, records a temperature-0 completion, puts the engine to
# sleep (vLLM /sleep?level=1, SGLang release_memory_occupation), checkpoints,
# restores (onto --restore-gpus), wakes it and repeats the completion. PASS if
# the output is identical and the sandbox is on the requested GPUs.
#
# Env: MODEL, PORT, JOB (1: global --cuda-checkpoint-path), CKPT_FLAGS,
# EXTRA_ENV (newline-separated KEY=V for the container), NO_PTRACE_CAP=1,
# NUMA (1: pin to the GPUs' node), BOOT_TIMEOUT, CKPT_TIMEOUT, CURL_TIMEOUT,
# KEEP_RUNNING=1 (leave the restored sandbox up), BENCH_DIR, HF_DIR.
set -u
S=$(cd "$(dirname "$0")" && pwd)
B=${BENCH_DIR:-/data/bench}
ENGINE=vllm GPUS=0,1 TP=2 RGPUS="" RUNSC_DIR=$B/d2-bin SHIM=1 SLEEP=1 NAME=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --restore-gpus) RGPUS="$2"; shift 2;;
    --runsc) RUNSC_DIR="$2"; shift 2;;
    --no-shim) SHIM=0; shift;;
    --no-sleep) SLEEP=0; shift;;
    --name) NAME="$2"; shift 2;;
    --) shift; EXTRA_ARGS=("$@"); break;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
RGPUS=${RGPUS:-$GPUS}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
PORT=${PORT:-8000}
NAME=${NAME:-$ENGINE-tp$TP-$(date +%H%M%S)}
ID=$NAME
BASE=$B/runs/$NAME
RUNSC=$RUNSC_DIR/runsc

sudo umount -l "$BASE/merged" 2>/dev/null
sudo rm -rf "$BASE"
mkdir -p "$BASE"/{logs,applog,upper,work,merged,bundle,bundle-r,ckpt}

now() { date +%s.%N; }
dt() { echo "scale=1; ($2 - $1) / 1" | bc; }
say() { echo "[$NAME $(date +%H:%M:%S)] $*" | tee -a "$BASE/summary.txt"; }
fail() { say "RESULT: FAIL ($*)"; cleanup; exit 1; }
cleanup() {
  [ "${KEEP_RUNNING:-0}" = 1 ] && return
  R kill "$ID" KILL >/dev/null 2>&1; sleep 1; R delete -force "$ID" >/dev/null 2>&1
  sudo umount -l "$BASE/merged" 2>/dev/null
}

# numa_of GPUS: NUMA node of the first GPU in the list.
numa_of() {
  local bus
  bus=$(nvidia-smi -i "${1%%,*}" --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-F' 'a-f' | sed 's/^0000//')
  cat "/sys/bus/pci/devices/$bus/numa_node"
}
PIN=() RPIN=()
if [ "${NUMA:-1}" = 1 ]; then
  n=$(numa_of "$GPUS"); PIN=(numactl --cpunodebind="$n" --membind="$n")
  n=$(numa_of "$RGPUS"); RPIN=(numactl --cpunodebind="$n" --membind="$n")
fi

FLAGS=(--root "$BASE/runsc-root" --debug --debug-log="$BASE/logs/" --network=none
  --nvproxy --nvproxy-allowed-driver-capabilities=all)
[ "$SHIM" = 1 ] && FLAGS+=(--cuda-multicast-shim-source=EMBEDDED)
CKPT_PATH_ARG=(--cuda-checkpoint-path=/usr/local/bin/cuda-checkpoint)
if [ "${JOB:-1}" = 1 ]; then FLAGS+=(--cuda-checkpoint-path=/usr/local/bin/cuda-checkpoint); CKPT_PATH_ARG=(); fi
R() { sudo "${PIN[@]}" "$RUNSC" "${FLAGS[@]}" "$@"; }
RR() { sudo "${RPIN[@]}" "$RUNSC" "${FLAGS[@]}" "$@"; }

if [ "$ENGINE" = vllm ]; then
  # vLLM serves /sleep and /wake_up only in dev mode.
  ENGINE_ENV=VLLM_SERVER_DEV_MODE=1
  CMD="vllm serve $MODEL --host 127.0.0.1 --port $PORT --tensor-parallel-size $TP --enable-sleep-mode --gpu-memory-utilization 0.6 --max-model-len 4096 ${EXTRA_ARGS[*]:-}"
else
  ENGINE_ENV=
  CMD="python3 -m sglang.launch_server --model-path $MODEL --host 127.0.0.1 --port $PORT --tp $TP --enable-memory-saver --enable-weights-cpu-backup --mem-fraction-static 0.6 --context-length 4096 ${EXTRA_ARGS[*]:-}"
fi

bash "$S/prep_rootfs.sh" "$ENGINE" > "$BASE/logs/prep_rootfs.out" 2>&1 || { cat "$BASE/logs/prep_rootfs.out"; fail "rootfs prep"; }
sudo mount -t overlay overlay -o "lowerdir=$B/rootfs/$ENGINE,upperdir=$BASE/upper,workdir=$BASE/work" "$BASE/merged" || fail "overlay mount"
bundle() { # gpus dir
  GPUS="$1" ENVFILE="$B/rootfs/$ENGINE.env" EXTRA_ENV="$ENGINE_ENV"$'\n'"${EXTRA_ENV:-}" NO_PTRACE_CAP="${NO_PTRACE_CAP:-0}" \
    APPLOG="$BASE/applog" HF="${HF_DIR:-/data/hf}" CMD="exec $CMD > /applog/engine.log 2>&1" \
    CWD="$(cat "$B/rootfs/$ENGINE.cwd")" ROOTFS="$BASE/merged" NAME="$NAME" \
    python3 "$S/gen_bundle.py" > "$2/config.json"
}
bundle "$GPUS" "$BASE/bundle" || fail "bundle"
bundle "$RGPUS" "$BASE/bundle-r" || fail "restore bundle"

rexec() { R exec "$ID" "$@"; }
# Never redirect runsc's stdio to a fresh file: runsc chowns inherited stdio
# files to the sandbox user (root), so later appends from this shell fail.
curl_in() { rexec /usr/bin/curl -s --max-time "${CURL_TIMEOUT:-300}" "$@"; }
PROMPT='{"model":"'"$MODEL"'","prompt":"The three primary colors are","max_tokens":32,"temperature":0}'
infer() { curl_in -X POST -H 'Content-Type: application/json' -d "$PROMPT" "http://127.0.0.1:$PORT/v1/completions" | jq -r '.choices[0].text // empty'; }
health() { curl_in -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null; }
sleep_engine() {
  if [ "$ENGINE" = vllm ]; then curl_in -X POST "http://127.0.0.1:$PORT/sleep?level=1" -o /dev/null -w '%{http_code}'
  else curl_in -X POST -H 'Content-Type: application/json' -d '{}' "http://127.0.0.1:$PORT/release_memory_occupation" -o /dev/null -w '%{http_code}'; fi
}
wake_engine() {
  if [ "$ENGINE" = vllm ]; then curl_in -X POST "http://127.0.0.1:$PORT/wake_up" -o /dev/null -w '%{http_code}'
  else curl_in -X POST -H 'Content-Type: application/json' -d '{}' "http://127.0.0.1:$PORT/resume_memory_occupation" -o /dev/null -w '%{http_code}'; fi
}
wait_health() { # timeout
  local t0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$1" ]; do
    [ "$(health)" = 200 ] && return 0
    R state "$ID" 2>/dev/null | grep -q '"status": "running"' || return 1
    sleep 3
  done
  return 1
}

say "engine=$ENGINE tp=$TP gpus=$GPUS restore-gpus=$RGPUS shim=$SHIM sleep=$SLEEP runsc=$($RUNSC --version | head -n1 | awk '{print $3}') pin=${PIN[*]:-none} job=${JOB:-1} noptrace=${NO_PTRACE_CAP:-0} ckpt_flags=${CKPT_FLAGS:-none} model=$MODEL gpu=$(nvidia-smi -i 0 --query-gpu=name,driver_version --format=csv,noheader | tr -d ',') env=$(printf '%s' "${EXTRA_ENV:-}" | paste -sd' ')"
t0=$(now)
R run -detach --bundle "$BASE/bundle" "$ID" >"$BASE/logs/run.out" 2>&1 || fail "runsc run: $(tail -n 3 "$BASE/logs/run.out")"
wait_health "${BOOT_TIMEOUT:-900}" || { tail -n 20 "$BASE/applog/engine.log"; fail "engine never healthy"; }
t1=$(now); say "cold boot $(dt "$t0" "$t1")s"
REF="$(infer)"; [ -n "$REF" ] || fail "reference inference empty"
say "reference: $(printf %q "$REF" | cut -c1-80)"
if [ "$SLEEP" = 1 ]; then
  c=$(sleep_engine); [ "$c" = 200 ] || fail "sleep returned $c"
  say "engine asleep"
fi

t2=$(now)
sudo timeout "${CKPT_TIMEOUT:-900}" "${PIN[@]}" "$RUNSC" "${FLAGS[@]}" checkpoint "${CKPT_PATH_ARG[@]}" ${CKPT_FLAGS:-} --image-path "$BASE/ckpt" "$ID" >"$BASE/logs/ckpt.out" 2>&1
rc=$?; t3=$(now)
[ $rc = 0 ] || fail "checkpoint rc=$rc: $(head -c 300 "$BASE/logs/ckpt.out")"
say "checkpoint $(dt "$t2" "$t3")s image $(du -sh "$BASE/ckpt" | cut -f1)"
R delete -force "$ID" >/dev/null 2>&1

t4=$(now)
RR restore -detach --image-path "$BASE/ckpt" --bundle "$BASE/bundle-r" "$ID" >"$BASE/logs/restore.out" 2>&1 || fail "restore: $(tail -n 3 "$BASE/logs/restore.out")"
t5=$(now); say "restore returned $(dt "$t4" "$t5")s"
if [ "$SLEEP" = 1 ]; then
  c=$(wake_engine); [ "$c" = 200 ] || fail "wake returned $c"
fi
OUT="$(infer)"; t6=$(now)
say "first inference after restore $(dt "$t4" "$t6")s"
# Placement: host GPUs holding the sandbox's contexts.
SPID=$(pgrep -f "runsc-sandbox.*$ID" | head -n1)
PLACED=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', ' -v p="$SPID" '$1==p{print $2}' | while read -r b; do nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v b="$b" '$2==b{print $1}'; done | sort -n | uniq | paste -sd,)
WANT=$(echo "$RGPUS" | tr ',' '\n' | sort -n | paste -sd,)
say "restored on GPUs: ${PLACED:-?} (want $WANT)"
[ "$OUT" = "$REF" ] || fail "output mismatch: $(printf %q "$OUT" | cut -c1-80)"
[ "$PLACED" = "$WANT" ] || fail "placement mismatch"
say "RESULT: PASS (speedup to first inference $(echo "scale=1; ($t1-$t0)/($t6-$t4)" | bc)x)"
cleanup
