#!/usr/bin/env bash
# Usage: gate.sh <runsc-dir> <prefix> [cell...]
# Runs bench cells sequentially; writes $BENCH_DIR/<prefix>-gate.txt.
S=$(cd "$(dirname "$0")" && pwd)
B=${BENCH_DIR:-/data/bench}
RUNSC_DIR="$1"; P="$2"; shift 2
OUT=$B/$P-gate.txt
: > "$OUT"
settle() {
  for _ in $(seq 1 30); do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
    [ -z "$pids" ] && break
    for p in $pids; do sudo kill -9 "$p" 2>/dev/null; done; sleep 3
  done
  for m in $(mount | grep "$B/runs/" | awk '{print $3}'); do sudo umount -l "$m" 2>/dev/null; done
}
cell() { # name engine gpus tp rgpus [extra...]
  local n="$1" e="$2" g="$3" t="$4" r="$5"; shift 5
  settle
  sudo rm -f "$B/$P-$n.out"
  timeout 2400 env ${CELL_MODEL:+MODEL=$CELL_MODEL} bash "$S/bench.sh" --engine "$e" --gpus "$g" --tp "$t" --restore-gpus "$r" --runsc "$RUNSC_DIR" --name "$P-$n" -- "$@" > "$B/$P-$n.out" 2>&1
  local v; v=$(grep -o 'RESULT: [A-Z]*.*' "$B/runs/$P-$n/summary.txt" 2>/dev/null | tail -n1)
  local fd; fd=$(sudo cat $B/runs/$P-$n/logs/*boot* 2>/dev/null | grep -c 'allocation class 0x000000fd')
  local t; t=$(grep -E 'checkpoint [0-9.]+s|restore returned|first inference' "$B/runs/$P-$n/summary.txt" 2>/dev/null | sed -E 's/.*\] //' | paste -sd'|')
  local dm; dm=$(sudo cat $B/runs/$P-$n/logs/*boot* 2>/dev/null | grep -c 'cuda-checkpoint device map:')
  echo "$n | ${v:-RESULT: NONE} | devmap=$dm 0xfd=$fd | $t" | tee -a "$OUT"
  # KEEP_CKPT=0: drop the image of a passing run (failed runs keep theirs).
  if [ "${KEEP_CKPT:-1}" = 0 ] && echo "$v" | grep -q 'RESULT: PASS'; then sudo rm -rf "$B/runs/$P-$n/ckpt"; fi
}
CELLS="${*:-vllm_tp2 vllm_tp4 vllm_tp2_xgpu sglang_tp4 sglang_tp4_nvls sglang_tp4_fusion_xgpu sglang_tp4_symm}"
for c in $CELLS; do
  case $c in
    vllm_tp2) cell $c vllm 0,1 2 0,1;;
    vllm_tp4) cell $c vllm 0,1,2,3 4 0,1,2,3;;
    vllm_tp2_xgpu) cell $c vllm 0,1 2 4,5;;
    vllm_tp8) CELL_MODEL=Qwen/Qwen2.5-3B-Instruct cell $c vllm 0,1,2,3,4,5,6,7 8 0,1,2,3,4,5,6,7;;
    sglang_tp8) CELL_MODEL=Qwen/Qwen2.5-3B-Instruct cell $c sglang 0,1,2,3,4,5,6,7 8 0,1,2,3,4,5,6,7;;
    sglang_tp4) cell $c sglang 0,1,2,3 4 0,1,2,3;;
    sglang_tp4_nvls) cell $c sglang 0,1,2,3 4 0,1,2,3 --enable-nccl-nvls;;
    sglang_tp4_fusion_xgpu) cell $c sglang 0,1,2,3 4 4,5,6,7 --enable-flashinfer-allreduce-fusion --flashinfer-allreduce-fusion-backend trtllm;;
    sglang_tp4_symm) cell $c sglang 0,1,2,3 4 0,1,2,3 --enable-torch-symm-mem;;
    vllm_tp4_xgpu) cell $c vllm 0,1,2,3 4 4,5,6,7;;
    vllm_tp2_overlap) cell $c vllm 0,1 2 1,2;;
    sglang_tp4_nvls_xgpu) cell $c sglang 0,1,2,3 4 4,5,6,7 --enable-nccl-nvls;;
    sglang_tp4_symm_xgpu) cell $c sglang 0,1,2,3 4 4,5,6,7 --enable-torch-symm-mem;;
    sglang_tp4_xgpu) cell $c sglang 0,1,2,3 4 4,5,6,7;;
    sglang_tp4_fusion) cell $c sglang 0,1,2,3 4 0,1,2,3 --enable-flashinfer-allreduce-fusion --flashinfer-allreduce-fusion-backend trtllm;;
    *) echo "$c | unknown cell" | tee -a "$OUT";;
  esac
done
settle
echo GATE_DONE >> "$OUT"
