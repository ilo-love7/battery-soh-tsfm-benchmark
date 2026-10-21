#!/usr/bin/env bash
# 采集概率口径（CRPS / PICP / MPIW）所需的分位数预测。
#
# 产物落在 03_实验结果/prob/ 二级目录 —— report.py 只扫一级 */summary.json，
# 所以这里不会污染主表，主表的数字与图保持逐位不变。
#
# 用法：bash scripts/run_prob.sh [mit|nasa|calce|all]
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
SRC="$ROOT/01_源代码"
PY_CORE="$ROOT/venv/Scripts/python.exe"
PY_MODELS="$ROOT/venv-moirai/Scripts/python.exe"
OUT="$ROOT/03_实验结果/prob"
LOG="$OUT/_run.log"
mkdir -p "$OUT"

# 基础模型拆两组：uni2ts 硬依赖 numpy 1.26，与 chronos/timesfm 无法共存在一个 3.13 环境
CORE_MODELS="chronos-bolt,chronos,timesfm"
MOIRAI_MODELS="moirai,lag-llama"

run() {  # run <解释器> <模型串> <dataset> [extra args...]
    local py=$1 models=$2 ds=$3; shift 3
    local tag="$ds-$(echo "$models" | tr ',' '-')"
    echo "" | tee -a "$LOG"
    echo ">>> [$ds] $models  ($(date +%H:%M:%S))" | tee -a "$LOG"
    ( cd "$SRC" && "$py" run_benchmark.py --models "$models" --dataset "$ds" \
        --max-context 512 --output-dir "$OUT/$tag" "$@" ) 2>&1 | tee -a "$LOG"
}

ONLY=${1:-all}
for ds in mit nasa calce; do
    [ "$ONLY" != "all" ] && [ "$ONLY" != "$ds" ] && continue
    run "$PY_CORE"   "$CORE_MODELS"   "$ds" --split test
    run "$PY_MODELS" "$MOIRAI_MODELS" "$ds"
done
echo "" | tee -a "$LOG"
echo "== 完成 $(date) ==" | tee -a "$LOG"
