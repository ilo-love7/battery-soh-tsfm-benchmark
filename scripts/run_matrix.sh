#!/usr/bin/env bash
# 一键复现全矩阵（论文表 1–表 5 + 图 1–图 6 的原始数据）。
#
# 用法：
#   bash scripts/run_matrix.sh [all|mit|nasa|calce|sens|seed|report]
#
# 环境变量：
#   PY_CORE    跑 Chronos/TimesFM/基线/few-shot 的解释器（默认自动探测 venv）
#   PY_MODELS  跑 Moirai/Lag-Llama 的解释器（默认自动探测 venv-moirai）
# 两个变量分开是因为 uni2ts 硬依赖 numpy 1.26（无 cp313 wheel），
# 与 torch 2.14 + chronos-forecasting 无法共存在同一个 3.13 环境里。
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

detect_py() {  # $1=venv 目录名
  if [ -x "$ROOT/$1/bin/python" ]; then echo "$ROOT/$1/bin/python"
  elif [ -x "$ROOT/$1/Scripts/python.exe" ]; then echo "$ROOT/$1/Scripts/python.exe"
  else echo ""; fi
}
PY_CORE=${PY_CORE:-$(detect_py venv)}
PY_MODELS=${PY_MODELS:-$(detect_py venv-moirai)}
[ -z "$PY_CORE" ] && { echo "找不到 venv，请先运行 bash scripts/setup_env.sh"; exit 1; }

LOG_DIR="$ROOT/03_实验结果/_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/matrix_$(date +%Y%m%d_%H%M).log"
echo "== 开始 $(date) ==" | tee -a "$LOG"
echo "   PY_CORE=$PY_CORE" | tee -a "$LOG"
echo "   PY_MODELS=${PY_MODELS:-（未配置，Moirai/Lag-Llama 将跳过）}" | tee -a "$LOG"

run() {  # run <解释器> <参数...>
  local py=$1; shift
  if [ -z "$py" ]; then echo ">>> 跳过（缺解释器）：$*" | tee -a "$LOG"; return; fi
  echo "" | tee -a "$LOG"
  echo ">>> $*" | tee -a "$LOG"
  ( cd "$ROOT/01_源代码" && "$py" run_benchmark.py "$@" ) 2>&1 \
    | tee -a "$LOG" | grep -E "frac=|完成|跳过|Error|error|Traceback|MAE"
}

MODE="${1:-all}"
has() { [ "$MODE" = "all" ] || [ "$MODE" = "$1" ]; }

# ---- 表 1：MIT 主测试集（43 cells，batch2） ----
if has mit; then
  run "$PY_CORE" --model all-baselines --dataset mit --split test
  run "$PY_CORE" --model chronos-bolt --dataset mit --split test
  run "$PY_CORE" --model chronos      --dataset mit --split test
  run "$PY_CORE" --model timesfm      --dataset mit --split test
  run "$PY_MODELS" --model moirai     --dataset mit --split test
  run "$PY_MODELS" --model lag-llama  --dataset mit --split test
  # MIT 内部 OOD（batch3）
  run "$PY_CORE" --models chronos-bolt,naive_linear --dataset mit --split secondary_test
fi

# ---- 表 2/3：NASA、CALCE 跨库零样本 ----
if has nasa; then
  run "$PY_CORE"   --models chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay --dataset nasa
  run "$PY_MODELS" --models moirai,lag-llama --dataset nasa
fi
if has calce; then
  run "$PY_CORE"   --models chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay --dataset calce
  run "$PY_MODELS" --models moirai,lag-llama --dataset calce
fi

# ---- 表 4：起点敏感性 40/50/60% ----
if has sens; then
  run "$PY_CORE" --models chronos-bolt,chronos,naive_linear,exp_decay \
      --dataset mit --split test --start-fracs 0.4 0.5 0.6 --tag sens
  run "$PY_CORE" --models chronos-bolt,chronos,naive_linear,exp_decay \
      --dataset nasa --start-fracs 0.4 0.5 0.6 --tag sens
  run "$PY_CORE" --models chronos-bolt,chronos,naive_linear,exp_decay \
      --dataset calce --start-fracs 0.4 0.5 0.6 --tag sens
fi

# ---- 表 5：采样类模型的多随机种子方差（默认种子之外再跑 1/2/3） ----
# 注意：Chronos-T5 跑在 PY_CORE（chronos-forecasting 只装在 3.13 环境），
# Moirai / Lag-Llama 跑在 PY_MODELS；两者写同一个目录，summary.json 按键合并。
if has seed; then
  for s in 1 2 3; do
    run "$PY_CORE"   --model chronos --dataset mit --split test \
        --seed "$s" --output-dir "$ROOT/03_实验结果/_seeds/mit-s$s"
    run "$PY_MODELS" --models moirai,lag-llama --dataset mit --split test \
        --seed "$s" --output-dir "$ROOT/03_实验结果/_seeds/mit-s$s"
    run "$PY_CORE"   --model chronos --dataset nasa \
        --seed "$s" --output-dir "$ROOT/03_实验结果/_seeds/nasa-s$s"
    run "$PY_MODELS" --models moirai,lag-llama --dataset nasa \
        --seed "$s" --output-dir "$ROOT/03_实验结果/_seeds/nasa-s$s"
    run "$PY_CORE"   --model chronos --dataset calce \
        --seed "$s" --output-dir "$ROOT/03_实验结果/_seeds/calce-s$s"
    run "$PY_MODELS" --models moirai,lag-llama --dataset calce \
        --seed "$s" --output-dir "$ROOT/03_实验结果/_seeds/calce-s$s"
  done
fi

# ---- 汇总：结果表 + 显著性 + 机制 + 种子方差 + 图 ----
if has report; then
  echo "" | tee -a "$LOG"
  echo ">>> report / stats / figures" | tee -a "$LOG"
  ( cd "$ROOT/01_源代码" \
    && "$PY_CORE" report.py \
    && "$PY_CORE" stats_significance.py \
    && "$PY_CORE" analyze_shrinkage.py \
    && "$PY_CORE" analyze_seeds.py \
    && "$PY_CORE" figures.py ) 2>&1 | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "== 结束 $(date) ==" | tee -a "$LOG"
