#!/usr/bin/env bash
# 一键复现全矩阵（论文表 1–表 5 + 图 1–图 6 的原始数据）。
#
# 用法：
#   bash scripts/run_matrix.sh [all|mit|nasa|calce|sens|seed|extra|report]
#
#   all    = 以上全部（约 10 小时，CPU）
#   extra  = few-shot 微调评测 + naive_linear 拟合窗口消融（几分钟，不用跑模型推理）
#   report = 只跑汇总与出图（要有前面的产物）
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
# ⚠️ 目录名是下游所有脚本的"事实标准"（figures.py --runs / analyze_seeds.MAIN /
#    analyze_shrinkage.RUNS / plot_fewshot.RUNS 都按名字取数），改模型集就必须
#    同步改那四处，否则图会静默少画几根线。2026-10-21 审计发现：早先这里跑的是
#    四个单模型目录，主表真正引用的 `mit-zero-...-ctx512` 从来没被生成过，
#    别人照着脚本复现会得到一份缺 MIT 主表的结果。
if has mit; then
  # 主表（公平对照：所有模型上下文统一夹到 512）
  run "$PY_CORE" --models chronos-bolt,chronos,timesfm,naive_linear \
      --dataset mit --split test --max-context 512 --tag ctx512
  # 模型须在独立环境/单独跑出来的，目录名固定在 MAIN 里
  run "$PY_MODELS" --model moirai    --dataset mit --split test
  run "$PY_MODELS" --model lag-llama --dataset mit --split test
  run "$PY_CORE" --model arima      --dataset mit --split test
  run "$PY_CORE" --models naive_last,naive_linear,exp_decay \
      --dataset mit --split test --tag refactor
  # 原生上下文对照（Chronos-Bolt 2048 vs 512，论文表 1 的附注）
  run "$PY_CORE" --model chronos-bolt --dataset mit --split test
  run "$PY_CORE" --model chronos      --dataset mit --split test
  run "$PY_CORE" --model timesfm      --dataset mit --split test
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
# ⚠️ 模型集必须与 SENS_RUNS 指向的目录名逐字一致：NASA/CALCE 带 chronos，
#    MIT 不带（MIT 的 sens 目录是 mit-zero-chronos-bolt,naive_linear,exp_decay-sens）。
#    早先 MIT 也带了 chronos，生成的目录名多一段，fig4 直接跳过不画。
if has sens; then
  run "$PY_CORE" --models chronos-bolt,naive_linear,exp_decay \
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

# ---- 表 6：few-shot 微调 + naive_linear 拟合窗口消融 ----
# 两者都只依赖 run_benchmark/fewshot_finetune/ablate_naive_linear，不重跑推理模型。
if has extra; then
  for tag_k in "ft1:1" "ft4:4" "ft8:8" "ftall:41"; do
    tag=${tag_k%%:*}; k=${tag_k##*:}
    ( cd "$ROOT/01_源代码" && "$PY_CORE" fewshot_finetune.py \
        --k "$k" --steps 300 --out "$tag" ) 2>&1 | tee -a "$LOG"
  done
  # 微调权重评测：必须用 CHRONOS_MODEL_ID 注入，目录固定为 fewshot_eval/
  # （report.py 只扫一级目录，放这里才不会覆盖主表里的零样本 chronos-bolt 行）
  eval_ft() {  # eval_ft <tag>
    ( cd "$ROOT/01_源代码" && \
      CHRONOS_MODEL_ID="amazon/chronos-bolt-small-$1" "$PY_CORE" run_benchmark.py \
        --model chronos-bolt --dataset mit --split test --max-context 512 \
        --output-dir "$ROOT/03_实验结果/fewshot_eval/mit-zero-chronos-bolt-$1" ) \
      2>&1 | tee -a "$LOG"
  }
  for t in ft1 ft4 ft8 ftall; do eval_ft "$t"; done
  # 跨库迁移（MIT 微调 → CALCE / NASA，不再训练）
  for ds in calce nasa; do
    ( cd "$ROOT/01_源代码" && \
      CHRONOS_MODEL_ID="amazon/chronos-bolt-small-ftall" "$PY_CORE" run_benchmark.py \
        --model chronos-bolt --dataset "$ds" --max-context 512 \
        --output-dir "$ROOT/03_实验结果/fewshot_eval/xfer-$ds-ftall" ) \
      2>&1 | tee -a "$LOG"
  done
  # naive_linear 拟合窗口消融（本文 ⌊n/2⌋ vs 整段），同时与已发表数字对账
  ( cd "$ROOT/01_源代码" && "$PY_CORE" ablate_naive_linear.py ) 2>&1 | tee -a "$LOG"
fi

# ---- 汇总：结果表 + 显著性 + 机制 + 种子方差 + 图 ----
# 注意：各步独立执行，不串成 && 链——diag_curvature 需要原始数据，
# 而 report/figures 只读 03_实验结果，缺数据时不该被前一步拖死。
if has report; then
  an() {
    echo "" | tee -a "$LOG"
    echo ">>> $*" | tee -a "$LOG"
    ( cd "$ROOT/01_源代码" && "$PY_CORE" "$@" ) 2>&1 | tee -a "$LOG"
  }
  # 无第三方依赖的回归测试（锁死缓存指纹的四个坑 + 在线代理的特征口径），失败不影响后续汇总
  an tests/test_cache_fingerprint.py
  an tests/test_online_proxy.py
  an tests/test_conformal_w.py
  an report.py
  an stats_significance.py
  an analyze_shrinkage.py
  an analyze_seeds.py
  # 概率口径 / EOL 重设计 / 校准头三方对比（§9.1–9.3）。
  # 只读 03_实验结果/prob 与 03_实验结果/calib，不重跑推理；
  # 缺产物时脚本自己会给出"先跑 run_prob.sh / run_cal.sh"的提示，不拖死后续汇总。
  an analyze_eol2.py
  an analyze_calibration.py
  # 区间宽度干预（§10.8）：绕中位数缩放半宽，零模型重跑，只读 prob/*.npz。
  # 产物是 03_实验结果/width_calib/ 与 04_文档报告/区间宽度校准.md。
  an calibrate_width.py
  # 在线代理（§9.5）：只用上下文特征 + 已落盘预测，零推理。产物在
  # 03_实验结果/online_proxy/ 与 04_文档报告/在线代理.md。
  an online_proxy.py
  # 上下文内自测（§9.5 最后一条路）：report 段只重建文档，读已提交的
  # per_cell_all.csv + 已落盘分位数，不重跑伪任务推理（推理用
  # `python 01_源代码/conformal_w.py` 单独跑一次即可，产物已在仓库里）。
  an conformal_w.py --skip-inference
  # 上下文范围核查（§9.4 可行性）：零推理，只读原始数据。产物在
  # 03_实验结果/context_range.csv + context_truncation.csv + 04_文档报告/上下文范围.md。
  an diag_context_range.py
  an diag_curvature.py --dataset mit --split test
  an diag_curvature.py --dataset nasa
  an diag_curvature.py --dataset calce
  an ablate_naive_linear.py
  an plot_fewshot.py
  an figures.py --prefix mit_ \
     --runs "mit-zero-chronos-bolt,chronos,timesfm,naive_linear-ctx512;mit-zero-moirai;mit-zero-lag-llama;mit-zero-naive_last,naive_linear,exp_decay-refactor;mit-zero-arima"
  an figures.py --prefix nasa_ \
     --runs "nasa-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay;nasa-zero-moirai;nasa-zero-lag-llama"
  an figures.py --prefix calce_ \
     --runs "calce-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay;calce-zero-moirai;calce-zero-lag-llama"
fi

echo "" | tee -a "$LOG"
echo "== 结束 $(date) ==" | tee -a "$LOG"
