#!/usr/bin/env bash
# 评测校准头（§9.3）：冻结 chronos-bolt + 2 参数 (s, b)。
#
# 为什么要写两份产物：
#   - 一级目录 03_实验结果/<ds>-chronos-bolt-cal/  → 进主表（report.py 只扫一级），
#     与零样本 5 模型 + 4 基线同表，做三方对比；
#   - 二级目录 03_实验结果/prob/<ds>-chronos-bolt-cal/ → 给 analyze_eol2.py 用，
#     看同一组 (s, b) 对 CRPS/PICP/MPIW 与 EOL 重设计三量的影响。
#
# ⚠️ 前置：必须先拟合，否则 load_cal_params() 会显式报错（这是故意的）。
#     python 01_源代码/calibrate_head.py --dataset mit --split train
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
SRC="$ROOT/01_源代码"
PY="$ROOT/venv/Scripts/python.exe"
CAL_JSON="$ROOT/03_实验结果/calib/calibration_head.json"

if [ ! -f "$CAL_JSON" ]; then
    echo "找不到 $CAL_JSON" >&2
    echo "先拟合：python 01_源代码/calibrate_head.py --dataset mit --split train" >&2
    exit 1
fi

MODEL="chronos-bolt-cal"
ONLY=${1:-all}
for ds in mit nasa calce; do
    [ "$ONLY" != "all" ] && [ "$ONLY" != "$ds" ] && continue
    # 口径必须与主表逐字对齐，否则三方对比不成立：
    #   MIT      → 主表是 ctx512 公平对照（零样本 Bolt 4.110），
    #              故 --max-context 512 + --split test + --tag ctx512，
    #              让 report.py 把它排进 `mit · test · ctx512` 同一段；
    #   NASA/CALCE → 主表用**原生**上下文（Bolt 2048），故不传 --max-context。
    extra=""
    suffix=""
    if [ "$ds" = "mit" ]; then
        extra="--split test --max-context 512 --tag ctx512"
        # 目录名也带 ctx512：report.py 是按**目录名**推 tag 的（--tag 只写进 meta），
        # 不带后缀就会掉进 `mit · test`（原生 2048 上下文）那一段，
        # 与零样本 4.158 并列，读者会拿错基线比。
        suffix="-ctx512"
    fi
    # 一级目录（进主表）：口径跟主表走 —— MIT 用 ctx512，NASA/CALCE 用原生上下文。
    echo ""
    echo ">>> [$ds] $MODEL -> 主表口径  ($(date +%H:%M:%S))"
    mkdir -p "$ROOT/03_实验结果/$ds-$MODEL$suffix"
    # shellcheck disable=SC2086
    ( cd "$SRC" && "$PY" run_benchmark.py --model "$MODEL" --dataset "$ds" \
        --output-dir "$ROOT/03_实验结果/$ds-$MODEL$suffix" $extra ) 2>&1

    # 二级目录（给 analyze_eol2 / 概率口径）：**一律 ctx512**，
    # 因为 run_prob.sh 的零样本矩阵就是全库 ctx512，CRPS/PICP 只能同口径比。
    echo ""
    echo ">>> [$ds] $MODEL -> prob/ 口径(ctx512)  ($(date +%H:%M:%S))"
    mkdir -p "$ROOT/03_实验结果/prob/$ds-$MODEL-ctx512"
    ( cd "$SRC" && "$PY" run_benchmark.py --model "$MODEL" --dataset "$ds" \
        --max-context 512 --output-dir "$ROOT/03_实验结果/prob/$ds-$MODEL-ctx512" \
        $( [ "$ds" = "mit" ] && echo "--split test" ) ) 2>&1
done
echo ""
echo "== 校准头评测完成 $(date) =="
