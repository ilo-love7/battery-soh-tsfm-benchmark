#!/usr/bin/env bash
# 核心环境安装（macOS / Linux / Windows-Git-Bash 通用）。
# 只装跑经典基线与数据加载所需依赖；基础模型依赖见 requirements-models.txt。
set -eu
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
echo "== 使用解释器：$($PY -V) =="

if [ ! -d venv ]; then
  echo "== 创建 venv =="
  "$PY" -m venv venv
fi

if [ -x venv/bin/python ]; then VP=venv/bin/python; else VP=venv/Scripts/python.exe; fi
"$VP" -m pip install --upgrade pip -q
"$VP" -m pip install -r requirements-core.txt

"$VP" -c "import numpy,pandas,scipy,sklearn;print('core OK')"
echo
echo "== 完成 =="
echo "下一步："
echo "  cd 01_源代码"
echo "  ../$VP run_benchmark.py --model all-baselines --dataset mit --split test"
echo "基础模型依赖：pip install -r requirements-models.txt（并预先下载权重，见 README）"
