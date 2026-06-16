#!/bin/bash
# 双模测试：bit_exact + std，各生成独立 report CSV。
cd "$(dirname "$0")"

run_one() {
    local mode="$1"       # bit_exact | std
    local build_dir="$2"  # build | build_std

    echo ""
    echo "============================================"
    echo "  [$mode] 编译 ..."
    echo "============================================"
    cmake --build "$build_dir" 2>&1 | tail -1

    echo "  [$mode] 测试 + 生成 report_${mode}.csv ..."
    REPORT_CSV="report_${mode}.csv" python3 -m pytest test_all.py -q --tb=line || true
    echo "  → report_${mode}.csv 已生成"
}

# bit_exact 模式
run_one bit_exact build

# std 模式
run_one std build_std

echo ""
echo "============================================"
echo "  完成"
echo "============================================"
ls -lh report_bit_exact.csv report_std.csv
