#!/usr/bin/env bash
set -euo pipefail

# 渲染Mermaid流程图为PNG
# 依赖: @mermaid-js/mermaid-cli (mmdc)
# 安装: npm i -g @mermaid-js/mermaid-cli
# 备用: docker run --rm -v "$(pwd)":/data ghcr.io/mermaid-js/mermaid-cli mmdc -i input.mmd -o output.png

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT_DIR/docs/diagrams"
OUT_DIR="$SRC_DIR/png"
mkdir -p "$OUT_DIR"

diagrams=(
  "fig_pl_transformer"
  "fig_equivariant_gnn"
  "fig_multitask_discriminator"
)

# 检查mmdc
if ! command -v mmdc >/dev/null 2>&1; then
  echo "[ERROR] 未找到 mmdc (Mermaid CLI)。请先安装:"
  echo "  npm i -g @mermaid-js/mermaid-cli"
  echo "或使用Docker:"
  echo "  docker run --rm -v \"$SRC_DIR\":/data ghcr.io/mermaid-js/mermaid-cli mmdc -i fig_pl_transformer.mmd -o fig_pl_transformer.png"
  exit 1
fi

for name in "${diagrams[@]}"; do
  in_file="$SRC_DIR/${name}.mmd"
  out_file="$OUT_DIR/${name}.png"
  echo "[INFO] 渲染 $in_file → $out_file"
  mmdc -i "$in_file" -o "$out_file" -b transparent -s 1.2
  echo "[OK] 生成: $out_file"
done

echo "[DONE] 全部PNG已生成，路径: $OUT_DIR"
