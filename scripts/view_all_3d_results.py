#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看所有3D结果
扫描当前运行目录下的3D可视化文件并生成索引页面
"""

import sys
from pathlib import Path

CURR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURR))
sys.path.insert(0, str(CURR / 'scripts'))

from scripts.result_manager import result_manager


def main():
    run_dir = result_manager.get_current_run_dir()
    if not run_dir:
        # 尝试使用最近一次运行
        runs = result_manager.list_all_runs()
        if runs:
            runs_sorted = sorted(runs, key=lambda r: r.get('start_time', ''))
            run_dir = Path(runs_sorted[-1]['directory'])
        else:
            print('❌ 没有检测到任何运行。请先执行完整工作流程。')
            sys.exit(1)

    viz_dir = run_dir / 'visualization_3d'
    if not viz_dir.exists():
        print(f'❌ 未找到3D可视化目录: {viz_dir}')
        sys.exit(1)

    html_files = sorted([p for p in viz_dir.glob('*.html')])
    if not html_files:
        print('⚠️ 3D可视化目录中没有HTML文件。请先生成3D报告。')
        sys.exit(0)

    # 生成索引页面
    index_path = viz_dir / 'index.html'
    items = '\n'.join([f'<li><a href="{p.name}" target="_blank">{p.name}</a></li>' for p in html_files])
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>3D结果索引</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; margin: 20px; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    .card {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🎬 3D结果索引</h1>
    <div class="card">
      <p>目录：{viz_dir.name}</p>
      <ul>
        {items}
      </ul>
    </div>
  </div>
</body>
</html>
"""
    index_path.write_text(html, encoding='utf-8')

    print(f"✅ 3D索引页面: {index_path}")
    print("包含以下文件:")
    for p in html_files:
        print(f" - {p}")


if __name__ == '__main__':
    main()
