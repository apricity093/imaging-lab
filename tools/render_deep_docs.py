"""Render CT deep-method pages from metadata and retained card bodies."""
from pathlib import Path
import json
from inv_framework.solvers.deep.registry import ALGORITHMS

ROOT = Path(__file__).resolve().parents[1]


def render_deep(name, record, destination):
    source = ALGORITHMS[name]
    lines = [f'# {source.title}', '', source.summary, '', '## 实现与执行约定', '',
        f'固定来源为 peterw634/inv_framework 的 dc181fb65afbf300a9935c3c770231c44ffc0c78。实现位于 inv_framework/solvers/deep，公开 ID 为 {name}。', '',
        source.scope, '', source.paper, '',
        'CT 公共任务接收 float32 图像 (B,C,H,W)、观测 (B,C,角度,探测器列)，支持二维平行束与扇束。独立求解器保留原生类型及构造参数。模型域转换、资源与运行条件见 [深度先验](../ct/deep_priors.md)。', '',
        '## 公共参数', '',
        '| 参数 | 默认值 | 含义与约束 | 来源与推荐范围 |', '| --- | --- | --- | --- |']
    for p in record['parameter_specs']:
        details = {key: p[key] for key in ('type', 'nullable', 'minimum', 'maximum', 'exclusive_minimum', 'exclusive_maximum', 'inclusive_minimum', 'inclusive_maximum', 'choices', 'schema', 'items', 'theoretical_constraint') if key in p}
        text = p.get('description', '') + ' ' + json.dumps(details, ensure_ascii=False)
        original = next((item for item in source.params if item.name == p['name']), None)
        provenance = (original.source + '；' + original.recommended) if original else '公共 CT 参数，默认配置与注册表'
        lines.append(f"| {p['name']} | {p['default']} | {text.replace('|', '/')} | {provenance.replace('|', '/')} |")
    lines += ['', 'YAML 默认值和参数含义来自实际注册信息，来源卡片保留推荐值依据。显式参数覆盖任务文件，任务文件覆盖算法配置。直接构造 DIP 默认 2000 次、EMA 关闭，推荐配置为 3000 次和 EMA 0.99。', '',
        '## 独立调用与文件任务', '', '```python',
        'from inv_framework.ct.toy import toy_experiment',
        'from inv_framework.runner import run_experiment',
        f"config = toy_experiment('examples/ct_data/{Path(source.toy_data[0]).name}', '{name}')",
        "# Set config['params']['weights'], image_scale and image_offset for the selected prior.",
        f"result = run_experiment(config, output_dir='outputs/{name}')", '```', '',
        '完整文件请求见 docs/ct/deep_request_template.yaml。使用 python -m inv_framework.runner --config REQUEST.yaml --output OUTPUT 执行，输出 result.json 和物理域 reconstruction.npy。上层调用使用相同 CT API 和声明的外层预算，无需 Agent SDK。', '',
        '## 验证与资源限制', '',
        '模型资源和运行条件见 [深度先验](../ct/deep_priors.md)。短预算执行只说明流程可用，不能代替推荐预算下的重建效果评价。']
    if name == 'budgeted_dps':
        position = lines.index("# Set config['params']['weights'], image_scale and image_offset for the selected prior.")
        lines.insert(position, "config['params']['schedule_actions'] = [list(range(40))]")
        lines[-1] = '本方法按代码、接口映射、参数与来源资源记录交付范围。'
    lines += ['', '## 数学与方法差异', '',
        'DIP/INR 在固定观测下优化网络参数，使 A fθ 与 y 匹配。扩散方法的去噪估计、梯度引导、伪逆投影与数据一致性方程，以及 pMF 速度积分和 LoRA 内层更新见 [方法公式](../ct/diffusion_methods.md)。各算法的来源参数依据与使用限制保留在下方卡片。', '']
    if name == 'budgeted_dps':
        lines += ['', 'Budgeted DPS 保留显式子集调度、成本函数和 reformulated_gradient。CT 映射限于无探测器掩膜的 Torch 二维平行束，其他任务需原生 SubsettableForwardOperator。schedule_actions 为非空的非负整数索引数组列表，key_indices 可空。本次范围为代码、接口、参数与来源资源。']
    lines += retained_body(name)
    (destination / f'{name}.md').write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


def retained_body(name):
    lines = []
    supplement = ROOT / 'docs/algorithms/supplements' / f'{name}.md'
    if supplement.exists():
        lines += ['', supplement.read_text(encoding='utf-8')]
    card = ROOT / 'docs/algorithms/source_cards' / f'{name}.md'
    if card.exists():
        body = card.read_text(encoding='utf-8').replace('../../../examples/', '../../examples/')
        lines += ['', '## 来源卡片记录', '',
            '固定来源为 peterw634/inv_framework 的 dc181fb65afbf300a9935c3c770231c44ffc0c78。以下测量、环境与评价口径为来源记录。当前公共输出、参数映射、离散化和限制见 CT 手册及本页公共参数。来源模型域输出映射为公共物理域后保存。', '', body]
    return lines
