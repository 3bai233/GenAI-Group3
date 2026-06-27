#!/usr/bin/env python3
"""
分析 bbox2 (ReGrounding) 对正确率的贡献

功能:
1. 统计 bbox1/bbox2 各自正确的数量
2. 统计 Judge 选择分布
3. 分析 bbox2 带来的额外正确案例
4. 按任务类型细分统计
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


def check_point_in_box(point, bbox):
    """检查点是否在 bbox 内"""
    if not point or not bbox:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def compute_bbox_center(bbox):
    """计算 bbox 中心点"""
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def analyze_results(json_path: str):
    """分析评测结果"""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取详细结果
    results = data.get('details', [])
    if not results:
        print("未找到详细结果数据")
        return
    
    # 尝试从 pipeline 文件中补充 selected_image
    json_dir = Path(json_path).parent
    log_name = Path(json_path).stem
    pipeline_dir = json_dir / log_name / "pipelines"
    
    if pipeline_dir.exists():
        print(f"从 pipeline 文件补充 selected_image...")
        for i, result in enumerate(results):
            if result.get('selected_box') is None and result.get('selected_image') is None:
                pipeline_file = pipeline_dir / f"{i}.pipeline.json"
                if pipeline_file.exists():
                    try:
                        with open(pipeline_file, 'r') as pf:
                            pdata = json.load(pf)
                        step6 = pdata.get('results', {}).get('6', {})
                        selected = step6.get('selected_image')
                        if selected:
                            result['selected_image'] = selected
                    except Exception:
                        pass
    
    print("=" * 80)
    print("Bbox2 (ReGrounding) 贡献分析")
    print("=" * 80)
    print(f"结果文件: {json_path}")
    print(f"总样本数: {len(results)}")
    print()
    
    # 统计变量
    stats = {
        'total': 0,
        'bbox1_correct': 0,  # bbox1 正确
        'bbox2_correct': 0,  # bbox2 正确
        'both_correct': 0,   # 两个都正确
        'neither_correct': 0, # 两个都错误
        'only_bbox1_correct': 0,  # 仅 bbox1 正确
        'only_bbox2_correct': 0,  # 仅 bbox2 正确
        'selected_box1': 0,  # Judge 选了 box1
        'selected_box2': 0,  # Judge 选了 box2
        'final_correct': 0,  # 最终结果正确
        'judge_chose_wrong': 0,  # Judge 选错了（有正确选项但选了错的）
        'bbox2_contribution': 0,  # bbox2 带来的正确案例（bbox2正确且被选中，bbox1错误）
    }
    
    # 按任务类型统计
    task_stats = defaultdict(lambda: {
        'total': 0, 'bbox1_correct': 0, 'bbox2_correct': 0,
        'selected_box2': 0, 'bbox2_contribution': 0, 'final_correct': 0
    })
    
    # bbox2 贡献的案例列表
    bbox2_contribution_cases = []
    
    for result in results:
        stats['total'] += 1
        
        bbox1 = result.get('bbox1')
        bbox2 = result.get('bbox2')
        gt_bbox = result.get('gt')
        # 支持两种字段名: selected_box 或 selected_image
        selected_box = result.get('selected_box') or result.get('selected_image')
        correctness = result.get('correctness')
        task_filename = result.get('task_filename', 'unknown')
        
        # 计算 bbox1 和 bbox2 的中心点
        point1 = compute_bbox_center(bbox1)
        point2 = compute_bbox_center(bbox2)
        
        # 检查各自是否正确
        bbox1_is_correct = check_point_in_box(point1, gt_bbox) if point1 and gt_bbox else False
        bbox2_is_correct = check_point_in_box(point2, gt_bbox) if point2 and gt_bbox else False
        
        # 更新统计
        if bbox1_is_correct:
            stats['bbox1_correct'] += 1
            task_stats[task_filename]['bbox1_correct'] += 1
        if bbox2_is_correct:
            stats['bbox2_correct'] += 1
            task_stats[task_filename]['bbox2_correct'] += 1
        
        if bbox1_is_correct and bbox2_is_correct:
            stats['both_correct'] += 1
        elif not bbox1_is_correct and not bbox2_is_correct:
            stats['neither_correct'] += 1
        elif bbox1_is_correct and not bbox2_is_correct:
            stats['only_bbox1_correct'] += 1
        elif not bbox1_is_correct and bbox2_is_correct:
            stats['only_bbox2_correct'] += 1
        
        # Judge 选择统计
        if selected_box == '1' or selected_box == 1:
            stats['selected_box1'] += 1
        elif selected_box == '2' or selected_box == 2:
            stats['selected_box2'] += 1
            task_stats[task_filename]['selected_box2'] += 1
        
        # 最终结果正确
        if correctness == 'correct':
            stats['final_correct'] += 1
            task_stats[task_filename]['final_correct'] += 1
        
        # Judge 选错统计
        if bbox1_is_correct and not bbox2_is_correct and (selected_box == '2' or selected_box == 2):
            stats['judge_chose_wrong'] += 1
        elif not bbox1_is_correct and bbox2_is_correct and (selected_box == '1' or selected_box == 1):
            stats['judge_chose_wrong'] += 1
        
        # bbox2 贡献（bbox2正确且被选中，同时 bbox1 错误）
        if bbox2_is_correct and (selected_box == '2' or selected_box == 2) and not bbox1_is_correct:
            stats['bbox2_contribution'] += 1
            task_stats[task_filename]['bbox2_contribution'] += 1
            bbox2_contribution_cases.append({
                'task': task_filename,
                'uid': result.get('uid'),
                'query': result.get('prompt_to_evaluate', '')[:80] + '...',
                'bbox1': bbox1,
                'bbox2': bbox2,
                'gt': gt_bbox
            })
        
        task_stats[task_filename]['total'] += 1
    
    # 输出统计结果
    print("-" * 80)
    print("总体统计")
    print("-" * 80)
    print(f"总样本数:           {stats['total']}")
    print(f"最终正确:           {stats['final_correct']} ({100*stats['final_correct']/stats['total']:.2f}%)")
    print()
    
    print("Bbox 正确性分析:")
    print(f"  Bbox1 正确:       {stats['bbox1_correct']} ({100*stats['bbox1_correct']/stats['total']:.2f}%)")
    print(f"  Bbox2 正确:       {stats['bbox2_correct']} ({100*stats['bbox2_correct']/stats['total']:.2f}%)")
    print(f"  两个都正确:       {stats['both_correct']} ({100*stats['both_correct']/stats['total']:.2f}%)")
    print(f"  两个都错误:       {stats['neither_correct']} ({100*stats['neither_correct']/stats['total']:.2f}%)")
    print(f"  仅 Bbox1 正确:    {stats['only_bbox1_correct']} ({100*stats['only_bbox1_correct']/stats['total']:.2f}%)")
    print(f"  仅 Bbox2 正确:    {stats['only_bbox2_correct']} ({100*stats['only_bbox2_correct']/stats['total']:.2f}%)")
    print()
    
    print("Judge 选择分布:")
    print(f"  选择 Box1:        {stats['selected_box1']} ({100*stats['selected_box1']/stats['total']:.2f}%)")
    print(f"  选择 Box2:        {stats['selected_box2']} ({100*stats['selected_box2']/stats['total']:.2f}%)")
    print(f"  Judge 选错:       {stats['judge_chose_wrong']} ({100*stats['judge_chose_wrong']/stats['total']:.2f}%)")
    print()
    
    print("-" * 80)
    print("Bbox2 (ReGrounding) 贡献")
    print("-" * 80)
    print(f"Bbox2 额外贡献案例: {stats['bbox2_contribution']}")
    print(f"  (即: Bbox2 正确 + Judge 选择了 Box2 + Bbox1 错误)")
    print()
    
    if stats['only_bbox1_correct'] > 0 or stats['only_bbox2_correct'] > 0:
        print("理论最优 vs 实际:")
        theoretical_max = stats['bbox1_correct'] + stats['only_bbox2_correct']  # 或等价 stats['bbox2_correct'] + stats['only_bbox1_correct']
        theoretical_max = max(stats['bbox1_correct'], stats['bbox2_correct']) + stats['both_correct']
        # 更正确的计算：如果总是选对的，最多能对多少
        theoretical_max = stats['both_correct'] + stats['only_bbox1_correct'] + stats['only_bbox2_correct']
        print(f"  如果 Judge 总是选对: {theoretical_max} ({100*theoretical_max/stats['total']:.2f}%)")
        print(f"  实际正确:            {stats['final_correct']} ({100*stats['final_correct']/stats['total']:.2f}%)")
        print(f"  差距:                {theoretical_max - stats['final_correct']}")
    
    # 按任务类型统计
    if len(task_stats) > 1:
        print()
        print("-" * 80)
        print("按任务类型统计")
        print("-" * 80)
        print(f"{'任务名称':<35} {'总数':>6} {'正确':>6} {'正确率':>8} {'B2贡献':>6} {'B2选中':>6}")
        print("-" * 80)
        
        for task in sorted(task_stats.keys()):
            ts = task_stats[task]
            acc = 100 * ts['final_correct'] / ts['total'] if ts['total'] > 0 else 0
            print(f"{task:<35} {ts['total']:>6} {ts['final_correct']:>6} {acc:>7.2f}% {ts['bbox2_contribution']:>6} {ts['selected_box2']:>6}")
    
    # 输出 bbox2 贡献案例（最多显示 10 个）
    if bbox2_contribution_cases:
        print()
        print("-" * 80)
        print(f"Bbox2 贡献案例详情 (显示前 10 个，共 {len(bbox2_contribution_cases)} 个)")
        print("-" * 80)
        for i, case in enumerate(bbox2_contribution_cases[:10]):
            print(f"\n案例 {i+1}:")
            print(f"  任务: {case['task']}")
            print(f"  UID: {case['uid']}")
            print(f"  查询: {case['query']}")
            print(f"  Bbox1: {case['bbox1']}")
            print(f"  Bbox2: {case['bbox2']}")
            print(f"  GT: {case['gt']}")
    
    print()
    print("=" * 80)
    
    # 保存分析结果
    output_path = json_path.replace('.json', '_bbox2_analysis.json')
    analysis_result = {
        'summary': stats,
        'task_breakdown': dict(task_stats),
        'bbox2_contribution_cases': bbox2_contribution_cases
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='分析 bbox2 (ReGrounding) 对正确率的贡献')
    parser.add_argument('json_path', type=str, help='评测结果 JSON 文件路径')
    args = parser.parse_args()
    
    if not Path(args.json_path).exists():
        print(f"错误: 文件不存在: {args.json_path}")
        return
    
    analyze_results(args.json_path)


if __name__ == '__main__':
    main()
