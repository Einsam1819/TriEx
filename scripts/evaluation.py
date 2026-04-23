#!/usr/bin/env python3
"""Poker game evaluation tool: produces tables and figures from experiment results."""

import json
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, suitable for WSL / headless
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5

def load_experiment_data(experiment_path: str) -> Dict:
    results_file = os.path.join(experiment_path, 'results.json')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Experiment results file not found: {results_file}")

    with open(results_file, 'r') as f:
        return json.load(f)

def get_player_display_name(player_id: str) -> str:
    """Map an internal player alias or model id to its display name."""
    alias_to_model = {
        'Alex Chen': 'GPT-4.1-mini',
        'Sarah Johnson': 'Llama-4-Maverick',
        'David Wilson': 'Monte Carlo',
        'Emily Zhang': 'DeepSeek-V3.2',
        'Robert Garcia': 'Qwen3-32B',
        'Jessica Liu': 'Gemini-2.5-Flash-Lite',
        'Niko Grey': 'Grok-3-Mini',
        'Lily Grant': 'Tight Passive',
        'Jade Park': 'Tight Aggressive',
        'Noah Kim': 'Loose Passive',
        'Ava Park': 'Loose Aggressive',
    }

    if player_id in alias_to_model:
        return alias_to_model[player_id]

    model_mapping = {
        'gpt': 'GPT-4o mini',
        'llama': 'Llama 4 Maverick',
        'claude': 'Claude 3.5 Haiku',
        'monte': 'Monte Carlo',
        'deepseek': 'DeepSeek Chat V3',
        'qwen': 'Qwen 2.5 Instruct',
        'gemini': 'Gemini 2.5 Flash'
    }

    return model_mapping.get(player_id, player_id)

def calculate_metrics(data: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute accumulated profit, win rate, and average chips per player."""
    battles = data['battles']

    player_data = {}

    for battle in battles:
        for player_name, player_info in battle['players'].items():
            if player_name not in player_data:
                player_data[player_name] = {
                    'accumulated_profits': [],
                    'avg_chips': [],
                    'total_hands_won': 0,
                    'total_hands_played': 0
                }

            initial_stack = player_info['initial_stack']
            total_profit_loss = player_info['total_profit_loss']
            accumulated_profit = (total_profit_loss / initial_stack) * 100
            player_data[player_name]['accumulated_profits'].append(accumulated_profit)

            hands_won = player_info.get('hands_won', 0)
            hands_played = player_info.get('hands_played', 0)
            player_data[player_name]['total_hands_won'] += hands_won
            player_data[player_name]['total_hands_played'] += hands_played

            current_stack = player_info['current_stack']
            player_data[player_name]['avg_chips'].append(current_stack)

    results = {}
    for player_name, data_lists in player_data.items():
        results[player_name] = {}

        accumulated_profits = data_lists['accumulated_profits']
        if accumulated_profits:
            results[player_name]['accumulated_profit'] = {
                'mean': np.mean(accumulated_profits),
                'std': np.std(accumulated_profits, ddof=1) if len(accumulated_profits) > 1 else 0.0
            }
        else:
            results[player_name]['accumulated_profit'] = {'mean': 0.0, 'std': 0.0}

        # Win rate is computed from totals across battles, not as the mean of
        # per-battle rates, so std is not meaningful.
        total_hands_won = data_lists['total_hands_won']
        total_hands_played = data_lists['total_hands_played']
        overall_win_rate = (total_hands_won / total_hands_played * 100) if total_hands_played > 0 else 0.0
        results[player_name]['win_rate'] = {
            'mean': overall_win_rate,
            'std': 0.0
        }

        avg_chips_list = data_lists['avg_chips']
        if avg_chips_list:
            results[player_name]['avg_chips'] = {
                'mean': np.mean(avg_chips_list),
                'std': np.std(avg_chips_list, ddof=1) if len(avg_chips_list) > 1 else 0.0
            }
        else:
            results[player_name]['avg_chips'] = {'mean': 0.0, 'std': 0.0}

    return results

def calculate_behavior_stats(data: Dict) -> Dict[str, Dict[str, float]]:
    """Aggregate fold / call / raise rates and aggression factor per player."""
    battles = data['battles']

    player_behavior_data = {}

    for battle in battles:
        for player_name, player_info in battle['players'].items():
            if player_name not in player_behavior_data:
                player_behavior_data[player_name] = {
                    'total_hands_played': 0,
                    'total_hands_folded': 0,
                    'total_hands_called': 0,
                    'total_hands_raised': 0,
                    'total_aggression_factor_sum': 0.0,
                    'battle_count': 0
                }

            player_behavior_data[player_name]['total_hands_played'] += player_info.get('hands_played', 0)
            player_behavior_data[player_name]['total_hands_folded'] += player_info.get('hands_folded', 0)
            player_behavior_data[player_name]['total_hands_called'] += player_info.get('hands_called', 0)
            player_behavior_data[player_name]['total_hands_raised'] += player_info.get('hands_raised', 0)
            player_behavior_data[player_name]['total_aggression_factor_sum'] += player_info.get('aggression_factor', 0.0)
            player_behavior_data[player_name]['battle_count'] += 1

    behavior_stats = {}
    for player_name, data in player_behavior_data.items():
        total_hands = data['total_hands_played']
        if total_hands > 0:
            fold_rate = (data['total_hands_folded'] / total_hands) * 100
            call_rate = (data['total_hands_called'] / total_hands) * 100
            raise_rate = (data['total_hands_raised'] / total_hands) * 100
        else:
            fold_rate = call_rate = raise_rate = 0.0

        avg_aggression_factor = data['total_aggression_factor_sum'] / data['battle_count'] if data['battle_count'] > 0 else 0.0

        behavior_stats[player_name] = {
            'hands_played': total_hands,
            'fold_rate': fold_rate,
            'call_rate': call_rate,
            'raise_rate': raise_rate,
            'aggression_factor': avg_aggression_factor
        }

    return behavior_stats

def collect_hand_strength_actions(data: Dict) -> Dict[str, List[Tuple[float, str]]]:
    """Collect (hand_strength, action) pairs per player from reasoning history."""
    battles = data['battles']

    player_hand_strength_actions = defaultdict(list)

    for battle in battles:
        for player_name, player_info in battle['players'].items():
            reasoning_history = player_info.get('reasoning_history', {})

            for round_num, round_data in reasoning_history.items():
                for street in ['preflop', 'flop', 'turn', 'river']:
                    if street in round_data and isinstance(round_data[street], dict):
                        street_data = round_data[street]

                        if ('game_state' in street_data and
                            'hand_strength' in street_data['game_state'] and
                            'action' in street_data):

                            hand_strength = street_data['game_state']['hand_strength']
                            action_data = street_data['action']

                            if isinstance(action_data, dict) and 'action' in action_data:
                                action = action_data['action']
                                player_hand_strength_actions[player_name].append((hand_strength, action))

    return dict(player_hand_strength_actions)

def print_results_table(results: Dict[str, Dict[str, Dict[str, float]]]):
    print("\nTable 1. Main Quantitative results of PokerBench")
    print("=" * 80)
    print(f"{'Agent':<20} {'Accumulated Profit (%)':<25} {'Per-Round Win Rate (%)':<22} {'Avg. Chips/Match':<15}")
    print("-" * 80)

    sorted_players = sorted(results.keys())

    for player in sorted_players:
        display_name = get_player_display_name(player)
        metrics = results[player]

        profit_mean = metrics['accumulated_profit']['mean']
        profit_std = metrics['accumulated_profit']['std']
        winrate_mean = metrics['win_rate']['mean']
        winrate_std = metrics['win_rate']['std']
        chips_mean = metrics['avg_chips']['mean']
        chips_std = metrics['avg_chips']['std']

        profit_str = f"{profit_mean:.1f}±{profit_std:.1f}"
        winrate_str = f"{winrate_mean:.1f}"
        chips_str = f"{chips_mean:.1f}±{chips_std:.1f}"

        print(f"{display_name:<20} {profit_str:<25} {winrate_str:<22} {chips_str:<15}")

    print("=" * 80)

def print_behavior_stats_table(behavior_stats: Dict[str, Dict[str, float]]):
    print("\nTable 2. Behavioral Statistics of PokerBench Agents")
    print("=" * 95)
    print(f"{'Agent':<20} {'Hands Played':<12} {'Fold Rate (%)':<12} {'Call Rate (%)':<12} {'Raise Rate (%)':<13} {'Aggression Factor':<15}")
    print("-" * 95)

    sorted_players = sorted(behavior_stats.keys())

    for player in sorted_players:
        display_name = get_player_display_name(player)
        stats = behavior_stats[player]

        hands_played = int(stats['hands_played'])
        fold_rate = stats['fold_rate']
        call_rate = stats['call_rate']
        raise_rate = stats['raise_rate']
        aggression_factor = stats['aggression_factor']

        print(f"{display_name:<20} {hands_played:<12} {fold_rate:<12.1f} {call_rate:<12.1f} {raise_rate:<13.1f} {aggression_factor:<15.2f}")

    print("=" * 95)

def generate_chips_over_rounds_figure(data: Dict, experiment_name: str):
    """Plot average chips per round across battles, with std-deviation band."""
    battles = data['battles']

    player_battle_chips = {}

    for battle_idx, battle in enumerate(battles):
        for player_name, player_info in battle['players'].items():
            if player_name not in player_battle_chips:
                player_battle_chips[player_name] = []

            initial_stack = player_info['initial_stack']
            battle_chips = [initial_stack]

            reasoning_history = player_info.get('reasoning_history', {})

            for round_num in range(1, 31):
                round_key = str(round_num)
                round_chips = initial_stack

                if round_key in reasoning_history:
                    round_data = reasoning_history[round_key]

                    # Use stack snapshot recorded at the preflop decision as
                    # the round-start stack.
                    if 'preflop' in round_data and isinstance(round_data['preflop'], dict):
                        preflop_data = round_data['preflop']
                        if 'game_state' in preflop_data:
                            game_state = preflop_data['game_state']
                            if 'position_info' in game_state and 'my_stack' in game_state['position_info']:
                                round_chips = game_state['position_info']['my_stack']

                battle_chips.append(round_chips)

            player_battle_chips[player_name].append(battle_chips)

    player_avg_chips = {}
    player_std_chips = {}

    for player_name, battle_list in player_battle_chips.items():
        if battle_list:
            chips_array = np.array(battle_list)
            avg_chips = np.mean(chips_array, axis=0).tolist()
            std_chips = np.std(chips_array, axis=0).tolist()
            player_avg_chips[player_name] = avg_chips
            player_std_chips[player_name] = std_chips

    plt.figure(figsize=(14, 6))

    # Colorblind-friendly palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    max_rounds = 30
    rounds = list(range(max_rounds + 1))

    for i, (player_name, avg_chips) in enumerate(player_avg_chips.items()):
        display_name = get_player_display_name(player_name)
        std_chips = player_std_chips[player_name]
        color = colors[i % len(colors)]

        upper_bound = [avg + std for avg, std in zip(avg_chips, std_chips)]
        lower_bound = [avg - std for avg, std in zip(avg_chips, std_chips)]

        plt.plot(rounds, avg_chips,
                label=display_name,
                color=color,
                linewidth=2,
                marker='o' if i < 3 else 's',
                markersize=3,
                markevery=5)

        plt.fill_between(rounds, lower_bound, upper_bound,
                        color=color, alpha=0.2)

        min_idx = np.argmin(lower_bound)
        max_idx = np.argmax(upper_bound)
        min_value = lower_bound[min_idx]
        max_value = upper_bound[max_idx]
        min_round = rounds[min_idx]
        max_round = rounds[max_idx]

        plt.scatter(min_round, min_value, color=color, s=30, marker='v', zorder=5)
        plt.scatter(max_round, max_value, color=color, s=30, marker='^', zorder=5)

        plt.annotate(f'{min_value:.0f}',
                    xy=(min_round, min_value),
                    xytext=(5, -15),
                    textcoords='offset points',
                    fontsize=9,
                    color=color,
                    fontweight='bold',
                    ha='left')

        plt.annotate(f'{max_value:.0f}',
                    xy=(max_round, max_value),
                    xytext=(5, 10),
                    textcoords='offset points',
                    fontsize=9,
                    color=color,
                    fontweight='bold',
                    ha='left')

    plt.xlabel('Round', fontsize=14, fontweight='bold')
    plt.ylabel('Average Chips', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 30)
    plt.tight_layout()

    filename = f"{experiment_name}_chips_over_rounds.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    plt.close()

    print(f"\nFigure saved: {filename}")

def generate_hand_strength_action_figure(hand_strength_data: Dict[str, List[Tuple[float, str]]], experiment_name: str):
    """Plot per-player action distribution stacked across hand-strength buckets."""

    filtered_data = {player: data for player, data in hand_strength_data.items() if data}

    if not filtered_data:
        print("No hand strength data found, skipping hand strength-action distribution figure")
        return

    n_players = len(filtered_data)
    if n_players <= 4:
        rows, cols = 1, n_players
    else:
        rows, cols = 2, 4

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))

    if n_players == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    action_colors = {
        'FOLD': '#d62728',
        'CALL': '#2ca02c',
        'RAISE': '#1f77b4',
        'CHECK': '#ff7f0e',
        'ALL_IN': '#9467bd'
    }

    strength_bins = [0, 0.5, 2, 10, 30, float('inf')]
    strength_labels = ['Very Weak\n(0-0.5)', 'Weak\n(0.5-2)', 'Medium\n(2-10)', 'Strong\n(10-30)']

    player_names = sorted(filtered_data.keys())

    for idx, player_name in enumerate(player_names):
        ax = axes_flat[idx]

        data = filtered_data[player_name]
        display_name = get_player_display_name(player_name)

        strength_action_counts = defaultdict(lambda: defaultdict(int))

        for hand_strength, action in data:
            strength_category = None
            for i, threshold in enumerate(strength_bins[1:]):
                if hand_strength <= threshold:
                    strength_category = strength_labels[i]
                    break

            if strength_category:
                strength_action_counts[strength_category][action] += 1

        categories = strength_labels
        actions = ['FOLD', 'CALL', 'RAISE', 'CHECK', 'ALL_IN']

        bottom = np.zeros(len(categories))

        for action in actions:
            counts = [strength_action_counts[cat][action] for cat in categories]
            if sum(counts) > 0:
                ax.bar(categories, counts, bottom=bottom,
                      label=action, color=action_colors.get(action, '#gray'),
                      alpha=0.8)
                bottom += counts

        ax.set_title(f'{display_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hand Strength', fontsize=12)
        ax.set_ylabel('Action Count', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

        ax.grid(True, alpha=0.3, axis='y')

        total_decisions = len(data)
        ax.text(0.98, 0.98, f'Total: {total_decisions}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for idx in range(n_players, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if n_players > 0:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                      ncol=len(handles), fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    filename = f"{experiment_name}_hand_strength_action_distribution.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    plt.close()

    print(f"Hand strength-action distribution figure saved: {filename}")

def list_experiments():
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        print("Experiments directory not found")
        return

    experiments = [d for d in os.listdir(experiments_dir)
                  if os.path.isdir(os.path.join(experiments_dir, d))]

    if not experiments:
        print("No experiments found")
        return

    print("Available experiments:")
    for exp in sorted(experiments):
        exp_path = os.path.join(experiments_dir, exp)
        results_file = os.path.join(exp_path, 'results.json')
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    info = data.get('experiment_info', {})
                    total_battles = info.get('total_battles', 'Unknown')
                    completed_battles = info.get('completed_battles', 'Unknown')
                    status = info.get('status', 'Unknown')
                    print(f"  {exp} ({completed_battles}/{total_battles} battles, {status})")
            except:
                print(f"  {exp} (unable to read info)")
        else:
            print(f"  {exp} (no results file)")

def main():
    parser = argparse.ArgumentParser(description='Poker Game Evaluation Tool')
    parser.add_argument('--list', action='store_true', help='List all available experiments')
    parser.add_argument('--experiment', type=str, help='Name of experiment to analyze')

    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if not args.experiment:
        print("Please specify experiment name to analyze, or use --list to see available experiments")
        return

    experiment_path = os.path.join("experiments", args.experiment)

    if not os.path.exists(experiment_path):
        print(f"Experiment does not exist: {args.experiment}")
        return

    try:
        print(f"Analyzing experiment: {args.experiment}")
        data = load_experiment_data(experiment_path)

        results = calculate_metrics(data)
        behavior_stats = calculate_behavior_stats(data)
        hand_strength_data = collect_hand_strength_actions(data)

        print_results_table(results)
        print_behavior_stats_table(behavior_stats)

        generate_chips_over_rounds_figure(data, args.experiment)
        generate_hand_strength_action_figure(hand_strength_data, args.experiment)

        print(f"\nAnalysis completed!")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
