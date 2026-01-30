#!/usr/bin/env python3
"""
Comprehensive analysis of concept effectiveness for geolocation.
Shows country-level patterns, per-concept analysis, training data correlations.

Usage:
    python scripts/evaluation/visualize_concept_help_geography.py \
        --results-dir results/model_comparison_stage3_latefusion \
        --output results/concept_help_geography.png \
        --train-csv path/to/train.csv
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import pandas as pd
from scipy import stats as scipy_stats

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not available, using simple scatter plot")

try:
    import reverse_geocoder as rg
    HAS_RG = True
except ImportError:
    HAS_RG = False
    print("Warning: reverse_geocoder not available, country detection will be limited")


def get_country_from_coords(lat: float, lng: float) -> Optional[str]:
    """Get country code from coordinates."""
    if not HAS_RG:
        return None
    try:
        result = rg.search([(lat, lng)], mode=1)
        if result and len(result) > 0:
            return result[0].get('cc', None)
    except Exception:
        pass
    return None


def batch_get_countries(coords: List[Tuple[float, float]]) -> List[Optional[str]]:
    """Get country codes for a batch of coordinates."""
    if not HAS_RG:
        return [None] * len(coords)
    try:
        results = rg.search(coords, mode=1)
        return [r.get('cc', None) if r else None for r in results]
    except Exception:
        return [None] * len(coords)


def load_results_from_compare_script(results_dir: Path, concept_vocab_path: Path) -> Dict:
    """
    Load results from compare_model_variants.py output.
    Since it doesn't save full predictions, we'll need to re-run or use a different approach.
    For now, this is a placeholder - we'll need to modify compare_model_variants to save full results.
    """
    # Check if there's a saved results file
    results_file = results_dir / "results_dict.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    
    # If not, we need to re-evaluate
    return None


def aggregate_by_country(
    true_lats: np.ndarray,
    true_lngs: np.ndarray,
    errors_both: np.ndarray,
    errors_image_only: np.ndarray,
    top_concepts: np.ndarray,
    idx_to_concept: Dict[int, str],
    concept_probs: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """
    Aggregate results by country.
    
    Returns:
        Dict mapping country code to aggregated stats
    """
    # Get countries for all coordinates
    print("Getting country codes for all samples...")
    coords = [(float(lat), float(lng)) for lat, lng in zip(true_lats, true_lngs)]
    countries = batch_get_countries(coords)
    
    # Compute improvement
    improvement = errors_image_only - errors_both  # Positive = concepts help
    
    # Compute activation scores for the top predicted concept per sample
    if concept_probs is not None:
        top_activations = np.array([concept_probs[i][top_concepts[i]] for i in range(len(top_concepts))])
    else:
        top_activations = None
    
    # Aggregate by country
    country_stats = defaultdict(lambda: {
        'samples': [],
        'improvements': [],
        'errors_both': [],
        'errors_image_only': [],
        'top_concepts': [],
        'activations': [],
        'lats': [],
        'lngs': [],
    })
    
    for i, country in enumerate(countries):
        if country is None:
            continue
        country_stats[country]['samples'].append(i)
        country_stats[country]['improvements'].append(improvement[i])
        country_stats[country]['errors_both'].append(errors_both[i])
        country_stats[country]['errors_image_only'].append(errors_image_only[i])
        country_stats[country]['top_concepts'].append(top_concepts[i])
        if top_activations is not None:
            country_stats[country]['activations'].append(top_activations[i])
        country_stats[country]['lats'].append(true_lats[i])
        country_stats[country]['lngs'].append(true_lngs[i])
    
    # Compute aggregated statistics
    aggregated = {}
    for country, stats in country_stats.items():
        improvements = np.array(stats['improvements'])
        errors_both_arr = np.array(stats['errors_both'])
        errors_img_arr = np.array(stats['errors_image_only'])
        
        # Most common helpful concept
        helps_mask = improvements > 0
        if np.sum(helps_mask) > 0:
            helpful_concepts = [stats['top_concepts'][i] for i in range(len(stats['top_concepts'])) if helps_mask[i]]
            if helpful_concepts:
                from collections import Counter
                concept_counts = Counter(helpful_concepts)
                most_common_concept_idx = concept_counts.most_common(1)[0][0]
                most_common_concept_name = idx_to_concept.get(most_common_concept_idx, f"Concept {most_common_concept_idx}")
            else:
                most_common_concept_name = None
        else:
            most_common_concept_name = None
        
        aggregated[country] = {
            'num_samples': len(stats['samples']),
            'mean_improvement': float(np.mean(improvements)),
            'median_improvement': float(np.median(improvements)),
            'std_improvement': float(np.std(improvements)),
            'pct_helps': float(np.mean(improvements > 0) * 100),
            'mean_error_both': float(np.mean(errors_both_arr)),
            'mean_error_image_only': float(np.mean(errors_img_arr)),
            'most_helpful_concept': most_common_concept_name,
            'center_lat': float(np.mean(stats['lats'])),
            'center_lng': float(np.mean(stats['lngs'])),
            # Raw data for drill-down analysis
            'concept_list': stats['top_concepts'],
            'improvement_list': stats['improvements'],
            'activation_list': stats['activations'] if stats['activations'] else None,
            'mean_activation': float(np.mean(stats['activations'])) if stats['activations'] else None,
        }
    
    return aggregated


def visualize_country_drilldown(
    country_stats: Dict[str, Dict],
    idx_to_concept: Dict[int, str],
    train_counts: Optional[Dict[int, int]],
    train_counts_by_country: Optional[Dict[str, int]],
    train_counts_by_concept_country: Optional[Dict[str, Dict[str, int]]],
    output_path: Path,
    worst_n: int = 6,
    include_mexico: bool = True,
):
    """
    Create detailed drill-down visualizations for countries where concepts hurt most.
    
    For each country, shows:
    - Concept distribution (which concepts were predicted)
    - Per-concept: samples that helped vs hurt
    - Training data: train_in_country vs train_overall
    """
    # Sort countries by mean improvement (worst first)
    sorted_countries = sorted(
        [(code, stats) for code, stats in country_stats.items() 
         if stats['num_samples'] >= 5 and 'concept_list' in stats],
        key=lambda x: x[1]['mean_improvement']
    )
    
    # Get worst N countries
    worst_countries = sorted_countries[:worst_n]
    
    # Add Mexico if not already included
    if include_mexico:
        mexico_found = any(code == 'MX' for code, _ in worst_countries)
        if not mexico_found:
            mexico_entry = next(((code, stats) for code, stats in sorted_countries if code == 'MX'), None)
            if mexico_entry:
                worst_countries.append(mexico_entry)
    
    if not worst_countries:
        print("No countries with enough data for drill-down")
        return
    
    # Create figure with subplots for each country
    n_countries = len(worst_countries)
    n_cols = 3
    n_rows = (n_countries + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (country_code, stats) in enumerate(worst_countries):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Get concept breakdown for this country
        concepts = stats['concept_list']
        improvements = stats['improvement_list']
        activations = stats.get('activation_list', None)
        
        # Count per concept: total, helps, hurts, activations
        concept_breakdown = defaultdict(lambda: {'total': 0, 'helps': 0, 'hurts': 0, 'improvements': [], 'activations': []})
        for i, (concept_idx, imp) in enumerate(zip(concepts, improvements)):
            concept_name = idx_to_concept.get(int(concept_idx), f'concept_{concept_idx}')
            concept_breakdown[concept_name]['total'] += 1
            concept_breakdown[concept_name]['improvements'].append(imp)
            if activations is not None and i < len(activations):
                concept_breakdown[concept_name]['activations'].append(activations[i])
            if imp > 0:
                concept_breakdown[concept_name]['helps'] += 1
            else:
                concept_breakdown[concept_name]['hurts'] += 1
        
        # Sort by total count
        sorted_concepts = sorted(concept_breakdown.items(), key=lambda x: x[1]['total'], reverse=True)[:10]
        
        if not sorted_concepts:
            ax.text(0.5, 0.5, f'{country_code}: No concept data', ha='center', va='center')
            ax.set_title(f'{country_code}')
            continue
        
        # Prepare data for stacked bar chart
        concept_names = []
        helps_counts = []
        hurts_counts = []
        mean_improvements = []
        mean_activations = []
        
        for name, data in sorted_concepts:
            # Format name nicely
            display_name = name.replace('_', ' ').title()
            if len(display_name) > 18:
                display_name = display_name[:15] + '...'
            
            # Get training counts: overall and in-country
            train_overall = 0
            train_in_country = 0
            if train_counts:
                # Find concept index for overall count
                for cidx, cname in idx_to_concept.items():
                    if cname == name:
                        train_overall = train_counts.get(cidx, 0)
                        break
            
            # Get in-country training count
            if train_counts_by_concept_country and country_code in train_counts_by_concept_country:
                train_in_country = train_counts_by_concept_country[country_code].get(name, 0)
            
            # Compute mean activation for this concept
            mean_act = np.mean(data['activations']) if data['activations'] else 0.0
            
            # Build label with counts, activation, and training info
            concept_names.append(f'{display_name}\n(n={data["total"]}, act={mean_act:.2f}, tr:{train_in_country}/{train_overall})')
            helps_counts.append(data['helps'])
            hurts_counts.append(data['hurts'])
            mean_improvements.append(np.mean(data['improvements']))
            mean_activations.append(mean_act)
        
        y_pos = np.arange(len(concept_names))
        
        # Create stacked horizontal bar chart
        bars_helps = ax.barh(y_pos, helps_counts, color='#2ecc71', alpha=0.8, label='Helps', edgecolor='white')
        bars_hurts = ax.barh(y_pos, [-h for h in hurts_counts], color='#e74c3c', alpha=0.8, label='Hurts', edgecolor='white')
        
        # Add mean improvement annotations
        for i, (bar_h, bar_hurt, mean_imp, mean_act) in enumerate(zip(bars_helps, bars_hurts, mean_improvements, mean_activations)):
            # Annotate on the right side
            max_x = max(helps_counts) + 2
            color = '#27ae60' if mean_imp > 0 else '#c0392b'
            ax.text(max_x, i, f'Δ={mean_imp:+.0f}km', va='center', ha='left', 
                   fontsize=7, fontweight='bold', color=color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(concept_names, fontsize=7)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('← Hurts | Helps →', fontsize=10)
        
        # Get train count for this country
        train_in_country_total = train_counts_by_country.get(country_code, 0) if train_counts_by_country else 0
        
        # Get mean activation for this country overall
        country_mean_act = stats.get('mean_activation', None)
        act_str = f', Avg Act={country_mean_act:.2f}' if country_mean_act is not None else ''
        
        # Title with key stats including activation
        title = (f'{country_code}: Mean Δ={stats["mean_improvement"]:.0f}km, '
                f'Median Δ={stats["median_improvement"]:.0f}km{act_str}\n'
                f'Test={stats["num_samples"]}, Train={train_in_country_total}, '
                f'Helps={stats["pct_helps"]:.0f}%')
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Set x limits symmetrically
        max_val = max(max(helps_counts), max(hurts_counts)) * 1.3
        ax.set_xlim(-max_val, max_val * 1.5)  # Extra space for annotations
        
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='lower right', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(worst_countries), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Country Drill-Down: Why Concepts Hurt\n(Per-Concept: Helps vs Hurts, act=activation score, tr=train in-country/overall)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    drilldown_path = output_path.parent / f"{output_path.stem}_country_drilldown.png"
    plt.savefig(drilldown_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved country drill-down to: {drilldown_path}")
    return drilldown_path


def visualize_top_countries_drilldown(
    country_stats: Dict[str, Dict],
    idx_to_concept: Dict[int, str],
    train_counts: Optional[Dict[int, int]],
    train_counts_by_country: Optional[Dict[str, int]],
    train_counts_by_concept_country: Optional[Dict[str, Dict[str, int]]],
    output_path: Path,
    top_n: int = 6,
):
    """
    Create detailed drill-down visualizations for countries where concepts help most.
    
    For each country, shows:
    - Concept distribution (which concepts were predicted)
    - Per-concept: samples that helped vs hurt
    - Training data: train_in_country vs train_overall
    """
    # Sort countries by mean improvement (best first)
    sorted_countries = sorted(
        [(code, stats) for code, stats in country_stats.items() 
         if stats['num_samples'] >= 5 and 'concept_list' in stats],
        key=lambda x: x[1]['mean_improvement'],
        reverse=True
    )[:top_n]
    
    if not sorted_countries:
        print("No countries with enough data for top countries drill-down")
        return
    
    # Create figure with subplots for each country
    n_countries = len(sorted_countries)
    n_cols = 3
    n_rows = (n_countries + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (country_code, stats) in enumerate(sorted_countries):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Get concept breakdown for this country
        concepts = stats['concept_list']
        improvements = stats['improvement_list']
        activations = stats.get('activation_list', None)
        
        # Count per concept: total, helps, hurts, activations
        concept_breakdown = defaultdict(lambda: {'total': 0, 'helps': 0, 'hurts': 0, 'improvements': [], 'activations': []})
        for i, (concept_idx, imp) in enumerate(zip(concepts, improvements)):
            concept_name = idx_to_concept.get(int(concept_idx), f'concept_{concept_idx}')
            concept_breakdown[concept_name]['total'] += 1
            concept_breakdown[concept_name]['improvements'].append(imp)
            if activations is not None and i < len(activations):
                concept_breakdown[concept_name]['activations'].append(activations[i])
            if imp > 0:
                concept_breakdown[concept_name]['helps'] += 1
            else:
                concept_breakdown[concept_name]['hurts'] += 1
        
        # Sort by total count
        sorted_concepts = sorted(concept_breakdown.items(), key=lambda x: x[1]['total'], reverse=True)[:10]
        
        if not sorted_concepts:
            ax.text(0.5, 0.5, f'{country_code}: No concept data', ha='center', va='center')
            ax.set_title(f'{country_code}')
            continue
        
        # Prepare data for stacked bar chart
        concept_names = []
        helps_counts = []
        hurts_counts = []
        mean_improvements = []
        mean_activations = []
        
        for name, data in sorted_concepts:
            # Format name nicely
            display_name = name.replace('_', ' ').title()
            if len(display_name) > 18:
                display_name = display_name[:15] + '...'
            
            # Get training counts: overall and in-country
            train_overall = 0
            train_in_country = 0
            if train_counts:
                # Find concept index for overall count
                for cidx, cname in idx_to_concept.items():
                    if cname == name:
                        train_overall = train_counts.get(cidx, 0)
                        break
            
            # Get in-country training count
            if train_counts_by_concept_country and country_code in train_counts_by_concept_country:
                train_in_country = train_counts_by_concept_country[country_code].get(name, 0)
            
            # Compute mean activation for this concept
            mean_act = np.mean(data['activations']) if data['activations'] else 0.0
            
            # Build label with counts, activation, and training info
            concept_names.append(f'{display_name}\n(n={data["total"]}, act={mean_act:.2f}, tr:{train_in_country}/{train_overall})')
            helps_counts.append(data['helps'])
            hurts_counts.append(data['hurts'])
            mean_improvements.append(np.mean(data['improvements']))
            mean_activations.append(mean_act)
        
        y_pos = np.arange(len(concept_names))
        
        # Create stacked horizontal bar chart
        bars_helps = ax.barh(y_pos, helps_counts, color='#2ecc71', alpha=0.8, label='Helps', edgecolor='white')
        bars_hurts = ax.barh(y_pos, [-h for h in hurts_counts], color='#e74c3c', alpha=0.8, label='Hurts', edgecolor='white')
        
        # Add mean improvement annotations
        for i, (bar_h, bar_hurt, mean_imp, mean_act) in enumerate(zip(bars_helps, bars_hurts, mean_improvements, mean_activations)):
            # Annotate on the right side
            max_x = max(helps_counts) + 2
            color = '#27ae60' if mean_imp > 0 else '#c0392b'
            ax.text(max_x, i, f'Δ={mean_imp:+.0f}km', va='center', ha='left', 
                   fontsize=7, fontweight='bold', color=color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(concept_names, fontsize=7)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('← Hurts | Helps →', fontsize=10)
        
        # Get train count for this country
        train_in_country_total = train_counts_by_country.get(country_code, 0) if train_counts_by_country else 0
        
        # Get mean activation for this country overall
        country_mean_act = stats.get('mean_activation', None)
        act_str = f', Avg Act={country_mean_act:.2f}' if country_mean_act is not None else ''
        
        # Title with key stats including activation
        title = (f'{country_code}: Mean Δ={stats["mean_improvement"]:.0f}km, '
                f'Median Δ={stats["median_improvement"]:.0f}km{act_str}\n'
                f'Test={stats["num_samples"]}, Train={train_in_country_total}, '
                f'Helps={stats["pct_helps"]:.0f}%')
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Set x limits symmetrically
        max_val = max(max(helps_counts), max(hurts_counts)) * 1.3
        ax.set_xlim(-max_val, max_val * 1.5)  # Extra space for annotations
        
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='lower right', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(sorted_countries), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Country Drill-Down: Why Concepts Help\n(Per-Concept: Helps vs Hurts, act=activation score, tr=train in-country/overall)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    drilldown_path = output_path.parent / f"{output_path.stem}_top_countries_drilldown.png"
    plt.savefig(drilldown_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved top countries drill-down to: {drilldown_path}")
    return drilldown_path


def visualize_activation_vs_help(
    errors_both: np.ndarray,
    errors_image_only: np.ndarray,
    top_concepts: np.ndarray,
    concept_probs: np.ndarray,
    idx_to_concept: Dict[int, str],
    output_path: Path,
    true_lats: np.ndarray,
    true_lngs: np.ndarray,
    test_csv: Optional[Path] = None,
    concept_vocab_path: Optional[Path] = None,
):
    """
    Visualize the relationship between activation scores and whether concepts help.
    Uses GROUND TRUTH concept activation (how confident the model is about the true concept).
    
    Creates a multi-panel plot showing:
    1. Scatter: ground truth concept activation vs improvement
    2. Box plots: activation distribution for helps vs hurts
    3. Binned analysis: average improvement by activation range
    4. Per-concept activation vs success rate
    """
    if concept_probs is None or len(concept_probs) == 0:
        print("Warning: No concept probabilities available for activation analysis")
        return
    
    # Load ground truth concept labels from test CSV and match by coordinates
    gt_concept_labels = None
    activation_label = "Predicted Concept Activation"
    
    print(f"\n🔍 Loading ground truth concept labels...")
    print(f"   Test CSV: {test_csv}")
    print(f"   Test CSV exists: {test_csv.exists() if test_csv else False}")
    print(f"   Concept vocab: {concept_vocab_path}")
    print(f"   Concept vocab exists: {concept_vocab_path.exists() if concept_vocab_path else False}")
    
    if test_csv and test_csv.exists() and concept_vocab_path and concept_vocab_path.exists():
        try:
            df = pd.read_csv(test_csv)
            print(f"   Loaded CSV with {len(df)} rows")
            with open(concept_vocab_path) as f:
                concept_vocab = json.load(f)
            concept_to_idx = {str(v): int(k) for k, v in concept_vocab["idx_to_concept"].items()}
            print(f"   Concept vocab has {len(concept_to_idx)} concepts")
            
            # Get concept column
            concept_col = None
            if 'generalized' in df.columns:
                concept_col = 'generalized'
                print(f"   Using 'generalized' column for concepts")
            elif 'meta_name' in df.columns:
                concept_col = 'meta_name'
                print(f"   Using 'meta_name' column for concepts")
            else:
                print(f"   ⚠️ No concept column found! Available columns: {list(df.columns)}")
            
            if concept_col:
                # Match samples by coordinates (lat, lng)
                print(f"   Matching samples by coordinates...")
                coord_to_concept = {}
                lat_col = 'lat' if 'lat' in df.columns else 'latitude'
                lng_col = 'lng' if 'lng' in df.columns else 'longitude'
                
                for _, row in df.iterrows():
                    concept_name = str(row[concept_col])
                    concept_idx = concept_to_idx.get(concept_name, -1)
                    lat_key = round(float(row[lat_col]), 4)
                    lng_key = round(float(row[lng_col]), 4)
                    coord_to_concept[(lat_key, lng_key)] = concept_idx
                
                # Match predictions to CSV by coordinates
                gt_concept_labels = []
                matched = 0
                for i in range(len(true_lats)):
                    lat_key = round(float(true_lats[i]), 4)
                    lng_key = round(float(true_lngs[i]), 4)
                    concept_idx = coord_to_concept.get((lat_key, lng_key), -1)
                    if concept_idx >= 0:
                        matched += 1
                    gt_concept_labels.append(concept_idx)
                
                gt_concept_labels = np.array(gt_concept_labels)
                print(f"✓ Matched {matched}/{len(true_lats)} samples by coordinates ({matched/len(true_lats)*100:.1f}%)")
                activation_label = "Ground Truth Concept Activation"
            else:
                print(f"   ⚠️ Cannot proceed without concept column")
                gt_concept_labels = None
        except Exception as e:
            print(f"⚠️ Warning: Could not load ground truth labels: {e}")
            import traceback
            traceback.print_exc()
            gt_concept_labels = None
    else:
        print(f"⚠️ Test CSV or concept vocab not available - will use predicted activations")
    
    # Use ground truth if available
    if gt_concept_labels is not None:
        # Use activation of GROUND TRUTH concept (how confident model is about the true concept)
        valid_mask = (gt_concept_labels >= 0) & (gt_concept_labels < concept_probs.shape[1])
        valid_count = np.sum(valid_mask)
        print(f"   Valid GT labels: {valid_count}/{len(gt_concept_labels)}")
        
        top_activations = np.array([
            concept_probs[i][gt_concept_labels[i]] 
            if 0 <= gt_concept_labels[i] < len(concept_probs[i])
            else 0.0 
            for i in range(len(concept_probs))
        ])
        print(f"✓ Using GROUND TRUTH concept activations for analysis")
        print(f"   Activation label: '{activation_label}'")
    else:
        # Fallback to predicted if GT not available
        top_activations = np.array([concept_probs[i][top_concepts[i]] for i in range(len(top_concepts))])
        print("⚠️ Warning: Using PREDICTED concept activations (ground truth not available)")
        activation_label = "Predicted Top Concept Activation"
        print(f"   Activation label: '{activation_label}'")
    
    # Compute improvement
    improvement = errors_image_only - errors_both  # Positive = helps
    
    helps_mask = improvement > 0
    hurts_mask = improvement <= 0
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ===== Panel 1: Scatter plot with trend line =====
    ax1 = axes[0, 0]
    
    # Sample for visualization (too many points can be slow)
    n_samples = min(2000, len(improvement))
    sample_idx = np.random.choice(len(improvement), n_samples, replace=False)
    
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvement[sample_idx]]
    ax1.scatter(top_activations[sample_idx], improvement[sample_idx], 
               c=colors, alpha=0.4, s=20, edgecolors='none')
    
    # Add trend line
    z = np.polyfit(top_activations, improvement, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, top_activations.max(), 100)
    ax1.plot(x_line, p(x_line), 'b-', linewidth=2, label=f'Trend: slope={z[0]:.1f}')
    
    # Add horizontal line at 0
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Compute correlation
    corr = np.corrcoef(top_activations, improvement)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: r={corr:.3f}', transform=ax1.transAxes,
            fontsize=11, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel(f'{activation_label}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Improvement (km) [+Helps, -Hurts]', fontsize=12, fontweight='bold')
    if "Ground Truth" in activation_label:
        ax1.set_title(f'{activation_label} vs Improvement\n(Does higher confidence in true concept help?)', fontsize=13, fontweight='bold')
    else:
        ax1.set_title(f'{activation_label} vs Improvement\n(Does higher confidence help?)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # ===== Panel 2: Box plots for helps vs hurts =====
    ax2 = axes[0, 1]
    
    helps_activations = top_activations[helps_mask]
    hurts_activations = top_activations[hurts_mask]
    
    bp = ax2.boxplot([hurts_activations, helps_activations], 
                     labels=['Hurts', 'Helps'],
                     patch_artist=True,
                     widths=0.6)
    
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#2ecc71')
    bp['boxes'][1].set_alpha(0.7)
    
    # Add means
    ax2.scatter([1], [np.mean(hurts_activations)], color='darkred', s=100, zorder=5, marker='D', label='Mean')
    ax2.scatter([2], [np.mean(helps_activations)], color='darkgreen', s=100, zorder=5, marker='D')
    
    # Add statistics text
    stats_text = (f'Hurts: μ={np.mean(hurts_activations):.3f}, med={np.median(hurts_activations):.3f}, n={len(hurts_activations)}\n'
                  f'Helps: μ={np.mean(helps_activations):.3f}, med={np.median(helps_activations):.3f}, n={len(helps_activations)}')
    ax2.text(0.5, 0.95, stats_text, transform=ax2.transAxes, fontsize=10, 
            va='top', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax2.set_ylabel(activation_label, fontsize=12, fontweight='bold')
    if "Ground Truth" in activation_label:
        ax2.set_title(f'{activation_label}: Helps vs Hurts\n(Higher = model more confident about true concept)', fontsize=13, fontweight='bold')
    else:
        ax2.set_title(f'{activation_label}: Helps vs Hurts\n(Higher = model more confident)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== Panel 3: Binned analysis =====
    ax3 = axes[1, 0]
    
    # Create activation bins
    bins = np.linspace(0, top_activations.max() * 1.01, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    binned_improvements = []
    binned_help_rates = []
    binned_counts = []
    
    for i in range(len(bins) - 1):
        mask = (top_activations >= bins[i]) & (top_activations < bins[i+1])
        if np.sum(mask) > 0:
            binned_improvements.append(np.mean(improvement[mask]))
            binned_help_rates.append(np.mean(improvement[mask] > 0) * 100)
            binned_counts.append(np.sum(mask))
        else:
            binned_improvements.append(0)
            binned_help_rates.append(0)
            binned_counts.append(0)
    
    # Bar chart of improvement by bin
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in binned_improvements]
    bars = ax3.bar(bin_centers, binned_improvements, width=(bins[1]-bins[0])*0.8, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add count labels
    for bar, count, imp in zip(bars, binned_counts, binned_improvements):
        if count > 0:
            y_pos = imp + (5 if imp >= 0 else -10)
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos, f'n={count}', 
                    ha='center', va='bottom' if imp >= 0 else 'top', fontsize=8)
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel(f'{activation_label} (Binned)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Improvement (km)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Mean Improvement by {activation_label} Range\n(Positive = Concepts Help)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== Panel 4: Success rate by activation bin =====
    ax4 = axes[1, 1]
    
    bars = ax4.bar(bin_centers, binned_help_rates, width=(bins[1]-bins[0])*0.8, 
                   color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add 50% reference line
    ax4.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% (random)')
    
    # Overall success rate
    overall_rate = np.mean(helps_mask) * 100
    ax4.axhline(y=overall_rate, color='orange', linestyle='--', linewidth=2, 
               label=f'Overall: {overall_rate:.1f}%')
    
    ax4.set_xlabel(f'{activation_label} (Binned)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('% Samples Where Concepts Help', fontsize=12, fontweight='bold')
    ax4.set_title(f'Success Rate by {activation_label} Range\n(Higher = concepts help more often)', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    if "Ground Truth" in activation_label:
        plt.suptitle('Ground Truth Concept Activation Analysis:\nDoes High Confidence in the True Concept Predict Success?', 
                     fontsize=16, fontweight='bold', y=1.02)
    else:
        plt.suptitle('Concept Activation Analysis:\nDoes High Confidence Predict Success?', 
                     fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    activation_path = output_path.parent / f"{output_path.stem}_activation_analysis.png"
    plt.savefig(activation_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved activation analysis to: {activation_path}")
    
    # Print key insights
    print("\n📊 ACTIVATION ANALYSIS SUMMARY:")
    print(f"   Correlation (activation vs improvement): r = {corr:.3f}")
    print(f"   Mean activation when HELPS: {np.mean(helps_activations):.3f}")
    print(f"   Mean activation when HURTS: {np.mean(hurts_activations):.3f}")
    diff = np.mean(helps_activations) - np.mean(hurts_activations)
    if diff > 0.01:
        print(f"   ✅ Higher activation correlates with BETTER predictions (+{diff:.3f})")
    elif diff < -0.01:
        print(f"   ⚠️ Higher activation correlates with WORSE predictions ({diff:.3f})")
    else:
        print(f"   ➖ No significant difference in activation between helps/hurts")
    
    return activation_path


def visualize_gt_concept_rank_analysis(
    errors_both: np.ndarray,
    errors_image_only: np.ndarray,
    concept_probs: np.ndarray,
    idx_to_concept: Dict[int, str],
    output_path: Path,
    true_lats: np.ndarray,
    true_lngs: np.ndarray,
    test_csv: Optional[Path] = None,
    concept_vocab_path: Optional[Path] = None,
):
    """
    Visualize the relationship between the RANK of the ground truth concept 
    (1st most activated, 2nd, 3rd, etc.) and whether concepts help.
    
    Creates a multi-panel plot showing:
    1. Distribution of ranks (histogram)
    2. Mean improvement by rank
    3. Success rate (% helps) by rank
    4. Scatter: rank vs improvement
    5. Box plot: improvement distribution for different ranks
    """
    if concept_probs is None or len(concept_probs) == 0:
        print("Warning: No concept probabilities available for rank analysis")
        return
    
    # Load ground truth concept labels from test CSV and match by coordinates
    gt_concept_labels = None
    
    print(f"\n🔍 Loading ground truth concept labels for rank analysis...")
    
    if test_csv and test_csv.exists() and concept_vocab_path and concept_vocab_path.exists():
        try:
            df = pd.read_csv(test_csv)
            with open(concept_vocab_path) as f:
                concept_vocab = json.load(f)
            concept_to_idx = {str(v): int(k) for k, v in concept_vocab["idx_to_concept"].items()}
            
            # Get concept column
            concept_col = None
            if 'generalized' in df.columns:
                concept_col = 'generalized'
            elif 'meta_name' in df.columns:
                concept_col = 'meta_name'
            
            if concept_col:
                # Match samples by coordinates (lat, lng)
                # Create a lookup from (lat, lng) -> concept_idx
                print(f"   Matching samples by coordinates...")
                coord_to_concept = {}
                lat_col = 'lat' if 'lat' in df.columns else 'latitude'
                lng_col = 'lng' if 'lng' in df.columns else 'longitude'
                
                for _, row in df.iterrows():
                    concept_name = str(row[concept_col])
                    concept_idx = concept_to_idx.get(concept_name, -1)
                    # Round to 4 decimal places for matching (about 11m precision)
                    lat_key = round(float(row[lat_col]), 4)
                    lng_key = round(float(row[lng_col]), 4)
                    coord_to_concept[(lat_key, lng_key)] = concept_idx
                
                # Match predictions to CSV by coordinates
                gt_concept_labels = []
                matched = 0
                for i in range(len(true_lats)):
                    lat_key = round(float(true_lats[i]), 4)
                    lng_key = round(float(true_lngs[i]), 4)
                    concept_idx = coord_to_concept.get((lat_key, lng_key), -1)
                    if concept_idx >= 0:
                        matched += 1
                    gt_concept_labels.append(concept_idx)
                
                gt_concept_labels = np.array(gt_concept_labels)
                print(f"   Matched {matched}/{len(true_lats)} samples by coordinates ({matched/len(true_lats)*100:.1f}%)")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load ground truth labels: {e}")
            import traceback
            traceback.print_exc()
            gt_concept_labels = None
    
    if gt_concept_labels is None:
        print("⚠️ Cannot perform rank analysis without ground truth labels")
        return
    
    # Compute rank of GT concept for each sample
    # Rank 1 = most activated, rank 2 = second most, etc.
    gt_ranks = []
    valid_mask = np.zeros(len(gt_concept_labels), dtype=bool)
    
    for i in range(len(concept_probs)):
        gt_idx = gt_concept_labels[i]
        if gt_idx < 0 or gt_idx >= len(concept_probs[i]):
            gt_ranks.append(-1)  # Invalid
            continue
        
        # Get activation of GT concept
        gt_activation = concept_probs[i][gt_idx]
        
        # Count how many concepts have higher activation
        # (using argsort: higher activation = lower index in sorted order)
        sorted_indices = np.argsort(concept_probs[i])[::-1]  # Descending order
        rank = np.where(sorted_indices == gt_idx)[0][0] + 1  # 1-indexed
        
        gt_ranks.append(rank)
        valid_mask[i] = True
    
    gt_ranks = np.array(gt_ranks)
    
    # Filter to valid samples
    valid_indices = valid_mask & (gt_ranks > 0)
    gt_ranks_valid = gt_ranks[valid_indices]
    
    if len(gt_ranks_valid) == 0:
        print("⚠️ No valid rank data available")
        return
    
    # Compute improvement for valid samples
    improvement = errors_image_only - errors_both  # Positive = helps
    improvement_valid = improvement[valid_indices]
    helps_mask = improvement_valid > 0
    
    # Also compute concept prediction accuracy for verification
    # (Concept accuracy = when pred matches GT = when GT is rank 1)
    top_predictions = np.argmax(concept_probs, axis=1)
    concept_accuracy = np.mean(top_predictions[valid_mask] == gt_concept_labels[valid_mask]) * 100
    
    print(f"✓ Computed ranks for {len(gt_ranks_valid)} samples")
    print(f"   Rank distribution: min={gt_ranks_valid.min()}, max={gt_ranks_valid.max()}, mean={gt_ranks_valid.mean():.1f}")
    print(f"   Concept Accuracy (pred == GT): {concept_accuracy:.1f}%")
    print(f"   Rank 1 percentage (GT is top): {np.mean(gt_ranks_valid == 1) * 100:.1f}%")
    if abs(concept_accuracy - np.mean(gt_ranks_valid == 1) * 100) > 1.0:
        print(f"   ⚠️ WARNING: Accuracy and Rank 1% should match! Difference: {abs(concept_accuracy - np.mean(gt_ranks_valid == 1) * 100):.1f}%")
    else:
        print(f"   ✓ Accuracy and Rank 1% match as expected!")
    
    # Create figure with 2x3 subplots
    fig = plt.figure(figsize=(20, 14))
    
    # ===== Panel 1: Rank distribution (histogram) =====
    ax1 = fig.add_subplot(2, 3, 1)
    max_rank = min(20, int(gt_ranks_valid.max()))  # Show up to rank 20
    bins = np.arange(0.5, max_rank + 1.5, 1)
    counts, _, _ = ax1.hist(gt_ranks_valid[gt_ranks_valid <= max_rank], bins=bins, 
                            color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        if count > 0:
            ax1.text(i + 1, count + max(counts) * 0.01, f'{int(count)}', 
                    ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('GT Concept Rank (1 = Most Activated)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Ground Truth Concept Ranks\n(How often is the true concept top-ranked?)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(range(1, max_rank + 1))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    pct_top1 = np.mean(gt_ranks_valid == 1) * 100
    pct_top3 = np.mean(gt_ranks_valid <= 3) * 100
    pct_top5 = np.mean(gt_ranks_valid <= 5) * 100
    stats_text = f'Top 1: {pct_top1:.1f}%\nTop 3: {pct_top3:.1f}%\nTop 5: {pct_top5:.1f}%'
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
            va='top', ha='right', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # ===== Panel 2: Mean improvement by rank =====
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Group by rank
    rank_improvements = {}
    rank_counts = {}
    for rank, imp in zip(gt_ranks_valid, improvement_valid):
        if rank not in rank_improvements:
            rank_improvements[rank] = []
            rank_counts[rank] = 0
        rank_improvements[rank].append(imp)
        rank_counts[rank] += 1
    
    # Get ranks up to max_rank
    ranks_to_plot = sorted([r for r in rank_improvements.keys() if r <= max_rank])
    mean_improvements = [np.mean(rank_improvements[r]) for r in ranks_to_plot]
    counts_by_rank = [rank_counts[r] for r in ranks_to_plot]
    
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in mean_improvements]
    bars = ax2.bar(ranks_to_plot, mean_improvements, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    # Add count labels
    for bar, count, imp in zip(bars, counts_by_rank, mean_improvements):
        y_pos = imp + (20 if imp >= 0 else -30)
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f'n={count}', 
                ha='center', va='bottom' if imp >= 0 else 'top', fontsize=8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('GT Concept Rank', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Improvement (km)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Improvement by GT Concept Rank\n(Positive = Concepts Help)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(ranks_to_plot)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== Panel 3: Success rate by rank =====
    ax3 = fig.add_subplot(2, 3, 3)
    
    rank_help_rates = {}
    for rank in ranks_to_plot:
        mask = gt_ranks_valid == rank
        if np.sum(mask) > 0:
            rank_help_rates[rank] = np.mean(improvement_valid[mask] > 0) * 100
    
    help_rates = [rank_help_rates.get(r, 0) for r in ranks_to_plot]
    
    bars = ax3.bar(ranks_to_plot, help_rates, color='steelblue', alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    # Add 50% reference line
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% (random)')
    
    # Overall success rate
    overall_rate = np.mean(helps_mask) * 100
    ax3.axhline(y=overall_rate, color='orange', linestyle='--', linewidth=2, 
               label=f'Overall: {overall_rate:.1f}%')
    
    # Add count labels
    for bar, count in zip(bars, counts_by_rank):
        if count > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'n={count}', 
                    ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('GT Concept Rank', fontsize=12, fontweight='bold')
    ax3.set_ylabel('% Samples Where Concepts Help', fontsize=12, fontweight='bold')
    ax3.set_title('Success Rate by GT Concept Rank\n(Higher = concepts help more often)', 
                  fontsize=13, fontweight='bold')
    ax3.set_xticks(ranks_to_plot)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== Panel 4: Scatter plot: rank vs improvement =====
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Sample for visualization if too many points
    n_samples = min(2000, len(improvement_valid))
    sample_idx = np.random.choice(len(improvement_valid), n_samples, replace=False)
    
    colors_scatter = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvement_valid[sample_idx]]
    ax4.scatter(gt_ranks_valid[sample_idx], improvement_valid[sample_idx], 
               c=colors_scatter, alpha=0.4, s=20, edgecolors='none')
    
    # Add trend line
    z = np.polyfit(gt_ranks_valid, improvement_valid, 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, min(20, gt_ranks_valid.max()), 100)
    ax4.plot(x_line, p(x_line), 'b-', linewidth=2, label=f'Trend: slope={z[0]:.1f}')
    
    # Add horizontal line at 0
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Compute correlation
    corr = np.corrcoef(gt_ranks_valid, improvement_valid)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: r={corr:.3f}', transform=ax4.transAxes,
            fontsize=11, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax4.set_xlabel('GT Concept Rank (1 = Most Activated)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Improvement (km) [+Helps, -Hurts]', fontsize=12, fontweight='bold')
    ax4.set_title('GT Concept Rank vs Improvement\n(Does higher rank = better geolocation?)', 
                  fontsize=13, fontweight='bold')
    ax4.set_xlim(0.5, min(20.5, gt_ranks_valid.max() + 0.5))
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # ===== Panel 5: Box plot: improvement by rank groups =====
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Group ranks: 1, 2, 3, 4-5, 6-10, 11-20, 21+
    rank_groups = {
        'Rank 1': (gt_ranks_valid == 1),
        'Rank 2': (gt_ranks_valid == 2),
        'Rank 3': (gt_ranks_valid == 3),
        'Rank 4-5': (gt_ranks_valid >= 4) & (gt_ranks_valid <= 5),
        'Rank 6-10': (gt_ranks_valid >= 6) & (gt_ranks_valid <= 10),
        'Rank 11-20': (gt_ranks_valid >= 11) & (gt_ranks_valid <= 20),
        'Rank 21+': (gt_ranks_valid > 20),
    }
    
    # Filter groups that have data
    rank_groups_filtered = {k: v for k, v in rank_groups.items() if np.sum(v) > 0}
    group_labels = list(rank_groups_filtered.keys())
    group_data = [improvement_valid[mask] for mask in rank_groups_filtered.values()]
    
    bp = ax5.boxplot(group_data, labels=group_labels, patch_artist=True, widths=0.6)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    
    # Add means
    means = [np.mean(data) for data in group_data]
    ax5.scatter(range(1, len(means) + 1), means, color='darkred', s=100, zorder=5, 
               marker='D', label='Mean')
    
    # Add statistics text
    stats_lines = []
    for label, data in zip(group_labels, group_data):
        n = len(data)
        mean = np.mean(data)
        med = np.median(data)
        stats_lines.append(f'{label}: n={n}, μ={mean:.0f}, med={med:.0f}')
    
    stats_text = '\n'.join(stats_lines)
    ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes, fontsize=9, 
            va='top', ha='left', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax5.set_ylabel('Improvement (km)', fontsize=12, fontweight='bold')
    ax5.set_title('Improvement Distribution by Rank Group\n(Positive = Concepts Help)', 
                  fontsize=13, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ===== Panel 6: Summary statistics =====
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Compute statistics by rank
    rank1_mask = gt_ranks_valid == 1
    rank2_mask = gt_ranks_valid == 2
    rank3_mask = gt_ranks_valid == 3
    rank_low_mask = (gt_ranks_valid >= 4) & (gt_ranks_valid <= 10)
    rank_high_mask = gt_ranks_valid > 10
    
    summary_text = f"""
    RANK ANALYSIS SUMMARY
    {'='*50}
    
    Total Valid Samples: {len(gt_ranks_valid):,}
    
    Rank Distribution:
      Rank 1 (top):     {np.sum(rank1_mask):,} ({np.mean(rank1_mask)*100:.1f}%)
      Rank 2:           {np.sum(rank2_mask):,} ({np.mean(rank2_mask)*100:.1f}%)
      Rank 3:           {np.sum(rank3_mask):,} ({np.mean(rank3_mask)*100:.1f}%)
      Rank 4-10:        {np.sum(rank_low_mask):,} ({np.mean(rank_low_mask)*100:.1f}%)
      Rank 11+:         {np.sum(rank_high_mask):,} ({np.mean(rank_high_mask)*100:.1f}%)
    
    Mean Improvement by Rank:
      Rank 1:           {np.mean(improvement_valid[rank1_mask]):.1f} km
      Rank 2:           {np.mean(improvement_valid[rank2_mask]):.1f} km
      Rank 3:           {np.mean(improvement_valid[rank3_mask]):.1f} km
      Rank 4-10:        {np.mean(improvement_valid[rank_low_mask]):.1f} km
      Rank 11+:         {np.mean(improvement_valid[rank_high_mask]):.1f} km
    
    Success Rate by Rank:
      Rank 1:         {np.mean(improvement_valid[rank1_mask] > 0)*100:.1f}%
      Rank 2:          {np.mean(improvement_valid[rank2_mask] > 0)*100:.1f}%
      Rank 3:          {np.mean(improvement_valid[rank3_mask] > 0)*100:.1f}%
      Rank 4-10:       {np.mean(improvement_valid[rank_low_mask] > 0)*100:.1f}%
      Rank 11+:        {np.mean(improvement_valid[rank_high_mask] > 0)*100:.1f}%
    
    Correlation (rank vs improvement):
      r = {corr:.3f}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    plt.suptitle('Ground Truth Concept Rank Analysis:\nDoes the Position of the True Concept in the Activation Ranking Matter?', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    rank_path = output_path.parent / f"{output_path.stem}_rank_analysis.png"
    plt.savefig(rank_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved rank analysis to: {rank_path}")
    
    # Print key insights
    print("\n📊 RANK ANALYSIS SUMMARY:")
    print(f"   Correlation (rank vs improvement): r = {corr:.3f}")
    print(f"   Rank 1 samples: {np.sum(rank1_mask):,} ({np.mean(rank1_mask)*100:.1f}%)")
    print(f"   Rank 1 mean improvement: {np.mean(improvement_valid[rank1_mask]):.1f} km")
    print(f"   Rank 1 success rate: {np.mean(improvement_valid[rank1_mask] > 0)*100:.1f}%")
    if np.sum(rank2_mask) > 0:
        print(f"   Rank 2 mean improvement: {np.mean(improvement_valid[rank2_mask]):.1f} km")
        print(f"   Rank 2 success rate: {np.mean(improvement_valid[rank2_mask] > 0)*100:.1f}%")
    
    return rank_path


def compute_per_concept_stats(
    errors_both: np.ndarray,
    errors_image_only: np.ndarray,
    top_concepts: np.ndarray,
    idx_to_concept: Dict[int, str],
    train_counts: Optional[Dict[int, int]] = None,
) -> Dict[int, Dict]:
    """
    Compute per-concept statistics for when concepts help vs. hurt.
    
    Returns:
        Dict mapping concept_idx to statistics
    """
    improvement = errors_image_only - errors_both  # Positive = concepts help
    
    concept_stats = {}
    unique_concepts = np.unique(top_concepts)
    
    for concept_idx in unique_concepts:
        mask = top_concepts == concept_idx
        concept_improvements = improvement[mask]
        concept_errors_both = errors_both[mask]
        concept_errors_img = errors_image_only[mask]
        
        n_samples = np.sum(mask)
        n_helps = np.sum(concept_improvements > 0)
        n_hurts = np.sum(concept_improvements < 0)
        
        concept_stats[int(concept_idx)] = {
            'name': idx_to_concept.get(int(concept_idx), f'concept_{concept_idx}'),
            'n_samples': int(n_samples),
            'n_helps': int(n_helps),
            'n_hurts': int(n_hurts),
            'pct_helps': float(n_helps / n_samples * 100) if n_samples > 0 else 0,
            'mean_improvement': float(np.mean(concept_improvements)),
            'median_improvement': float(np.median(concept_improvements)),
            'std_improvement': float(np.std(concept_improvements)),
            'mean_error_both': float(np.mean(concept_errors_both)),
            'mean_error_image_only': float(np.mean(concept_errors_img)),
            'total_improvement': float(np.sum(concept_improvements)),
            'train_count': train_counts.get(int(concept_idx), 0) if train_counts else 0,
        }
    
    return concept_stats


def compute_sample_level_stats(
    errors_both: np.ndarray,
    errors_image_only: np.ndarray,
    top_concepts: np.ndarray,
    idx_to_concept: Dict[int, str],
) -> Dict:
    """
    Compute sample-level statistics for when concepts help vs. hurt.
    """
    improvement = errors_image_only - errors_both
    
    helps_mask = improvement > 0
    hurts_mask = improvement < 0
    neutral_mask = np.abs(improvement) < 1  # Within 1km = neutral
    
    # Concept distribution when helps vs. hurts
    helps_concepts = top_concepts[helps_mask]
    hurts_concepts = top_concepts[hurts_mask]
    
    helps_concept_counts = Counter(helps_concepts)
    hurts_concept_counts = Counter(hurts_concepts)
    
    # Convert to concept names
    helps_concept_names = {idx_to_concept.get(int(k), f'concept_{k}'): v 
                           for k, v in helps_concept_counts.most_common(20)}
    hurts_concept_names = {idx_to_concept.get(int(k), f'concept_{k}'): v 
                           for k, v in hurts_concept_counts.most_common(20)}
    
    return {
        'total_samples': len(improvement),
        'n_helps': int(np.sum(helps_mask)),
        'n_hurts': int(np.sum(hurts_mask)),
        'n_neutral': int(np.sum(neutral_mask)),
        'pct_helps': float(np.mean(helps_mask) * 100),
        'pct_hurts': float(np.mean(hurts_mask) * 100),
        'mean_improvement_overall': float(np.mean(improvement)),
        'median_improvement_overall': float(np.median(improvement)),
        'mean_improvement_when_helps': float(np.mean(improvement[helps_mask])) if np.sum(helps_mask) > 0 else 0,
        'median_improvement_when_helps': float(np.median(improvement[helps_mask])) if np.sum(helps_mask) > 0 else 0,
        'mean_degradation_when_hurts': float(np.mean(improvement[hurts_mask])) if np.sum(hurts_mask) > 0 else 0,
        'median_degradation_when_hurts': float(np.median(improvement[hurts_mask])) if np.sum(hurts_mask) > 0 else 0,
        'max_improvement': float(np.max(improvement)),
        'max_degradation': float(np.min(improvement)),
        'top_concepts_when_helps': helps_concept_names,
        'top_concepts_when_hurts': hurts_concept_names,
        'mean_error_both': float(np.mean(errors_both)),
        'mean_error_image_only': float(np.mean(errors_image_only)),
        'median_error_both': float(np.median(errors_both)),
        'median_error_image_only': float(np.median(errors_image_only)),
    }


def visualize_comprehensive_analysis(
    errors_both: np.ndarray,
    errors_image_only: np.ndarray,
    top_concepts: np.ndarray,
    idx_to_concept: Dict[int, str],
    concept_stats: Dict[int, Dict],
    sample_stats: Dict,
    output_path: Path,
):
    """
    Create comprehensive visualization with multiple analysis plots.
    """
    fig = plt.figure(figsize=(24, 20))
    
    improvement = errors_image_only - errors_both
    
    # ============== ROW 1: Overall Statistics ==============
    
    # 1. Sample-level improvement distribution
    ax1 = fig.add_subplot(3, 3, 1)
    bins = np.linspace(-500, 500, 51)
    helps_mask = improvement > 0
    hurts_mask = improvement < 0
    
    ax1.hist(improvement[helps_mask], bins=bins, alpha=0.7, color='green', label=f'Helps (n={np.sum(helps_mask)})')
    ax1.hist(improvement[hurts_mask], bins=bins, alpha=0.7, color='red', label=f'Hurts (n={np.sum(hurts_mask)})')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax1.axvline(x=np.mean(improvement), color='blue', linestyle='-', linewidth=2, label=f'Mean: {np.mean(improvement):.1f} km')
    ax1.set_xlabel('Improvement (km)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Sample-Level Improvement Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Summary statistics box
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.axis('off')
    summary_text = f"""
    OVERALL STATISTICS
    {'='*40}
    
    Total Samples: {sample_stats['total_samples']:,}
    
    Concepts Help: {sample_stats['n_helps']:,} ({sample_stats['pct_helps']:.1f}%)
    Concepts Hurt: {sample_stats['n_hurts']:,} ({sample_stats['pct_hurts']:.1f}%)
    
    Improvement:
      Mean:  {sample_stats['mean_improvement_overall']:.1f} km
      Median: {sample_stats['median_improvement_overall']:.1f} km
    
    When Helps:
      Mean:  {sample_stats['mean_improvement_when_helps']:.1f} km
      Median: {sample_stats['median_improvement_when_helps']:.1f} km
      Max:   {sample_stats['max_improvement']:.1f} km
    
    When Hurts:
      Mean:  {sample_stats['mean_degradation_when_hurts']:.1f} km
      Median: {sample_stats['median_degradation_when_hurts']:.1f} km
      Max:   {sample_stats['max_degradation']:.1f} km
    
    Error Comparison:
      Image+Concepts:
        Mean:   {sample_stats['mean_error_both']:.1f} km
        Median: {sample_stats['median_error_both']:.1f} km
      Image Only:
        Mean:   {sample_stats['mean_error_image_only']:.1f} km
        Median: {sample_stats['median_error_image_only']:.1f} km
    """
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax2.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    # 3. Scatter: Image Only error vs. Image+Concepts error
    ax3 = fig.add_subplot(3, 3, 3)
    max_err = min(5000, max(errors_both.max(), errors_image_only.max()))
    
    ax3.scatter(errors_image_only[helps_mask], errors_both[helps_mask], 
                alpha=0.3, c='green', s=10, label='Concepts Help', rasterized=True)
    ax3.scatter(errors_image_only[hurts_mask], errors_both[hurts_mask], 
                alpha=0.3, c='red', s=10, label='Concepts Hurt', rasterized=True)
    ax3.plot([0, max_err], [0, max_err], 'k--', linewidth=2, label='Equal')
    ax3.set_xlabel('Image Only Error (km)', fontsize=11)
    ax3.set_ylabel('Image+Concepts Error (km)', fontsize=11)
    ax3.set_title('Error Comparison', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, max_err)
    ax3.set_ylim(0, max_err)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ============== ROW 2: Per-Concept Analysis ==============
    
    # 4. Top concepts that HELP most (by mean improvement)
    ax4 = fig.add_subplot(3, 3, 4)
    sorted_by_help = sorted(concept_stats.items(), key=lambda x: x[1]['mean_improvement'], reverse=True)
    top_helpers = [(k, v) for k, v in sorted_by_help if v['n_samples'] >= 10][:15]
    
    if top_helpers:
        names = [v['name'].replace('_', ' ').title()[:25] for _, v in top_helpers]
        values = [v['mean_improvement'] for _, v in top_helpers]
        medians = [v['median_improvement'] for _, v in top_helpers]
        train_counts_list = [v['train_count'] for _, v in top_helpers]
        
        y_pos = np.arange(len(names))
        colors = ['green' if v > 0 else 'red' for v in values]
        bars = ax4.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f'{n} (train={t})' for n, t in zip(names, train_counts_list)], fontsize=8)
        ax4.set_xlabel('Mean Improvement (km)', fontsize=11)
        ax4.set_title('Concepts That HELP Most\n(Mean shown, median in text)', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add median values as text annotations
        for i, (bar, median_val) in enumerate(zip(bars, medians)):
            ax4.text(bar.get_width() + (5 if bar.get_width() > 0 else -5), 
                    bar.get_y() + bar.get_height()/2,
                    f'med={median_val:.1f}', fontsize=7, 
                    ha='left' if bar.get_width() > 0 else 'right', va='center', 
                    style='italic', color='gray')
    
    # 5. Top concepts that HURT most (by mean improvement)
    ax5 = fig.add_subplot(3, 3, 5)
    sorted_by_hurt = sorted(concept_stats.items(), key=lambda x: x[1]['mean_improvement'])
    top_hurters = [(k, v) for k, v in sorted_by_hurt if v['n_samples'] >= 10][:15]
    
    if top_hurters:
        names = [v['name'].replace('_', ' ').title()[:25] for _, v in top_hurters]
        values = [v['mean_improvement'] for _, v in top_hurters]
        medians = [v['median_improvement'] for _, v in top_hurters]
        train_counts_list = [v['train_count'] for _, v in top_hurters]
        
        y_pos = np.arange(len(names))
        colors = ['green' if v > 0 else 'red' for v in values]
        bars = ax5.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([f'{n} (train={t})' for n, t in zip(names, train_counts_list)], fontsize=8)
        ax5.set_xlabel('Mean Improvement (km)', fontsize=11)
        ax5.set_title('Concepts That HURT Most\n(Mean shown, median in text)', fontsize=12, fontweight='bold')
        ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Add median values as text annotations
        for i, (bar, median_val) in enumerate(zip(bars, medians)):
            ax5.text(bar.get_width() + (5 if bar.get_width() > 0 else -5), 
                    bar.get_y() + bar.get_height()/2,
                    f'med={median_val:.1f}', fontsize=7, 
                    ha='left' if bar.get_width() > 0 else 'right', va='center', 
                    style='italic', color='gray')
    
    # 6. Training count vs. performance correlation
    ax6 = fig.add_subplot(3, 3, 6)
    train_counts = [v['train_count'] for v in concept_stats.values() if v['train_count'] > 0]
    improvements = [v['mean_improvement'] for v in concept_stats.values() if v['train_count'] > 0]
    n_samples = [v['n_samples'] for v in concept_stats.values() if v['train_count'] > 0]
    
    if train_counts and improvements:
        # Size by test samples
        sizes = [min(200, s * 2) for s in n_samples]
        colors_by_imp = ['green' if imp > 0 else 'red' for imp in improvements]
        
        ax6.scatter(train_counts, improvements, s=sizes, c=colors_by_imp, alpha=0.6, edgecolors='black', linewidths=0.5)
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        # Compute correlation
        if len(train_counts) > 2:
            corr, p_val = scipy_stats.pearsonr(train_counts, improvements)
            ax6.set_title(f'Training Count vs. Improvement\n(r={corr:.3f}, p={p_val:.3f})', fontsize=12, fontweight='bold')
        else:
            ax6.set_title('Training Count vs. Improvement', fontsize=12, fontweight='bold')
        
        ax6.set_xlabel('Training Sample Count', fontsize=11)
        ax6.set_ylabel('Mean Improvement (km)', fontsize=11)
        ax6.grid(True, alpha=0.3)
    
    # ============== ROW 3: Concept Type Analysis ==============
    
    # 7. Concept category analysis
    ax7 = fig.add_subplot(3, 3, 7)
    
    def categorize_concept(concept_name: str) -> str:
        """
        Categorize a concept based on its name using prefix matching and specific rules.
        Uses prefix matching (startswith) for more precise categorization than substring matching.
        """
        name_lower = concept_name.lower()
        
        # Use prefix matching for main categories (more precise than substring)
        if name_lower.startswith('landscape_'):
            return 'Landscape'
        elif name_lower.startswith('road_'):
            return 'Road'
        elif name_lower.startswith('vegetation_'):
            return 'Vegetation'
        elif name_lower.startswith('script_'):
            return 'Script'
        elif name_lower.startswith('building_'):
            return 'Building'
        elif name_lower.startswith('car_'):
            return 'Car'
        elif name_lower.startswith('infrastructure_'):
            return 'Infrastructure'
        elif name_lower.startswith('pole_') or name_lower.startswith('bollard_'):
            return 'Infrastructure'
        # Specific infrastructure items
        elif name_lower in ['guardrail', 'hydrant', 'railroad', 'traffic_light', 
                           'tram', 'chevron', 'bin', 'post_sign','sticker','urban_artifacts']:
            return 'Infrastructure'
        # Sign-related (but not road_sign which is already caught above)
        elif name_lower.startswith('sign_'):
            return 'Infrastructure'
        # Other specific categories
        elif name_lower in ['camera_meta', 'phone_code','people_seastars', 'trekkers']:
            return 'Other'
        else:
            return 'Other'
    
    # Categorize concepts
    categories = defaultdict(lambda: {'n_samples': 0, 'total_improvement': 0, 'count': 0})
    for idx, stats in concept_stats.items():
        cat = categorize_concept(stats['name'])
        categories[cat]['n_samples'] += stats['n_samples']
        categories[cat]['total_improvement'] += stats['total_improvement']
        categories[cat]['count'] += 1
    
    # Compute average improvement per category
    cat_names = list(categories.keys())
    cat_improvements = [categories[c]['total_improvement'] / categories[c]['n_samples'] 
                       if categories[c]['n_samples'] > 0 else 0 for c in cat_names]
    cat_samples = [categories[c]['n_samples'] for c in cat_names]
    
    y_pos = np.arange(len(cat_names))
    colors = ['green' if v > 0 else 'red' for v in cat_improvements]
    bars = ax7.barh(y_pos, cat_improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels([f'{n} (n={s})' for n, s in zip(cat_names, cat_samples)], fontsize=9)
    ax7.set_xlabel('Mean Improvement (km)', fontsize=11)
    ax7.set_title('Improvement by Concept Category', fontsize=12, fontweight='bold')
    ax7.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. Success rate by concept (% of samples where concept helps)
    ax8 = fig.add_subplot(3, 3, 8)
    sorted_by_pct = sorted(concept_stats.items(), key=lambda x: x[1]['pct_helps'], reverse=True)
    top_by_pct = [(k, v) for k, v in sorted_by_pct if v['n_samples'] >= 20][:15]
    
    if top_by_pct:
        names = [v['name'].replace('_', ' ').title()[:25] for _, v in top_by_pct]
        pcts = [v['pct_helps'] for _, v in top_by_pct]
        train_counts_list = [v['train_count'] for _, v in top_by_pct]
        
        y_pos = np.arange(len(names))
        colors = ['green' if p > 50 else 'red' for p in pcts]
        bars = ax8.barh(y_pos, pcts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax8.set_yticks(y_pos)
        ax8.set_yticklabels([f'{n} (train={t})' for n, t in zip(names, train_counts_list)], fontsize=8)
        ax8.set_xlabel('% Samples Where Helps', fontsize=11)
        ax8.set_title('Success Rate by Concept (≥20 samples)', fontsize=12, fontweight='bold')
        ax8.axvline(x=50, color='black', linestyle='--', linewidth=1)
        ax8.set_xlim(0, 100)
        ax8.grid(True, alpha=0.3, axis='x')
    
    # 9. Training count vs. success rate
    ax9 = fig.add_subplot(3, 3, 9)
    train_counts = [v['train_count'] for v in concept_stats.values() if v['train_count'] > 0 and v['n_samples'] >= 5]
    pct_helps = [v['pct_helps'] for v in concept_stats.values() if v['train_count'] > 0 and v['n_samples'] >= 5]
    n_samples = [v['n_samples'] for v in concept_stats.values() if v['train_count'] > 0 and v['n_samples'] >= 5]
    
    if train_counts and pct_helps:
        sizes = [min(200, s * 2) for s in n_samples]
        colors_by_pct = ['green' if p > 50 else 'red' for p in pct_helps]
        
        ax9.scatter(train_counts, pct_helps, s=sizes, c=colors_by_pct, alpha=0.6, edgecolors='black', linewidths=0.5)
        ax9.axhline(y=50, color='black', linestyle='--', linewidth=1)
        
        # Compute correlation
        if len(train_counts) > 2:
            corr, p_val = scipy_stats.pearsonr(train_counts, pct_helps)
            ax9.set_title(f'Training Count vs. Success Rate\n(r={corr:.3f}, p={p_val:.3f})', fontsize=12, fontweight='bold')
        else:
            ax9.set_title('Training Count vs. Success Rate', fontsize=12, fontweight='bold')
        
        ax9.set_xlabel('Training Sample Count', fontsize=11)
        ax9.set_ylabel('% Samples Where Helps', fontsize=11)
        ax9.set_ylim(0, 100)
        ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    analysis_path = output_path.parent / f"{output_path.stem}_analysis.png"
    plt.savefig(analysis_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved comprehensive analysis to: {analysis_path}")
    return analysis_path


def visualize_concept_help_geography(
    country_stats: Dict[str, Dict],
    output_path: Path,
    min_samples: int = 5,
    train_counts: Optional[Dict[int, int]] = None,
    idx_to_concept: Optional[Dict[int, str]] = None,
    train_counts_by_country: Optional[Dict[str, int]] = None,
):
    """
    Visualize aggregated country-level concept help on world map.
    
    Args:
        country_stats: Dict mapping country code to aggregated stats
        output_path: Where to save visualization
        min_samples: Minimum number of samples per country to include
        train_counts: Training sample counts per concept
        idx_to_concept: Concept index to name mapping
        train_counts_by_country: Training sample counts per country
    """
    # Build reverse mapping: concept_name -> train_count
    concept_train_counts = {}
    if train_counts and idx_to_concept:
        for idx, name in idx_to_concept.items():
            if idx in train_counts:
                concept_train_counts[name] = train_counts[idx]
    
    # Default to empty dict if not provided
    if train_counts_by_country is None:
        train_counts_by_country = {}
    # Filter countries with enough samples
    filtered_stats = {k: v for k, v in country_stats.items() if v['num_samples'] >= min_samples}
    
    if len(filtered_stats) == 0:
        print(f"Warning: No countries with >= {min_samples} samples. Using all countries.")
        filtered_stats = country_stats
    
    print(f"Visualizing {len(filtered_stats)} countries")
    
    # Compute country-level summary statistics
    total_test_samples = sum(s['num_samples'] for s in filtered_stats.values())
    total_train_samples = sum(train_counts_by_country.get(c, 0) for c in filtered_stats.keys())
    helps_countries = sum(1 for s in filtered_stats.values() if s['mean_improvement'] > 0)
    hurts_countries = sum(1 for s in filtered_stats.values() if s['mean_improvement'] < 0)
    
    # Count unique concepts across countries
    all_concepts = set()
    for stats in filtered_stats.values():
        if stats['most_helpful_concept']:
            all_concepts.add(stats['most_helpful_concept'])
    
    # Compute total samples in countries where helps vs. hurts
    test_in_helps_countries = sum(s['num_samples'] for s in filtered_stats.values() if s['mean_improvement'] > 0)
    test_in_hurts_countries = sum(s['num_samples'] for s in filtered_stats.values() if s['mean_improvement'] < 0)
    train_in_helps_countries = sum(train_counts_by_country.get(c, 0) for c, s in filtered_stats.items() if s['mean_improvement'] > 0)
    train_in_hurts_countries = sum(train_counts_by_country.get(c, 0) for c, s in filtered_stats.items() if s['mean_improvement'] < 0)
    
    # Create figure - expanded to 3x2 layout
    fig = plt.figure(figsize=(28, 18))
    
    if HAS_CARTOPY:
        # Row 1: Main map (spans 2 columns) + Summary stats
        ax1 = fig.add_subplot(2, 3, (1, 2), projection=ccrs.Robinson())
        ax1.set_global()
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--', alpha=0.5)
        ax1.add_feature(cfeature.LAND, alpha=0.2, facecolor='#e8e8e8')
        ax1.add_feature(cfeature.OCEAN, alpha=0.2, facecolor='#d4e6f1')
        ax1.gridlines(draw_labels=False, alpha=0.3, linewidth=0.5)
        
        # Prepare data for plotting
        countries = list(filtered_stats.keys())
        improvements = [filtered_stats[c]['mean_improvement'] for c in countries]
        lats = [filtered_stats[c]['center_lat'] for c in countries]
        lngs = [filtered_stats[c]['center_lng'] for c in countries]
        sizes = [min(500, filtered_stats[c]['num_samples'] * 2) for c in countries]
        
        # Color by improvement (green = helps, red = hurts)
        scatter = ax1.scatter(
            lngs, lats,
            c=improvements, s=sizes, alpha=0.7, marker='o',
            cmap='RdYlGn', vmin=-50, vmax=50,
            transform=ccrs.PlateCarree(), zorder=3,
            edgecolors='black', linewidths=0.5
        )
        
        # Add country labels for significant improvements
        for country, stats in filtered_stats.items():
            if abs(stats['mean_improvement']) > 10 and stats['num_samples'] >= 10:
                ax1.text(
                    stats['center_lng'], stats['center_lat'] + 2,
                    country, transform=ccrs.PlateCarree(),
                    fontsize=8, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray', linewidth=0.5)
                )
        
        cbar = plt.colorbar(scatter, ax=ax1, label='Average Improvement (km)', shrink=0.8, pad=0.02)
        cbar.ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.set_title('Geographic Distribution: Where Concepts Help Geolocation\n(Country-Level Aggregation, Marker Size = Sample Count)', 
                      fontsize=14, fontweight='bold', pad=20)
        
        # Summary statistics box (top right)
        ax_summary = fig.add_subplot(2, 3, 3)
        ax_summary.axis('off')
        
        summary_text = f"""
    COUNTRY-LEVEL STATISTICS
    {'='*45}
    
    Countries Analyzed: {len(filtered_stats)}
    Train Samples: {total_train_samples:,}
    Test Samples: {total_test_samples:,}
    Unique Concepts: {len(all_concepts)}
    
    Countries Where Concepts Help: {helps_countries} ({helps_countries/len(filtered_stats)*100:.1f}%)
    Countries Where Concepts Hurt: {hurts_countries} ({hurts_countries/len(filtered_stats)*100:.1f}%)
    
    Test samples in helps countries: {test_in_helps_countries:,} ({test_in_helps_countries/total_test_samples*100:.1f}%)
    Test samples in hurts countries: {test_in_hurts_countries:,} ({test_in_hurts_countries/total_test_samples*100:.1f}%)
    
    Train samples in helps countries: {train_in_helps_countries:,}
    Train samples in hurts countries: {train_in_hurts_countries:,}
    
    Mean Improvement (per country): {np.mean([s['mean_improvement'] for s in filtered_stats.values()]):.1f} km
    Median Improvement (per country): {np.median([s['mean_improvement'] for s in filtered_stats.values()]):.1f} km
    
    Weighted Mean Improvement:
      (by test samples): {sum(s['mean_improvement']*s['num_samples'] for s in filtered_stats.values())/total_test_samples:.1f} km
    """
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax_summary.set_title('Country-Level Summary', fontsize=12, fontweight='bold')
        
        # Row 2: Three subplots
        
        # Bottom left: Top countries by improvement (with median)
        ax2 = fig.add_subplot(2, 3, 4)
        sorted_countries = sorted(filtered_stats.items(), key=lambda x: x[1]['mean_improvement'], reverse=True)
        top_10 = sorted_countries[:10]
        
        top_countries = [c[0] for c in top_10]
        top_improvements = [c[1]['mean_improvement'] for c in top_10]
        top_medians = [c[1]['median_improvement'] for c in top_10]
        top_test_samples = [c[1]['num_samples'] for c in top_10]
        top_train_samples = [train_counts_by_country.get(c, 0) for c in top_countries]
        top_pct_helps = [c[1]['pct_helps'] for c in top_10]
        
        y_pos = np.arange(len(top_countries))
        colors = ['green' if imp > 0 else 'red' for imp in top_improvements]
        bars = ax2.barh(y_pos, top_improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Labels: country code, train samples, test samples, % helps
        labels = [f"{c} (train={t}, test={n}, {p:.0f}%)" for c, t, n, p in zip(top_countries, top_train_samples, top_test_samples, top_pct_helps)]
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel('Mean Improvement (km)', fontsize=11, fontweight='bold')
        ax2.set_title('Top 10 Countries\n(mean | median improvement)', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Extend x-axis to make room for annotations
        x_max = max(top_improvements) * 1.4
        ax2.set_xlim(right=x_max)
        
        # Add mean/median as single line outside bars
        for i, (bar, mean_val, med_val) in enumerate(zip(bars, top_improvements, top_medians)):
            ax2.text(x_max * 0.95, bar.get_y() + bar.get_height()/2,
                    f'{mean_val:.0f} | {med_val:.0f}', 
                    ha='right', va='center', fontsize=8, fontweight='bold')
        
        # Bottom middle: Bottom 10 countries (where concepts hurt most)
        ax4 = fig.add_subplot(2, 3, 5)
        bottom_10 = sorted_countries[-10:]
        
        bottom_countries = [c[0] for c in bottom_10]
        bottom_improvements = [c[1]['mean_improvement'] for c in bottom_10]
        bottom_medians = [c[1]['median_improvement'] for c in bottom_10]
        bottom_test_samples = [c[1]['num_samples'] for c in bottom_10]
        bottom_train_samples = [train_counts_by_country.get(c, 0) for c in bottom_countries]
        bottom_pct_helps = [c[1]['pct_helps'] for c in bottom_10]
        
        y_pos = np.arange(len(bottom_countries))
        colors = ['green' if imp > 0 else 'red' for imp in bottom_improvements]
        bars = ax4.barh(y_pos, bottom_improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Labels: country code, train samples, test samples, % helps
        labels = [f"{c} (train={t}, test={n}, {p:.0f}%)" for c, t, n, p in zip(bottom_countries, bottom_train_samples, bottom_test_samples, bottom_pct_helps)]
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels, fontsize=8)
        ax4.set_xlabel('Mean Improvement (km)', fontsize=11, fontweight='bold')
        ax4.set_title('Bottom 10 Countries\n(mean | median improvement)', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Extend x-axis to make room for annotations (on left side for negative values)
        x_min = min(bottom_improvements) * 1.4 if min(bottom_improvements) < 0 else 0
        ax4.set_xlim(left=x_min)
        
        # Add mean/median as single line outside bars
        for i, (bar, mean_val, med_val) in enumerate(zip(bars, bottom_improvements, bottom_medians)):
            ax4.text(x_min * 0.95 if x_min < 0 else bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f'{mean_val:.0f} | {med_val:.0f}', 
                    ha='left' if x_min >= 0 else 'left', va='center', fontsize=8, fontweight='bold')
        
        # Bottom right: Most helpful concepts by country (with training counts)
        ax3 = fig.add_subplot(2, 3, 6)
        
        # Count concept occurrences as most helpful
        concept_counts = defaultdict(int)
        concept_countries = defaultdict(list)
        for country, stats in filtered_stats.items():
            if stats['most_helpful_concept']:
                concept_counts[stats['most_helpful_concept']] += 1
                concept_countries[stats['most_helpful_concept']].append(country)
        
        if concept_counts:
            sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:12]
            concept_names = [name for name, _ in sorted_concepts]
            counts = [count for _, count in sorted_concepts]
            
            # Build labels with training counts
            labels_with_train = []
            for name in concept_names:
                display_name = name.replace('_', ' ').title()
                train_count = concept_train_counts.get(name, 0)
                if train_count > 0:
                    labels_with_train.append(f'{display_name} (train={train_count})')
                else:
                    labels_with_train.append(display_name)
            
            y_pos = np.arange(len(labels_with_train))
            bars = ax3.barh(y_pos, counts, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(labels_with_train, fontsize=8)
            ax3.set_xlabel('Number of Countries Where This Concept Helps Most', fontsize=10, fontweight='bold')
            ax3.set_title('Most Helpful Concepts Across Countries\n(with training sample counts)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add value labels at end of bars (not overlapping)
            for bar, val in zip(bars, counts):
                ax3.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                        f'{val}', ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No concept data available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Most Helpful Concepts Across Countries', fontsize=12, fontweight='bold')
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, wspace=0.35, hspace=0.3)
        
    else:
        # Fallback: simple plots
        countries = list(filtered_stats.keys())
        improvements = [filtered_stats[c]['mean_improvement'] for c in countries]
        lats = [filtered_stats[c]['center_lat'] for c in countries]
        lngs = [filtered_stats[c]['center_lng'] for c in countries]
        
        ax1 = fig.add_subplot(1, 2, 1)
        scatter = ax1.scatter(lngs, lats, c=improvements, cmap='RdYlGn', vmin=-50, vmax=50, s=100, alpha=0.7)
        plt.colorbar(scatter, ax=ax1, label='Improvement (km)')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Where Concepts Help')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(1, 2, 2)
        sorted_countries = sorted(filtered_stats.items(), key=lambda x: x[1]['mean_improvement'], reverse=True)[:15]
        top_countries = [c[0] for c in sorted_countries]
        top_improvements = [c[1]['mean_improvement'] for c in sorted_countries]
        ax2.barh(range(len(top_countries)), top_improvements, color=['green' if imp > 0 else 'red' for imp in top_improvements])
        ax2.set_yticks(range(len(top_countries)))
        ax2.set_yticklabels(top_countries)
        ax2.set_xlabel('Improvement (km)')
        ax2.set_title('Top Countries by Improvement')
        ax2.axvline(x=0, color='black', linestyle='--')
        ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("Country-Level Concept Help Statistics:")
    print(f"{'='*80}")
    print(f"Total countries analyzed: {len(filtered_stats)}")
    
    helps_countries = sum(1 for s in filtered_stats.values() if s['mean_improvement'] > 0)
    hurts_countries = sum(1 for s in filtered_stats.values() if s['mean_improvement'] < 0)
    
    print(f"Countries where concepts help: {helps_countries} ({helps_countries/len(filtered_stats)*100:.1f}%)")
    print(f"Countries where concepts hurt: {hurts_countries} ({hurts_countries/len(filtered_stats)*100:.1f}%)")
    
    if filtered_stats:
        avg_improvement = np.mean([s['mean_improvement'] for s in filtered_stats.values()])
        print(f"Overall average improvement: {avg_improvement:.2f} km")
        
        print(f"\nTop 5 countries where concepts help most:")
        sorted_by_help = sorted(filtered_stats.items(), key=lambda x: x[1]['mean_improvement'], reverse=True)[:5]
        for country, stats in sorted_by_help:
            print(f"  {country}: {stats['mean_improvement']:.1f} km improvement (n={stats['num_samples']}, "
                  f"concept: {stats['most_helpful_concept'] or 'N/A'})")


def load_training_counts(train_csv: Path, concept_vocab_path: Path) -> Dict[int, int]:
    """Load training sample counts per concept from training CSV."""
    if not train_csv or not train_csv.exists():
        return {}
    
    try:
        df = pd.read_csv(train_csv)
        
        # Load concept vocab to get concept_to_idx mapping
        with open(concept_vocab_path) as f:
            concept_vocab = json.load(f)
        concept_to_idx = {str(v): int(k) for k, v in concept_vocab["idx_to_concept"].items()}
        
        # Count samples per concept
        counts = {}
        if 'generalized' in df.columns:
            concept_counts = df['generalized'].value_counts()
            for concept_name, count in concept_counts.items():
                if str(concept_name) in concept_to_idx:
                    counts[concept_to_idx[str(concept_name)]] = int(count)
        elif 'meta_name' in df.columns:
            concept_counts = df['meta_name'].value_counts()
            for concept_name, count in concept_counts.items():
                if str(concept_name) in concept_to_idx:
                    counts[concept_to_idx[str(concept_name)]] = int(count)
        
        print(f"Loaded training counts for {len(counts)} concepts")
        return counts
    except Exception as e:
        print(f"Warning: Could not load training counts: {e}")
        return {}


def load_training_counts_by_country(train_csv: Path) -> Dict[str, int]:
    """Load training sample counts per country from training CSV."""
    if not train_csv or not train_csv.exists():
        return {}
    
    try:
        df = pd.read_csv(train_csv)
        
        # Check for required columns
        if 'lat' not in df.columns or 'lng' not in df.columns:
            print("Warning: Training CSV missing lat/lng columns")
            return {}
        
        # Filter valid coordinates
        df = df.dropna(subset=['lat', 'lng'])
        
        # Get countries for training coordinates
        print(f"Geocoding {len(df)} training samples to countries...")
        coords = [(float(row['lat']), float(row['lng'])) for _, row in df.iterrows()]
        countries = batch_get_countries(coords)
        
        # Count samples per country
        from collections import Counter
        country_counts = Counter(countries)
        # Remove None entries
        country_counts = {k: v for k, v in country_counts.items() if k is not None}
        
        print(f"Found training samples in {len(country_counts)} countries")
        return dict(country_counts)
    except Exception as e:
        print(f"Warning: Could not load training counts by country: {e}")
        return {}


def load_training_counts_by_concept_and_country(
    train_csv: Path, 
    concept_vocab_path: Path
) -> Dict[str, Dict[str, int]]:
    """
    Load training sample counts per concept per country.
    
    Returns:
        Dict mapping country_code -> Dict mapping concept_name -> count
    """
    if not train_csv or not train_csv.exists():
        return {}
    
    try:
        df = pd.read_csv(train_csv)
        
        # Check for required columns
        if 'lat' not in df.columns or 'lng' not in df.columns:
            print("Warning: Training CSV missing lat/lng columns")
            return {}
        
        # Load concept vocab
        with open(concept_vocab_path) as f:
            concept_vocab = json.load(f)
        concept_to_idx = {str(v): int(k) for k, v in concept_vocab["idx_to_concept"].items()}
        idx_to_concept = {int(k): str(v) for k, v in concept_vocab["idx_to_concept"].items()}
        
        # Get concept column
        concept_col = None
        if 'generalized' in df.columns:
            concept_col = 'generalized'
        elif 'meta_name' in df.columns:
            concept_col = 'meta_name'
        else:
            print("Warning: Training CSV missing concept column")
            return {}
        
        # Filter valid data
        df = df.dropna(subset=['lat', 'lng', concept_col])
        
        # Get countries for training coordinates
        print(f"Geocoding {len(df)} training samples to countries for concept-country mapping...")
        coords = [(float(row['lat']), float(row['lng'])) for _, row in df.iterrows()]
        countries = batch_get_countries(coords)
        
        # Build nested dict: country -> concept -> count
        country_concept_counts = defaultdict(lambda: defaultdict(int))
        
        for country, row in zip(countries, df.itertuples()):
            if country is None:
                continue
            
            concept_name = str(getattr(row, concept_col))
            # Map to concept name if needed
            if concept_name in concept_to_idx:
                concept_idx = concept_to_idx[concept_name]
                concept_name = idx_to_concept.get(concept_idx, concept_name)
            
            country_concept_counts[country][concept_name] += 1
        
        print(f"Found concept-country pairs for {len(country_concept_counts)} countries")
        return {k: dict(v) for k, v in country_concept_counts.items()}
    except Exception as e:
        print(f"Warning: Could not load training counts by concept and country: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True,
                       help="Directory with model comparison results (from compare_model_variants.py)")
    parser.add_argument("--concept-vocab", type=Path, required=True,
                       help="Concept vocabulary JSON file")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path for visualization")
    parser.add_argument("--train-csv", type=Path, default=None,
                       help="Training CSV for sample counts")
    parser.add_argument("--min-samples", type=int, default=5,
                       help="Minimum samples per country to include")
    parser.add_argument("--re-evaluate", action="store_true",
                       help="Re-run evaluation instead of loading saved results")
    
    # If re-evaluating, need model checkpoints
    parser.add_argument("--checkpoint-both", type=Path, default=None,
                       help="Checkpoint for Image+Concepts model (if re-evaluating)")
    parser.add_argument("--checkpoint-image-only", type=Path, default=None,
                       help="Checkpoint for Image Only model (if re-evaluating)")
    parser.add_argument("--test-csv", type=Path, default=None,
                       help="Test CSV path (if re-evaluating)")
    parser.add_argument("--cached-dir", type=Path, default=None,
                       help="Cached embeddings directory (if re-evaluating)")
    
    args = parser.parse_args()
    
    # Load concept vocabulary
    with open(args.concept_vocab) as f:
        concept_vocab = json.load(f)
    idx_to_concept = {int(k): v for k, v in concept_vocab["idx_to_concept"].items()}
    
    # Load training counts if available
    train_counts = {}
    train_counts_by_country = {}
    train_counts_by_concept_country = {}
    if args.train_csv:
        train_counts = load_training_counts(args.train_csv, args.concept_vocab)
        train_counts_by_country = load_training_counts_by_country(args.train_csv)
        train_counts_by_concept_country = load_training_counts_by_concept_and_country(args.train_csv, args.concept_vocab)
    
    # Try to load saved results
    results_file = args.results_dir / "results_dict.json"
    
    if args.re_evaluate or not results_file.exists():
        print("Results file not found. Need to re-evaluate models.")
        print("Please run compare_model_variants.py first, or use --re-evaluate with checkpoint paths.")
        return
    
    # Load results
    print(f"Loading results from {results_file}...")
    with open(results_file) as f:
        results_dict = json.load(f)
    
    # Extract data
    if 'both' not in results_dict or 'image_only' not in results_dict:
        print("Error: Results must include both 'both' and 'image_only' models")
        return
    
    both_results = results_dict['both']
    image_only_results = results_dict['image_only']
    
    # Get predictions
    both_preds = both_results['predictions']
    image_only_preds = image_only_results['predictions']
    
    true_lats = np.array([p['true_lat'] for p in both_preds])
    true_lngs = np.array([p['true_lng'] for p in both_preds])
    errors_both = np.array([p['error_km'] for p in both_preds])
    errors_image_only = np.array([p['error_km'] for p in image_only_preds])
    
    # Get top concepts
    top_concepts = np.array(both_results.get('concept_preds', []))
    if len(top_concepts) == 0:
        print("Warning: No concept predictions found. Using dummy values.")
        top_concepts = np.zeros(len(true_lats), dtype=int)
    
    # Get concept probabilities (activation scores)
    concept_probs = np.array(both_results.get('concept_probs', []))
    if len(concept_probs) == 0:
        print("Warning: No concept probabilities found.")
        concept_probs = None
    
    # ============== COMPREHENSIVE ANALYSIS ==============
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CONCEPT EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    # 1. Compute per-concept statistics
    print("\n1. Computing per-concept statistics...")
    concept_stats = compute_per_concept_stats(
        errors_both, errors_image_only, top_concepts, idx_to_concept, train_counts
    )
    
    # 2. Compute sample-level statistics
    print("2. Computing sample-level statistics...")
    sample_stats = compute_sample_level_stats(
        errors_both, errors_image_only, top_concepts, idx_to_concept
    )
    
    # 3. Create comprehensive analysis visualization
    print("3. Creating comprehensive analysis visualization...")
    visualize_comprehensive_analysis(
        errors_both, errors_image_only, top_concepts, idx_to_concept,
        concept_stats, sample_stats, args.output
    )
    
    # 4. Aggregate by country
    print("4. Aggregating by country...")
    country_stats = aggregate_by_country(
        true_lats, true_lngs,
        errors_both, errors_image_only,
        top_concepts, idx_to_concept,
        concept_probs=concept_probs
    )
    
    # 5. Create geographic visualization
    print("5. Creating geographic visualization...")
    visualize_concept_help_geography(
        country_stats, args.output, 
        min_samples=args.min_samples,
        train_counts=train_counts,
        idx_to_concept=idx_to_concept,
        train_counts_by_country=train_counts_by_country
    )
    
    # 6. Create country drill-down visualization (worst countries)
    print("6. Creating country drill-down visualization (worst countries)...")
    visualize_country_drilldown(
        country_stats, idx_to_concept,
        train_counts, train_counts_by_country,
        train_counts_by_concept_country,
        args.output, worst_n=6, include_mexico=True
    )
    
    # 7. Create top countries drill-down visualization
    print("7. Creating top countries drill-down visualization...")
    visualize_top_countries_drilldown(
        country_stats, idx_to_concept,
        train_counts, train_counts_by_country,
        train_counts_by_concept_country,
        args.output, top_n=6
    )
    
    # 8. Create activation vs help analysis
    print("8. Creating activation vs help analysis...")
    visualize_activation_vs_help(
        errors_both, errors_image_only, top_concepts, concept_probs,
        idx_to_concept, args.output,
        true_lats=true_lats, true_lngs=true_lngs,
        test_csv=args.test_csv,
        concept_vocab_path=args.concept_vocab
    )
    
    # 9. Create GT concept rank analysis
    print("9. Creating GT concept rank analysis...")
    visualize_gt_concept_rank_analysis(
        errors_both, errors_image_only, concept_probs,
        idx_to_concept, args.output,
        true_lats=true_lats, true_lngs=true_lngs,
        test_csv=args.test_csv,
        concept_vocab_path=args.concept_vocab
    )
    
    # ============== SAVE ALL STATISTICS ==============
    
    # Save country stats (without raw lists to keep file size reasonable)
    stats_file = args.output.parent / f"{args.output.stem}_stats.json"
    # Create a copy without the raw lists
    country_stats_for_save = {}
    for code, stats in country_stats.items():
        country_stats_for_save[code] = {k: v for k, v in stats.items() 
                                         if k not in ('concept_list', 'improvement_list')}
    with open(stats_file, 'w') as f:
        json.dump(country_stats_for_save, f, indent=2)
    
    # Save per-concept stats
    concept_stats_file = args.output.parent / f"{args.output.stem}_concept_stats.json"
    with open(concept_stats_file, 'w') as f:
        json.dump(concept_stats, f, indent=2)
    
    # Save sample-level stats
    sample_stats_file = args.output.parent / f"{args.output.stem}_sample_stats.json"
    with open(sample_stats_file, 'w') as f:
        json.dump(sample_stats, f, indent=2)
    
    # ============== PRINT SUMMARY ==============
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n📊 SAMPLE-LEVEL:")
    print(f"   Total samples: {sample_stats['total_samples']:,}")
    print(f"   Concepts help: {sample_stats['n_helps']:,} ({sample_stats['pct_helps']:.1f}%)")
    print(f"   Concepts hurt: {sample_stats['n_hurts']:,} ({sample_stats['pct_hurts']:.1f}%)")
    print(f"   Mean improvement: {sample_stats['mean_improvement_overall']:.1f} km")
    print(f"   When helps: avg +{sample_stats['mean_improvement_when_helps']:.1f} km")
    print(f"   When hurts: avg {sample_stats['mean_degradation_when_hurts']:.1f} km")
    
    print(f"\n📍 COUNTRY-LEVEL:")
    helps_countries = sum(1 for s in country_stats.values() if s['mean_improvement'] > 0)
    hurts_countries = sum(1 for s in country_stats.values() if s['mean_improvement'] < 0)
    print(f"   Countries where helps: {helps_countries}/{len(country_stats)}")
    print(f"   Countries where hurts: {hurts_countries}/{len(country_stats)}")
    
    print(f"\n🏆 TOP 5 CONCEPTS THAT HELP:")
    sorted_concepts = sorted(concept_stats.items(), key=lambda x: x[1]['mean_improvement'], reverse=True)
    for idx, (k, v) in enumerate(sorted_concepts[:5]):
        print(f"   {idx+1}. {v['name']}: +{v['mean_improvement']:.1f} km (n={v['n_samples']}, train={v['train_count']})")
    
    print(f"\n⚠️ TOP 5 CONCEPTS THAT HURT:")
    for idx, (k, v) in enumerate(sorted_concepts[-5:]):
        print(f"   {idx+1}. {v['name']}: {v['mean_improvement']:.1f} km (n={v['n_samples']}, train={v['train_count']})")
    
    print(f"\n📈 TOP CONCEPTS WHEN HELPS:")
    for concept, count in list(sample_stats['top_concepts_when_helps'].items())[:5]:
        print(f"   {concept}: {count} samples")
    
    print(f"\n📉 TOP CONCEPTS WHEN HURTS:")
    for concept, count in list(sample_stats['top_concepts_when_hurts'].items())[:5]:
        print(f"   {concept}: {count} samples")
    
    # Correlation analysis
    if train_counts:
        train_vals = [v['train_count'] for v in concept_stats.values() if v['train_count'] > 0]
        imp_vals = [v['mean_improvement'] for v in concept_stats.values() if v['train_count'] > 0]
        if len(train_vals) > 2:
            corr, p_val = scipy_stats.pearsonr(train_vals, imp_vals)
            print(f"\n📊 TRAINING COUNT CORRELATION:")
            print(f"   Correlation with improvement: r={corr:.3f} (p={p_val:.3f})")
            if p_val < 0.05:
                if corr > 0:
                    print(f"   → More training data → better improvement (significant)")
                else:
                    print(f"   → More training data → worse improvement (significant)")
            else:
                print(f"   → No significant correlation")
    
    print(f"\n✅ Saved files:")
    print(f"   - {args.output}")
    print(f"   - {args.output.parent / f'{args.output.stem}_analysis.png'}")
    print(f"   - {stats_file}")
    print(f"   - {concept_stats_file}")
    print(f"   - {sample_stats_file}")


if __name__ == "__main__":
    main()

