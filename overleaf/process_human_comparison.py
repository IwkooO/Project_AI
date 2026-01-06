import csv
import glob
import os
import statistics

def process_human_comparison():
    """Aggregate and deduplicate human comparison data from CSV files."""
    
    # Get all CSV files from human directory
    csv_files = glob.glob('human/*.csv')
    csv_files.sort()
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Read and combine all data
    all_rows = []
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['source_file'] = csv_file  # Track source file
                all_rows.append(row)
    
    # Deduplicate human scores/distances
    # Each round has two rows (main and baseline) with the same human values
    # We need to count each human result only once per round
    # Use (source_file, round) as unique identifier since rounds repeat across games
    seen_rounds = set()
    human_distances = []
    human_scores = []
    
    main_distances = []
    main_scores = []
    baseline_distances = []
    baseline_scores = []
    
    for row in all_rows:
        round_key = (row['source_file'], row['round'])
        model = row['model']
        
        # Collect human data (only once per round)
        if round_key not in seen_rounds:
            human_distances.append(float(row['human_distance_km']))
            human_scores.append(float(row['human_score']))
            seen_rounds.add(round_key)
        
        # Collect model-specific data
        if model == 'main':
            main_distances.append(float(row['distance_km']))
            main_scores.append(float(row['score']))
        elif model == 'baseline':
            baseline_distances.append(float(row['distance_km']))
            baseline_scores.append(float(row['score']))
    
    # Calculate statistics
    def median_mean(values):
        if not values:
            return 0.0, 0.0
        return statistics.median(values), statistics.mean(values)
    
    num_rounds = len(human_distances)
    
    main_med_dist, main_mean_dist = median_mean(main_distances)
    main_med_score, main_mean_score = median_mean(main_scores)
    
    baseline_med_dist, baseline_mean_dist = median_mean(baseline_distances)
    baseline_med_score, baseline_mean_score = median_mean(baseline_scores)
    
    human_med_dist, human_mean_dist = median_mean(human_distances)
    human_med_score, human_mean_score = median_mean(human_scores)
    
    stats = {
        'num_rounds': num_rounds,
        'main': {
            'median_distance': main_med_dist,
            'mean_distance': main_mean_dist,
            'median_score': main_med_score,
            'mean_score': main_mean_score,
        },
        'baseline': {
            'median_distance': baseline_med_dist,
            'mean_distance': baseline_mean_dist,
            'median_score': baseline_med_score,
            'mean_score': baseline_mean_score,
        },
        'human': {
            'median_distance': human_med_dist,
            'mean_distance': human_mean_dist,
            'median_score': human_med_score,
            'mean_score': human_mean_score,
        }
    }
    
    # Print statistics
    print("\n" + "="*60)
    print("HUMAN COMPARISON STATISTICS")
    print("="*60)
    print(f"\nNumber of rounds evaluated: {stats['num_rounds']}")
    print("\nDistance Error (km):")
    print(f"  Main Model - Median: {stats['main']['median_distance']:.2f}, Mean: {stats['main']['mean_distance']:.2f}")
    print(f"  Baseline   - Median: {stats['baseline']['median_distance']:.2f}, Mean: {stats['baseline']['mean_distance']:.2f}")
    print(f"  Human      - Median: {stats['human']['median_distance']:.2f}, Mean: {stats['human']['mean_distance']:.2f}")
    print("\nGeoGuessr Score:")
    print(f"  Main Model - Median: {stats['main']['median_score']:.0f}, Mean: {stats['main']['mean_score']:.0f}")
    print(f"  Baseline   - Median: {stats['baseline']['median_score']:.0f}, Mean: {stats['baseline']['mean_score']:.0f}")
    print(f"  Human      - Median: {stats['human']['median_score']:.0f}, Mean: {stats['human']['mean_score']:.0f}")
    print("="*60)
    
    # Generate LaTeX table
    latex_table = f"""
\\begin{{table}}[t]
\\centering
\\caption{{GeoGuessr game comparison: main model vs baseline (GeoCLIP) vs human performance on {stats['num_rounds']} rounds.}}
\\label{{tab:human_comparison}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{Median Distance (km)}} & \\textbf{{Mean Distance (km)}} & \\textbf{{Median Score}} & \\textbf{{Mean Score}} \\\\
\\midrule
Main Model & {stats['main']['median_distance']:.1f} & {stats['main']['mean_distance']:.1f} & {stats['main']['median_score']:.0f} & {stats['main']['mean_score']:.0f} \\\\
Baseline (GeoCLIP) & {stats['baseline']['median_distance']:.1f} & {stats['baseline']['mean_distance']:.1f} & {stats['baseline']['median_score']:.0f} & {stats['baseline']['mean_score']:.0f} \\\\
Human & {stats['human']['median_distance']:.1f} & {stats['human']['mean_distance']:.1f} & {stats['human']['median_score']:.0f} & {stats['human']['mean_score']:.0f} \\\\
\\bottomrule
\\end{{tabular}}
"""
    
    print("\nLaTeX Table Code:")
    print(latex_table)
    
    return stats

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    stats = process_human_comparison()
