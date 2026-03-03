"""
Analyze difference between generated and real IR images.
Creates visualization with difference map and distribution plots.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm


def analyze_single_image(rgb_path, gen_path, real_path, output_dir=None):
    """Analyze single image pair and return metrics + visualization."""
    
    # Load images
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    gen = cv2.imread(str(gen_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    real = cv2.imread(str(real_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    
    # Normalize al range [-1, 1] come nel modello
    gen = gen * 2 - 1
    real = real * 2 - 1
    
    # Calculate difference (original - predicted)
    diff = real - gen
    
    # Extract error categories
    diff_max = 0.02
    diff_min = -0.02
    
    diff_inf = diff[diff < diff_min]      # Undershooting (blu)
    diff_sup = diff[diff > diff_max]      # Overshooting (rosso)
    near_zero = diff[(diff <= diff_max) & (diff >= diff_min)]  # Near zero (grigio)
    
    # Metrics
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    max_error = np.max(np.abs(diff))
    psnr = 20 * np.log10(2.0 / np.sqrt(np.mean(diff ** 2)))
    ssim_val = np.mean((2 * gen * real + 1e-4) / (gen**2 + real**2 + 1e-4))
    
    # Count error types
    n_under = len(diff_inf)
    n_over = len(diff_sup)
    n_near = len(near_zero)
    total_pixels = diff.size
    
    pct_under = 100 * n_under / total_pixels
    pct_over = 100 * n_over / total_pixels
    pct_near = 100 * n_near / total_pixels
    
    # Create visualization
    fig = plt.figure(figsize=(18, 5))
    
    # 1. Difference map (seismic colormap)
    ax1 = plt.subplot(1, 3, 1)
    im = ax1.imshow(diff, cmap='seismic', vmin=-1, vmax=1)
    ax1.set_title("IR Difference Map\n(Original - Predicted)", fontsize=12, fontweight='bold')
    ax1.set_xlabel(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    plt.colorbar(im, ax=ax1, label='Difference')
    
    # 2. Error distribution histograms
    ax2 = plt.subplot(1, 3, 2)
    bins = np.linspace(-1, 1, 256)
    
    ax2.hist(diff_inf, bins=bins, color='#0000b8', alpha=0.7, label=f'Under (−) {pct_under:.1f}%')
    ax2.hist(diff_sup, bins=bins, color='#8b0000', alpha=0.7, label=f'Over (+) {pct_over:.1f}%')
    ax2.hist(near_zero, bins=bins, color='#cccccc', alpha=0.7, label=f'Near-zero {pct_near:.1f}%')
    
    ax2.axvline(-diff_min, color='darkblue', linestyle='--', linewidth=2, alpha=0.5)
    ax2.axvline(diff_min, color='darkred', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Difference Value')
    ax2.set_ylabel('Frequency (pixels)')
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error statistics
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    stats_text = f"""
    IMAGE ANALYSIS REPORT
    {'='*40}
    
    ERROR METRICS:
    • MAE (Mean Absolute Error): {mae:.6f}
    • RMSE (Root Mean Squared): {rmse:.6f}
    • Max Error: {max_error:.6f}
    • PSNR: {psnr:.2f} dB
    • SSIM (est): {ssim_val:.4f}
    
    ERROR DISTRIBUTION:
    • Undershooting (blue):  {n_under:,} pixels ({pct_under:5.2f}%)
    • Overshooting (red):    {n_over:,}  pixels ({pct_over:5.2f}%)
    • Near-zero (gray):      {n_near:,} pixels ({pct_near:5.2f}%)
    
    THRESHOLD: ±{diff_max}
    Total pixels analyzed: {total_pixels:,}
    """
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{Path(rgb_path).stem}_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved analysis to {output_path}")
    
    plt.close()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'psnr': psnr,
        'ssim': ssim_val,
        'pct_under': pct_under,
        'pct_over': pct_over,
        'pct_near': pct_near,
        'diff_map': diff
    }


def batch_analysis(rgb_dir, gen_dir, real_dir, output_dir=None):
    """Analyze all image pairs and generate report."""
    
    rgb_dir = Path(rgb_dir)
    gen_dir = Path(gen_dir)
    real_dir = Path(real_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_paths = sorted(rgb_dir.glob("*.jpg"))
    
    all_metrics = []
    all_diffs = []
    
    print(f"\n📊 Analyzing {len(rgb_paths)} images...\n")
    
    for rgb_path in tqdm(rgb_paths):
        gen_path = gen_dir / rgb_path.name
        real_path = real_dir / rgb_path.name
        
        if not gen_path.exists() or not real_path.exists():
            print(f"⚠️  Skipping {rgb_path.name} (missing generated or real)")
            continue
        
        metrics = analyze_single_image(rgb_path, gen_path, real_path, output_dir)
        all_metrics.append(metrics)
        all_diffs.append(metrics['diff_map'])
    
    # Aggregate statistics
    mae_values = [m['mae'] for m in all_metrics]
    rmse_values = [m['rmse'] for m in all_metrics]
    psnr_values = [m['psnr'] for m in all_metrics]
    ssim_values = [m['ssim'] for m in all_metrics]
    
    # Summary report
    print("\n" + "="*60)
    print("BATCH ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nImages analyzed: {len(all_metrics)}")
    print(f"\nMAE:  {np.mean(mae_values):.6f} ± {np.std(mae_values):.6f}")
    print(f"RMSE: {np.mean(rmse_values):.6f} ± {np.std(rmse_values):.6f}")
    print(f"PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"SSIM: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
    print("\nError type distribution (averaged):")
    print(f"  Undershooting (blue):  {np.mean([m['pct_under'] for m in all_metrics]):.2f}%")
    print(f"  Overshooting (red):    {np.mean([m['pct_over'] for m in all_metrics]):.2f}%")
    print(f"  Near-zero (gray):      {np.mean([m['pct_near'] for m in all_metrics]):.2f}%")
    print("="*60 + "\n")
    
    # Find best and worst
    best_idx = np.argmin(mae_values)
    worst_idx = np.argmax(mae_values)
    
    print(f"🏆 Best:  {rgb_paths[best_idx].name} (MAE: {mae_values[best_idx]:.6f})")
    print(f"💥 Worst: {rgb_paths[worst_idx].name} (MAE: {mae_values[worst_idx]:.6f})")
    
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze difference between generated and real IR images"
    )
    parser.add_argument("--rgb-dir", required=True, help="Directory with RGB images")
    parser.add_argument("--gen-dir", required=True, help="Directory with generated IR")
    parser.add_argument("--real-dir", required=True, help="Directory with real IR")
    parser.add_argument("--output-dir", default=None, help="Output directory for analysis plots")
    
    args = parser.parse_args()
    
    batch_analysis(args.rgb_dir, args.gen_dir, args.real_dir, args.output_dir)