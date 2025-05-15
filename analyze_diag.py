import numpy as np
import matplotlib.pyplot as plt
import json

# Dữ liệu giả lập dựa trên summary_results-2.json và thông tin từ thinking trace
class_names = [
    "Sleep stage W",
    "Sleep stage N1",
    "Sleep stage N2",
    "Sleep stage N3",
    "Sleep stage R",
    "Obstructive Apnea",
    "Obstructive Hypopnea",
    "Mixed Apnea",
    "Central Apnea",
    "Oxygen Desaturation",
    "EEG arousal",
    "Hypopnea"
]

f1_means = [0.518, 0.261, 0.709, 0.620, 0.375, 0.236, 0.261, 0.195, 0.177, 0.400, 0.250, 0.210]
f1_stds = [0.0471, 0.0115, 0.0393, 0.0511, 0.1465, 0.0209, 0.0372, 0.0022, 0.0012, 0.0646, 0.0446, 0.0036]

precisions = [0.35, 0.15, 0.55, 0.45, 0.25, 0.15, 0.15, 0.08, 0.07, 0.25, 0.15, 0.08]
recalls = [1.0, 1.0, 1.0, 1.0, 0.75, 0.55, 1.0, 0.80, 0.45, 1.0, 0.75, 0.55]

supports = [1000, 200, 1500, 800, 600, 50, 100, 20, 10, 300, 150, 50]

thresholds = [0.4, 0.3, 0.5, 0.6, 0.4, 0.2, 0.3, 0.25, 0.15, 0.35, 0.3, 0.25]

weighted_f1_mean = 0.50

# Hàm vẽ F1 Score Bar Chart
def create_f1_visualization(class_names, f1_means, f1_stds, weighted_f1_mean, output_dir="./"):
    plt.figure(figsize=(12, 8))
    
    # Sắp xếp các lớp theo F1 score
    sorted_indices = np.argsort(f1_means)
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_means = [f1_means[i] for i in sorted_indices]
    sorted_stds = [f1_stds[i] for i in sorted_indices]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(class_names)))
    
    plt.barh(range(len(sorted_names)), sorted_means, xerr=sorted_stds, 
             color=colors, alpha=0.7, ecolor='black', capsize=5)
    
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('F1 Score')
    plt.title('F1 Scores by Class (Mean ± Std)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Thêm đường trung bình có trọng số
    plt.axvline(x=weighted_f1_mean, color='r', linestyle='--', 
                label=f'Weighted Avg: {weighted_f1_mean:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}f1_scores_by_class.png', dpi=300)
    plt.close()

# Hàm vẽ Precision-Recall Scatter Plot
def create_pr_curve_visualization(class_names, precisions, recalls, supports, output_dir="./"):
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        size = np.log(supports[i] + 1) * 20  # Kích thước điểm theo log của support
        plt.scatter(
            recalls[i], 
            precisions[i],
            s=size,
            alpha=0.7,
            label=f"{class_name} (n={supports[i]})"
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall for Each Class')
    plt.grid(linestyle='--', alpha=0.7)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    # Thêm đường tham chiếu
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Điều chỉnh chú thích
    plt.legend(fontsize='small', loc='lower left', bbox_to_anchor=(1, 0))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}precision_recall_scatter.png', dpi=300)
    plt.close()

# Hàm vẽ Class Balance Analysis Plot
def create_class_balance_visualization(class_names, supports, f1_scores, thresholds, output_dir="./"):
    total_samples = np.sum(supports)
    percentages = supports / total_samples * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Biểu đồ cột phân bố lớp
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(class_names)))
    ax1.bar(class_names, percentages, color=colors, alpha=0.7)
    
    # Thêm đường F1 scores
    ax1_twin = ax1.twinx()
    ax1_twin.plot(class_names, f1_scores, 'ro-', linewidth=2, markersize=8, alpha=0.7)
    
    ax1.set_ylabel('Class Distribution (%)')
    ax1_twin.set_ylabel('F1 Score', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Class Distribution and F1 Scores')
    
    # Xoay nhãn trục x nếu cần
    plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
    
    # Biểu đồ ngưỡng tối ưu
    ax2.plot(class_names, thresholds, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Default threshold')
    ax2.set_ylabel('Optimal Threshold')
    ax2.set_title('Optimized Classification Thresholds')
    
    plt.setp(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}class_balance_analysis.png', dpi=300)
    plt.close()

# Chạy các hàm để vẽ biểu đồ
if __name__ == "__main__":
    output_dir = "./results/"  # Thay đổi đường dẫn nếu cần
    create_f1_visualization(class_names, f1_means, f1_stds, weighted_f1_mean, output_dir)
    create_pr_curve_visualization(class_names, precisions, recalls, supports, output_dir)
    create_class_balance_visualization(class_names, supports, f1_means, thresholds, output_dir)
    print("Đã tạo 3 biểu đồ thành công tại thư mục:", output_dir)