import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix

# Your multi-label results
multi_label_results = {
    "fold": 0,
    "class_metrics": {
        "Sleep stage W": {
            "precision": 0.30, 
            "recall": 0.92,     
            "f1": 0.54,       
            "support": 800,    
            "roc_auc": 0.65,   
            "pr_auc": 0.35,     
            "threshold": 0.15   
        },
        "Sleep stage N1": {
            "precision": 0.10,  
            "recall": 0.85,    
            "f1": 0.22,       
            "support": 200,     
            "roc_auc": 0.62,  
            "pr_auc": 0.15,     
            "threshold": 0.20   
        },
        "Sleep stage N2": {
            "precision": 0.45, 
            "recall": 0.95,    
            "f1": 0.71,        
            "support": 1300,  
            "roc_auc": 0.60,    
            "pr_auc": 0.50,     
            "threshold": 0.30   
        },
        "Sleep stage N3": {
            "precision": 0.35, 
            "recall": 0.93,    
            "f1": 0.56,       
            "support": 1050,    
            "roc_auc": 0.72,   
            "pr_auc": 0.55,     
            "threshold": 0.15   
        },
        "Sleep stage R": {
            "precision": 0.32, 
            "recall": 0.94,    
            "f1": 0.48,         
            "support": 900,     
            "roc_auc": 0.63,   
            "pr_auc": 0.40,     
            "threshold": 0.15   
        },
        "Obstructive Apnea": {
            "precision": 0.25, 
            "recall": 0.75,     
            "f1": 0.24,        
            "support": 30,      
            "roc_auc": 0.68,   
            "pr_auc": 0.30,    
            "threshold": 0.35   
        },
        "Obstructive Hypopnea": {
            "precision": 0.28,  
            "recall": 0.80,     
            "f1": 0.36,         
            "support": 100,     
            "roc_auc": 0.70,    
            "pr_auc": 0.35,     
            "threshold": 0.25   
        },
        "Mixed Apnea": {
            "precision": 0.22,  
            "recall": 0.70,   
            "f1": 0.30,        
            "support": 25,    
            "roc_auc": 0.65,   
            "pr_auc": 0.28,     
            "threshold": 0.35   
        },
        "Central Apnea": {
            "precision": 0.20,  
            "recall": 0.65,    
            "f1": 0.28,       
            "support": 20,   
            "roc_auc": 0.67,    
            "pr_auc": 0.25,     
            "threshold": 0.35  
        },
        "Oxygen Desaturation": {
            "precision": 0.30, 
            "recall": 0.88,     
            "f1": 0.45,        
            "support": 700,   
            "roc_auc": 0.62,  
            "pr_auc": 0.40,     
            "threshold": 0.20   
        },
        "EEG arousal": {
            "precision": 0.25, 
            "recall": 0.82,     
            "f1": 0.26,         
            "support": 250,   
            "roc_auc": 0.60,   
            "pr_auc": 0.30,     
            "threshold": 0.25  
        },
        "Hypopnea": {
            "precision": 0.18,  
            "recall": 0.60,   
            "f1": 0.20,        
            "support": 20,      
            "roc_auc": 0.62,   
            "pr_auc": 0.22,    
            "threshold": 0.35 
        }
    },
    "weighted_metrics": {
        "precision": 0.34, 
        "recall": 0.90,    
        "f1": 0.49,        
        "roc_auc": 0.64,    
        "pr_auc": 0.42      
    }
}

# Simulated single-label results
single_label_results = {
  "fold": 0,
  "class_metrics": {
    "Sleep stage W": {
      "precision": 0.15,
      "recall": 0.90,
      "f1": 0.26,
      "support": 735,
      "roc_auc": 0.52,
      "pr_auc": 0.18,
      "threshold": 0.5
    },
    "Sleep stage N1": {
      "precision": 0.03,
      "recall": 0.85,
      "f1": 0.06,
      "support": 158,
      "roc_auc": 0.50,
      "pr_auc": 0.04,
      "threshold": 0.5
    },
    "Sleep stage N2": {
      "precision": 0.28,
      "recall": 0.95,
      "f1": 0.43,
      "support": 1289,
      "roc_auc": 0.51,
      "pr_auc": 0.30,
      "threshold": 0.5
    },
    "Sleep stage N3": {
      "precision": 0.20,
      "recall": 0.90,
      "f1": 0.33,
      "support": 1013,
      "roc_auc": 0.60,
      "pr_auc": 0.35,
      "threshold": 0.5
    },
    "Sleep stage R": {
      "precision": 0.18,
      "recall": 0.93,
      "f1": 0.30,
      "support": 882,
      "roc_auc": 0.50,
      "pr_auc": 0.20,
      "threshold": 0.5
    },
    "Obstructive Apnea": {
      "precision": 0.05,
      "recall": 0.30,
      "f1": 0.09,
      "support": 9,
      "roc_auc": 0.50,
      "pr_auc": 0.05,
      "threshold": 0.5
    },
    "Obstructive Hypopnea": {
      "precision": 0.08,
      "recall": 0.40,
      "f1": 0.13,
      "support": 57,
      "roc_auc": 0.51,
      "pr_auc": 0.07,
      "threshold": 0.5
    },
    "Mixed Apnea": {
      "precision": 0.04,
      "recall": 0.25,
      "f1": 0.07,
      "support": 8,
      "roc_auc": 0.50,
      "pr_auc": 0.04,
      "threshold": 0.5
    },
    "Central Apnea": {
      "precision": 0.03,
      "recall": 0.20,
      "f1": 0.05,
      "support": 7,
      "roc_auc": 0.50,
      "pr_auc": 0.03,
      "threshold": 0.5
    },
    "Oxygen Desaturation": {
      "precision": 0.10,
      "recall": 0.50,
      "f1": 0.17,
      "support": 648,
      "roc_auc": 0.51,
      "pr_auc": 0.12,
      "threshold": 0.5
    },
    "EEG arousal": {
      "precision": 0.07,
      "recall": 0.35,
      "f1": 0.12,
      "support": 228,
      "roc_auc": 0.50,
      "pr_auc": 0.08,
      "threshold": 0.5
    },
    "Hypopnea": {
      "precision": 0.05,
      "recall": 0.20,
      "f1": 0.08,
      "support": 5,
      "roc_auc": 0.50,
      "pr_auc": 0.04,
      "threshold": 0.5
    }
  },
  "weighted_metrics": {
    "precision": 0.19,
    "recall": 0.85,
    "f1": 0.31,
    "roc_auc": 0.52,
    "pr_auc": 0.23
  }
}

# Class names
class_names = [
    "Sleep stage W", "Sleep stage N1", "Sleep stage N2", "Sleep stage N3",
    "Sleep stage R", "Obstructive Apnea", "Obstructive Hypopnea", "Mixed Apnea",
    "Central Apnea", "Oxygen Desaturation", "EEG arousal", "Hypopnea"
]

# Extract metrics
single_f1 = [single_label_results["class_metrics"][cls]["f1"] for cls in class_names]
multi_f1 = [multi_label_results["class_metrics"][cls]["f1"] for cls in class_names]
single_precision = [single_label_results["class_metrics"][cls]["precision"] for cls in class_names]
multi_precision = [multi_label_results["class_metrics"][cls]["precision"] for cls in class_names]
single_recall = [single_label_results["class_metrics"][cls]["recall"] for cls in class_names]
multi_recall = [multi_label_results["class_metrics"][cls]["recall"] for cls in class_names]
single_roc_auc = [single_label_results["class_metrics"][cls]["roc_auc"] for cls in class_names]
multi_roc_auc = [multi_label_results["class_metrics"][cls]["roc_auc"] for cls in class_names]
supports = [multi_label_results["class_metrics"][cls]["support"] for cls in class_names]

# Print comparison
print("Comparison of Single-Label vs. Multi-Label:")
for cls, s_f1, m_f1, s_prec, m_prec, s_rec, m_rec, s_roc, m_roc, sup in zip(
    class_names, single_f1, multi_f1, single_precision, multi_precision, single_recall, multi_recall, single_roc_auc, multi_roc_auc, supports
):
    print(f"{cls} (Support: {sup}):")
    print(f"  Single-Label: F1={s_f1:.4f}, Precision={s_prec:.4f}, Recall={s_rec:.4f}, ROC-AUC={s_roc:.4f}")
    print(f"  Multi-Label:  F1={m_f1:.4f}, Precision={m_prec:.4f}, Recall={m_rec:.4f}, ROC-AUC={m_roc:.4f}")
print("\nWeighted Metrics:")
print(f"Single-Label: F1={single_label_results['weighted_metrics']['f1']:.4f}, Precision={single_label_results['weighted_metrics']['precision']:.4f}, Recall={single_label_results['weighted_metrics']['recall']:.4f}, ROC-AUC={single_label_results['weighted_metrics']['roc_auc']:.4f}")
print(f"Multi-Label:  F1={multi_label_results['weighted_metrics']['f1']:.4f}, Precision={multi_label_results['weighted_metrics']['precision']:.4f}, Recall={multi_label_results['weighted_metrics']['recall']:.4f}, ROC-AUC={multi_label_results['weighted_metrics']['roc_auc']:.4f}")

# Visualizations
# 1. Bar Plot: Per-class F1-scores
plt.figure(figsize=(14, 6))
x = np.arange(len(class_names))
width = 0.35
plt.bar(x - width/2, single_f1, width, label='Single-Label', color='skyblue')
plt.bar(x + width/2, multi_f1, width, label='Multi-Label', color='salmon')
plt.xticks(x, class_names, rotation=45, ha='right')
plt.ylabel('F1-Score')
plt.title('Per-Class F1-Scores: Single-Label vs. Multi-Label')
plt.legend()
plt.tight_layout()
plt.savefig('f1_scores_comparison.png')
plt.close()

# 2. Bar Plot: Weighted Metrics
weighted_metrics = ['F1', 'Precision', 'Recall', 'ROC-AUC']
single_weighted = [
    single_label_results['weighted_metrics']['f1'],
    single_label_results['weighted_metrics']['precision'],
    single_label_results['weighted_metrics']['recall'],
    single_label_results['weighted_metrics']['roc_auc']
]
multi_weighted = [
    multi_label_results['weighted_metrics']['f1'],
    multi_label_results['weighted_metrics']['precision'],
    multi_label_results['weighted_metrics']['recall'],
    multi_label_results['weighted_metrics']['roc_auc']
]
plt.figure(figsize=(10, 6))
x = np.arange(len(weighted_metrics))
plt.bar(x - width/2, single_weighted, width, label='Single-Label', color='skyblue')
plt.bar(x + width/2, multi_weighted, width, label='Multi-Label', color='salmon')
plt.xticks(x, weighted_metrics)
plt.ylabel('Score')
plt.title('Weighted Metrics Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('weighted_metrics_comparison.png')
plt.close()

# 3. Event Distribution (based on support and predictions)
# Approximate event counts based on support and recall
# 3. Event Distribution (based on support and predictions)
# Approximate event counts based on support and recall
single_counts = [single_label_results["class_metrics"][cls]["support"] * single_label_results["class_metrics"][cls]["recall"] for cls in class_names]
multi_counts = [multi_label_results["class_metrics"][cls]["support"] * multi_label_results["class_metrics"][cls]["recall"] for cls in class_names]
plt.figure(figsize=(14, 6))
x = np.arange(len(class_names))  # Define x based on the number of classes
width = 0.35
plt.bar(x - width/2, single_counts, width, label='Single-Label', color='skyblue')
plt.bar(x + width/2, multi_counts, width, label='Multi-Label', color='salmon')
plt.xticks(x, class_names, rotation=45, ha='right')
plt.ylabel('Predicted Event Count')
plt.title('Predicted Event Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('event_distribution.png')
plt.close()

# Note: Precision-Recall curves and confusion matrices require raw predictions, which are not available.
# If you have raw predictions, you can add these visualizations as shown in the previous response.