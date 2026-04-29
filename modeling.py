"""
=============================================================
  CRISP-DM: MODELING & EVALUATION
  Penelitian: Klasifikasi Risiko Vendor TAD (2023-2025)
  Model: Decision Tree, Random Forest, Gradient Boosting, 
         XGBoost, SVM
=============================================================
Cara menjalankan:
  1. Pastikan Data.xlsx ada di folder yang sama dengan file ini
  2. Buka terminal di VS Code
  3. Ketik: python modeling.py
  4. Tunggu sampai selesai, hasil tersimpan di folder output_modeling
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import (train_test_split, 
                                     StratifiedKFold, 
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                              GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, accuracy_score,
                             confusion_matrix, 
                             classification_report,
                             ConfusionMatrixDisplay)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI — sesuaikan nama file jika perlu
# ─────────────────────────────────────────────
DATA_FILE  = 'Data.xlsx'       # nama file Excel Anda
OUTPUT_DIR = Path('output_modeling')
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# FITUR YANG DIGUNAKAN (hasil feature selection)
# ─────────────────────────────────────────────
FEATURES = [
    'Proportion Delay',
    'Proportion Gap',
    'Mean Delay (X01)',
    'Mean Delay (X02)',
    'Mean Delay (X04)',
    'Mean Gap (X11)',
    'Mean Gap (X14)',
    'Lag Proportion Delay',
    'Lag Proportion Gap',
    'Lag Mean Delay (X02)',
    'Lag Mean Delay (X04)',   # sudah diperbaiki typo-nya
    'Lag  Mean Gap (X11)',
    'Lag Mean Gap (X14)',
]

LABEL_MAP = {'Patuh': 0, 'Waspada': 1, 'Kritis': 2}
LABEL_INV = {0: 'Patuh', 1: 'Waspada', 2: 'Kritis'}
CLASS_NAMES = ['Patuh', 'Waspada', 'Kritis']


# ═══════════════════════════════════════════════
# STEP 1 — LOAD & DATA PREPARATION
# ═══════════════════════════════════════════════
print("=" * 60)
print("  STEP 1: LOAD DATA & PERSIAPAN")
print("=" * 60)

df = pd.read_excel(DATA_FILE)

# Rename typo jika masih ada di file Excel
if 'MLag ean Delay (X04)' in df.columns:
    df = df.rename(columns={'MLag ean Delay (X04)': 'Lag Mean Delay (X04)'})
    print("  ✔ Typo kolom diperbaiki otomatis")

X = df[FEATURES].values
y = df['Klasifikasi Risiko'].map(LABEL_MAP).values

# Train-test split 80:20 stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Standardisasi
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Data training : {len(X_train)} baris")
print(f"  Data test     : {len(X_test)} baris")
print()
print("  Distribusi kelas:")
for label, code in LABEL_MAP.items():
    tr = (y_train == code).sum()
    te = (y_test  == code).sum()
    print(f"    {label:10s}: train={tr}, test={te}")


# ═══════════════════════════════════════════════
# STEP 2 — DEFINISI MODEL & HYPERPARAMETER
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  STEP 2: DEFINISI MODEL")
print("=" * 60)

cv_strategy = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE
)

# Definisi model dan grid hyperparameter
models_config = {

    'Decision Tree': {
        'model_fn': lambda **kw: DecisionTreeClassifier(
            **kw,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'max_depth'         : [3, 5, 7, None],
            'min_samples_split' : [2, 5, 10],
            'criterion'         : ['gini', 'entropy'],
        }
    },

    'Random Forest': {
        'model_fn': lambda **kw: RandomForestClassifier(
            **kw,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'n_estimators'      : [50, 100, 200],
            'max_depth'         : [3, 5, 7, None],
            'min_samples_split' : [2, 5, 10],
        }
    },

    'Gradient Boosting': {
        'model_fn': lambda **kw: GradientBoostingClassifier(
            **kw,
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'n_estimators'  : [50, 100, 200],
            'max_depth'     : [3, 5],
            'learning_rate' : [0.05, 0.1, 0.2],
        }
    },

    'XGBoost': {
        'model_fn': lambda **kw: XGBClassifier(
            **kw,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=RANDOM_STATE,
            verbosity=0
        ),
        'param_grid': {
            'n_estimators'  : [50, 100, 200],
            'max_depth'     : [3, 5, 7],
            'learning_rate' : [0.05, 0.1, 0.2],
            'subsample'     : [0.8, 1.0],
        }
    },

    'SVM': {
        'model_fn': lambda **kw: SVC(
            **kw,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'C'      : [0.1, 1, 10],
            'kernel' : ['rbf', 'linear'],
            'gamma'  : ['scale', 'auto'],
        }
    },
}

print(f"  Model yang diuji: {', '.join(models_config.keys())}")
print(f"  Cross-validation: 5-Fold Stratified")
print(f"  Metrik utama    : Macro F1-Score")


# ═══════════════════════════════════════════════
# STEP 3 — TRAINING & TUNING
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  STEP 3: TRAINING & HYPERPARAMETER TUNING")
print("  (Mohon tunggu, proses ini memakan waktu beberapa menit)")
print("=" * 60)

results = {}

for name, cfg in models_config.items():
    print(f"\n  [{name}] mencari parameter terbaik...")

    best_cv_f1 = 0
    best_params = {}
    best_model  = None

    # Manual grid search
    keys = list(cfg['param_grid'].keys())
    vals = list(cfg['param_grid'].values())

    for combo in itertools.product(*vals):
        params = dict(zip(keys, combo))
        model  = cfg['model_fn'](**params)
        cv_scores = cross_val_score(
            model, X_train_sc, y_train,
            cv=cv_strategy, scoring='f1_macro'
        )
        if cv_scores.mean() > best_cv_f1:
            best_cv_f1 = cv_scores.mean()
            best_params = params

    # Retrain dengan parameter terbaik
    best_model = cfg['model_fn'](**best_params)
    best_model.fit(X_train_sc, y_train)

    # Evaluasi pada test set
    y_pred   = best_model.predict(X_test_sc)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_per   = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2])
    acc      = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    gap      = abs(best_cv_f1 - f1_macro)

    results[name] = {
        'model'     : best_model,
        'params'    : best_params,
        'cv_f1'     : best_cv_f1,
        'test_f1'   : f1_macro,
        'f1_per'    : f1_per,
        'accuracy'  : acc,
        'cm'        : cm,
        'gap'       : gap,
        'y_pred'    : y_pred,
    }

    print(f"    CV F1    = {best_cv_f1:.4f}")
    print(f"    Test F1  = {f1_macro:.4f} "
          f"(Gap={gap:.4f}{'  ⚠ besar' if gap > 0.10 else '  ✓ stabil'})")
    print(f"    F1 Patuh={f1_per[0]:.3f} | "
          f"Waspada={f1_per[1]:.3f} | "
          f"Kritis={f1_per[2]:.3f}")
    print(f"    Best params: {best_params}")


# ═══════════════════════════════════════════════
# STEP 4 — PERBANDINGAN & PEMILIHAN MODEL TERBAIK
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  STEP 4: PERBANDINGAN MODEL")
print("=" * 60)
print()
print(f"  {'Model':<20} {'CV F1':>7} {'Test F1':>8} {'Gap':>7} "
      f"{'Patuh':>7} {'Waspada':>8} {'Kritis':>7}")
print("  " + "-" * 68)

for name, r in results.items():
    flag = '  ⚠ overfitting' if r['gap'] > 0.10 else ''
    print(f"  {name:<20} {r['cv_f1']:>7.4f} {r['test_f1']:>8.4f} "
          f"{r['gap']:>7.4f} {r['f1_per'][0]:>7.3f} "
          f"{r['f1_per'][1]:>8.3f} {r['f1_per'][2]:>7.3f}{flag}")

# Pilih model terbaik berdasarkan CV F1 (lebih reliable)
# Kecualikan model dengan gap > 0.10 (indikasi overfitting evaluasi)
stable_models = {n: r for n, r in results.items() if r['gap'] <= 0.10}
if stable_models:
    best_name = max(stable_models, key=lambda n: stable_models[n]['cv_f1'])
else:
    best_name = max(results, key=lambda n: results[n]['cv_f1'])

best = results[best_name]
print()
print(f"  ★ MODEL TERBAIK : {best_name}")
print(f"    Alasan        : CV F1 tertinggi ({best['cv_f1']:.4f}) "
      f"dengan gap stabil ({best['gap']:.4f})")
print(f"    Best params   : {best['params']}")
print()
print(f"  Classification Report — {best_name}:")
print(classification_report(y_test, best['y_pred'],
                             target_names=CLASS_NAMES, digits=3))
print("  Confusion Matrix:")
print("            Prediksi →")
print(f"  {'':12s} {'Patuh':>8} {'Waspada':>9} {'Kritis':>8}")
for i, label in enumerate(CLASS_NAMES):
    row = best['cm'][i]
    print(f"  {label + ' (aktual)':20s} {row[0]:>6} {row[1]:>9} {row[2]:>8}")


# ═══════════════════════════════════════════════
# STEP 5 — VISUALISASI
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  STEP 5: MEMBUAT VISUALISASI")
print("=" * 60)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.dpi' : 150,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

COLORS = {
    'Decision Tree'    : '#3498db',
    'Random Forest'    : '#2ecc71',
    'Gradient Boosting': '#e67e22',
    'XGBoost'          : '#e74c3c',
    'SVM'              : '#9b59b6',
}

# ── Grafik 1: Perbandingan CV F1 vs Test F1 ──────────────────
model_names = list(results.keys())
cv_scores   = [results[n]['cv_f1']   for n in model_names]
test_scores = [results[n]['test_f1'] for n in model_names]
colors_bar  = [COLORS[n] for n in model_names]

x = np.arange(len(model_names))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Perbandingan Kinerja Model — Klasifikasi Risiko Vendor',
             fontweight='bold', fontsize=13)

ax = axes[0]
bars1 = ax.bar(x - w/2, cv_scores,   w,
               label='CV Macro F1 (5-Fold)',
               color=colors_bar, alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + w/2, test_scores, w,
               label='Test Macro F1',
               color=colors_bar, alpha=0.50, edgecolor='white',
               linestyle='--', linewidth=1.2)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
            f'{h:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_ylim(0, 1.15)
ax.set_ylabel('Macro F1-Score')
ax.set_title('CV F1 vs Test F1 per Model')
ax.legend(fontsize=9)
ax.axhline(0.80, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
ax.grid(axis='y', alpha=0.3)
# Tandai model terbaik
ax.get_xticklabels()[model_names.index(best_name)].set_color('#2ecc71')
ax.get_xticklabels()[model_names.index(best_name)].set_fontweight('bold')

# ── Grafik 2: F1 per kelas ────────────────────────────────────
ax2 = axes[1]
kelas = ['Patuh', 'Waspada', 'Kritis']
n_models = len(model_names)
width = 0.15
offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)
x2 = np.arange(len(kelas))

for i, name in enumerate(model_names):
    vals = results[name]['f1_per']
    bars = ax2.bar(x2 + offsets[i], vals, width,
                   label=name, color=COLORS[name],
                   alpha=0.85, edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f'{h:.2f}', ha='center', va='bottom', fontsize=7)

ax2.set_xticks(x2)
ax2.set_xticklabels(kelas)
ax2.set_ylim(0, 1.18)
ax2.set_ylabel('F1-Score')
ax2.set_title('F1-Score per Kelas per Model')
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
path1 = OUTPUT_DIR / '01_perbandingan_model.png'
plt.savefig(path1, bbox_inches='tight')
plt.close()
print(f"  ✔ {path1}")

# ── Grafik 2: Confusion Matrix semua model ────────────────────
n = len(results)
fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
fig.suptitle('Confusion Matrix — Semua Model (Test Set)',
             fontweight='bold')

for ax, (name, r) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=r['cm'],
        display_labels=CLASS_NAMES
    )
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    title = f"{name}\nCV F1={r['cv_f1']:.3f} | Test={r['test_f1']:.3f}"
    if name == best_name:
        title += '\n★ TERBAIK'
    ax.set_title(title, fontsize=9)

plt.tight_layout()
path2 = OUTPUT_DIR / '02_confusion_matrix.png'
plt.savefig(path2, bbox_inches='tight')
plt.close()
print(f"  ✔ {path2}")

# ── Grafik 3: Feature Importance model terbaik ───────────────
if hasattr(best['model'], 'feature_importances_'):
    fi = pd.DataFrame({
        'Fitur'     : FEATURES,
        'Importance': best['model'].feature_importances_
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_fi = ['#e74c3c' if i >= len(fi)-3 else '#3498db'
                 for i in range(len(fi))]
    ax.barh(fi['Fitur'], fi['Importance'],
            color=colors_fi, edgecolor='white')
    ax.set_title(f'Feature Importance — {best_name} (Model Terbaik)',
                 fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.3)
    for i, (val, feat) in enumerate(zip(fi['Importance'], fi['Fitur'])):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    path3 = OUTPUT_DIR / '03_feature_importance.png'
    plt.savefig(path3, bbox_inches='tight')
    plt.close()
    print(f"  ✔ {path3}")


# ═══════════════════════════════════════════════
# STEP 6 — SIMPAN MODEL TERBAIK
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  STEP 6: SIMPAN MODEL TERBAIK")
print("=" * 60)

model_artifact = {
    'model'        : best['model'],
    'model_name'   : best_name,
    'scaler'       : scaler,
    'label_map'    : LABEL_MAP,
    'label_inv'    : LABEL_INV,
    'feature_names': FEATURES,
    'best_params'  : best['params'],
    'cv_f1'        : best['cv_f1'],
    'test_f1'      : best['test_f1'],
    'all_results'  : {
        n: {
            'cv_f1'  : v['cv_f1'],
            'test_f1': v['test_f1'],
            'f1_per' : v['f1_per'],
            'gap'    : v['gap'],
            'params' : v['params'],
        }
        for n, v in results.items()
    }
}

model_path = OUTPUT_DIR / 'best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_artifact, f)

# Simpan tabel perbandingan ke CSV
comparison = pd.DataFrame({
    name: {
        'CV Macro F1'  : f"{r['cv_f1']:.4f}",
        'Test Macro F1': f"{r['test_f1']:.4f}",
        'Gap'          : f"{r['gap']:.4f}",
        'F1 Patuh'     : f"{r['f1_per'][0]:.3f}",
        'F1 Waspada'   : f"{r['f1_per'][1]:.3f}",
        'F1 Kritis'    : f"{r['f1_per'][2]:.3f}",
        'Best Params'  : str(r['params']),
    }
    for name, r in results.items()
}).T
comparison.to_csv(OUTPUT_DIR / 'perbandingan_model.csv')

print(f"  ✔ Model tersimpan  : {model_path}")
print(f"  ✔ Tabel perbandingan: {OUTPUT_DIR}/perbandingan_model.csv")


# ═══════════════════════════════════════════════
# RINGKASAN AKHIR
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SELESAI! RINGKASAN AKHIR")
print("=" * 60)
print(f"  Model terbaik   : {best_name}")
print(f"  CV Macro F1     : {best['cv_f1']:.4f}")
print(f"  Test Macro F1   : {best['test_f1']:.4f}")
print(f"  Accuracy        : {best['accuracy']:.4f}")
print(f"  F1 per kelas    : Patuh={best['f1_per'][0]:.3f} | "
      f"Waspada={best['f1_per'][1]:.3f} | "
      f"Kritis={best['f1_per'][2]:.3f}")
print()
print(f"  Output tersimpan di folder: ./{OUTPUT_DIR}/")
print(f"    - best_model.pkl")
print(f"    - 01_perbandingan_model.png")
print(f"    - 02_confusion_matrix.png")
print(f"    - 03_feature_importance.png")
print(f"    - perbandingan_model.csv")
print()
print("  Model siap digunakan untuk prediksi data baru!")
print("=" * 60)