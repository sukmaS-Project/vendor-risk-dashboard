"""
=============================================================
  7 EKSPERIMEN SISTEMATIS
  Penelitian: Klasifikasi Risiko Vendor TAD (2023-2025)
=============================================================
Eksperimen:
  1  DT         | Tanpa lag | Tanpa class weight (baseline)
  2  RF         | Tanpa lag | Tanpa class weight
  3  RF         | Dengan lag| Tanpa class weight
  4  RF         | Dengan lag| Dengan class weight  <- terbaik?
  5  XGBoost    | Tanpa lag | Tanpa class weight
  6  XGBoost    | Dengan lag| Tanpa class weight
  7  XGBoost    | Dengan lag| Dengan class weight
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import itertools
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATA_FILE    = 'Data.xlsx'
OUTPUT_DIR   = Path('output_7eksperimen')
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# DEFINISI FITUR
# ─────────────────────────────────────────────

# Fitur tanpa lag (7 fitur)
FEATURES_NOLAG = [
    'Proportion Delay',
    'Proportion Gap',
    'Mean Delay (X01)',
    'Mean Delay (X02)',
    'Mean Delay (X04)',
    'Mean Gap (X11)',
    'Mean Gap (X14)',
]

# Fitur lag (6 fitur)
FEATURES_LAG = [
    'Lag Proportion Delay',
    'Lag Proportion Gap',
    'Lag Mean Delay (X02)',
    'Lag Mean Delay (X04)',
    'Lag  Mean Gap (X11)',
    'Lag Mean Gap (X14)',
]

# Fitur dengan lag (13 fitur)
FEATURES_WITHLAG = FEATURES_NOLAG + FEATURES_LAG

LABEL_MAP  = {'Patuh': 0, 'Waspada': 1, 'Kritis': 2}
LABEL_INV  = {0: 'Patuh', 1: 'Waspada', 2: 'Kritis'}
CLASS_NAMES = ['Patuh', 'Waspada', 'Kritis']

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("=" * 65)
print("  7 EKSPERIMEN KLASIFIKASI RISIKO VENDOR TAD")
print("=" * 65)

df = pd.read_excel(DATA_FILE)
if 'MLag ean Delay (X04)' in df.columns:
    df = df.rename(columns={'MLag ean Delay (X04)': 'Lag Mean Delay (X04)'})
    print("  ✔ Typo kolom diperbaiki otomatis")

y = df['Klasifikasi Risiko'].map(LABEL_MAP).values

print(f"\n  Dataset: {len(df)} baris | {y.shape[0]} observasi")
print("  Distribusi kelas:")
for label, code in LABEL_MAP.items():
    n = (y == code).sum()
    print(f"    {label:10s}: {n} ({n/len(y)*100:.1f}%)")

# ─────────────────────────────────────────────
# FUNGSI HELPER
# ─────────────────────────────────────────────
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def prepare_data(features):
    """Siapkan X_train, X_test dengan split 80:20 stratified."""
    X = df[features].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20,
        random_state=RANDOM_STATE, stratify=y
    )
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_tr)
    Xte_sc = sc.transform(X_te)
    return Xtr_sc, Xte_sc, y_tr, y_te, sc

def manual_grid_search(model_fn, param_grid, Xtr, y_tr):
    """Grid search manual, return best params dan CV F1."""
    best_cvf, best_p = 0, {}
    keys = list(param_grid.keys())
    for combo in itertools.product(*param_grid.values()):
        p = dict(zip(keys, combo))
        m = model_fn(**p)
        s = cross_val_score(m, Xtr, y_tr,
                            cv=cv5, scoring='f1_macro').mean()
        if s > best_cvf:
            best_cvf, best_p = s, p
    return best_cvf, best_p

def evaluate(model, Xte, y_te):
    """Evaluasi model pada test set."""
    yp    = model.predict(Xte)
    f1m   = f1_score(y_te, yp, average='macro')
    fpc   = f1_score(y_te, yp, average=None, labels=[0,1,2])
    acc   = accuracy_score(y_te, yp)
    cm    = confusion_matrix(y_te, yp, labels=[0,1,2])
    return yp, f1m, fpc, acc, cm

# ─────────────────────────────────────────────
# DEFINISI 7 EKSPERIMEN
# ─────────────────────────────────────────────
eksperimen_config = [
    {
        'no'      : 1,
        'nama'    : 'Decision Tree',
        'fitur'   : FEATURES_NOLAG,
        'cw'      : False,
        'deskripsi': 'Baseline — tanpa lag, tanpa class weight',
        'model_fn': lambda cw, **kw: DecisionTreeClassifier(
            **kw, class_weight='balanced' if cw else None,
            random_state=RANDOM_STATE),
        'grid'    : {
            'max_depth'         : [3, 5, 7, None],
            'min_samples_split' : [2, 5, 10],
            'criterion'         : ['gini', 'entropy'],
        }
    },
    {
        'no'      : 2,
        'nama'    : 'Random Forest',
        'fitur'   : FEATURES_NOLAG,
        'cw'      : False,
        'deskripsi': 'RF murni — tanpa lag, tanpa class weight',
        'model_fn': lambda cw, **kw: RandomForestClassifier(
            **kw, class_weight='balanced' if cw else None,
            random_state=RANDOM_STATE),
        'grid'    : {
            'n_estimators'      : [50, 100, 200],
            'max_depth'         : [3, 5, 7, None],
            'min_samples_split' : [2, 5, 10],
        }
    },
    {
        'no'      : 3,
        'nama'    : 'Random Forest',
        'fitur'   : FEATURES_WITHLAG,
        'cw'      : False,
        'deskripsi': 'RF + lag — tanpa class weight',
        'model_fn': lambda cw, **kw: RandomForestClassifier(
            **kw, class_weight='balanced' if cw else None,
            random_state=RANDOM_STATE),
        'grid'    : {
            'n_estimators'      : [50, 100, 200],
            'max_depth'         : [3, 5, 7, None],
            'min_samples_split' : [2, 5, 10],
        }
    },
    {
        'no'      : 4,
        'nama'    : 'Random Forest',
        'fitur'   : FEATURES_WITHLAG,
        'cw'      : True,
        'deskripsi': 'RF + lag + class weight [KONFIGURASI TERBAIK?]',
        'model_fn': lambda cw, **kw: RandomForestClassifier(
            **kw, class_weight='balanced' if cw else None,
            random_state=RANDOM_STATE),
        'grid'    : {
            'n_estimators'      : [50, 100, 200],
            'max_depth'         : [3, 5, 7, None],
            'min_samples_split' : [2, 5, 10],
        }
    },
    {
        'no'      : 5,
        'nama'    : 'XGBoost',
        'fitur'   : FEATURES_NOLAG,
        'cw'      : False,
        'deskripsi': 'XGBoost murni — tanpa lag, tanpa class weight',
        'model_fn': lambda cw, **kw: XGBClassifier(
            **kw, eval_metric='mlogloss',
            random_state=RANDOM_STATE, verbosity=0),
        'grid'    : {
            'n_estimators'  : [50, 100, 200],
            'max_depth'     : [3, 5, 7],
            'learning_rate' : [0.05, 0.1, 0.2],
            'subsample'     : [0.8, 1.0],
        }
    },
    {
        'no'      : 6,
        'nama'    : 'XGBoost',
        'fitur'   : FEATURES_WITHLAG,
        'cw'      : False,
        'deskripsi': 'XGBoost + lag — tanpa class weight',
        'model_fn': lambda cw, **kw: XGBClassifier(
            **kw, eval_metric='mlogloss',
            random_state=RANDOM_STATE, verbosity=0),
        'grid'    : {
            'n_estimators'  : [50, 100, 200],
            'max_depth'     : [3, 5, 7],
            'learning_rate' : [0.05, 0.1, 0.2],
            'subsample'     : [0.8, 1.0],
        }
    },
    {
        'no'      : 7,
        'nama'    : 'XGBoost',
        'fitur'   : FEATURES_WITHLAG,
        'cw'      : True,
        'deskripsi': 'XGBoost + lag + class weight [KONFIGURASI TERBAIK?]',
        'model_fn': lambda cw, **kw: XGBClassifier(
            **kw, eval_metric='mlogloss',
            random_state=RANDOM_STATE, verbosity=0),
        'grid'    : {
            'n_estimators'  : [50, 100, 200],
            'max_depth'     : [3, 5, 7],
            'learning_rate' : [0.05, 0.1, 0.2],
            'subsample'     : [0.8, 1.0],
        }
    },
]

# ─────────────────────────────────────────────
# JALANKAN 7 EKSPERIMEN
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MENJALANKAN 7 EKSPERIMEN")
print("  (Mohon tunggu ±10-15 menit)")
print("=" * 65)

hasil_semua = []

for cfg in eksperimen_config:
    no   = cfg['no']
    nama = cfg['nama']
    cw   = cfg['cw']
    fitur = cfg['fitur']
    label_lag = 'Dengan lag' if fitur == FEATURES_WITHLAG else 'Tanpa lag'
    label_cw  = 'Dengan CW'  if cw else 'Tanpa CW'

    print(f"\n  [Eks.{no}] {nama} | {label_lag} | {label_cw}")
    print(f"          {cfg['deskripsi']}")
    print(f"          Jumlah fitur: {len(fitur)}")

    # Siapkan data
    Xtr, Xte, y_tr, y_te, sc = prepare_data(fitur)

    # Grid search
    print("          Mencari hyperparameter terbaik...", end=' ')

    # Untuk XGBoost dengan class weight, hitung scale_pos_weight
    def make_model(**kw):
        if cfg['nama'] == 'XGBoost' and cw:
            # XGBoost pakai sample_weight bukan class_weight
            return cfg['model_fn'](cw=False, **kw)
        return cfg['model_fn'](cw=cw, **kw)

    best_cvf, best_p = manual_grid_search(
        make_model, cfg['grid'], Xtr, y_tr
    )

    # Retrain dengan best params
    if cfg['nama'] == 'XGBoost' and cw:
        # XGBoost: terapkan class weight via sample_weight
        best_model = cfg['model_fn'](cw=False, **best_p)
        # Hitung sample weight per kelas
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight('balanced', y_tr)
        best_model.fit(Xtr, y_tr, sample_weight=sw)
        # Re-evaluate CV dengan sample weight
        cv_scores = []
        kf = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=RANDOM_STATE)
        for tr_idx, val_idx in kf.split(Xtr, y_tr):
            Xtr_f, Xval_f = Xtr[tr_idx], Xtr[val_idx]
            ytr_f, yval_f = y_tr[tr_idx], y_tr[val_idx]
            sw_f = compute_sample_weight('balanced', ytr_f)
            m_tmp = cfg['model_fn'](cw=False, **best_p)
            m_tmp.fit(Xtr_f, ytr_f, sample_weight=sw_f)
            yp_v = m_tmp.predict(Xval_f)
            cv_scores.append(f1_score(yval_f, yp_v, average='macro'))
        best_cvf = np.mean(cv_scores)
    else:
        best_model = cfg['model_fn'](cw=cw, **best_p)
        best_model.fit(Xtr, y_tr)

    print(f"selesai")

    # Evaluasi
    yp, f1m, fpc, acc, cm = evaluate(best_model, Xte, y_te)
    gap = abs(best_cvf - f1m)

    print(f"          CV F1   = {best_cvf:.4f}")
    print(f"          Test F1 = {f1m:.4f}  (Gap={gap:.4f}"
          f"{'  ⚠ tidak stabil' if gap > 0.10 else '  ✓ stabil'})")
    print(f"          F1 per kelas → Patuh={fpc[0]:.3f} | "
          f"Waspada={fpc[1]:.3f} | Kritis={fpc[2]:.3f}")

    hasil_semua.append({
        'no'        : no,
        'algoritma' : nama,
        'lag'       : label_lag,
        'cw'        : label_cw,
        'n_fitur'   : len(fitur),
        'cv_f1'     : best_cvf,
        'test_f1'   : f1m,
        'gap'       : gap,
        'f1_patuh'  : fpc[0],
        'f1_waspada': fpc[1],
        'f1_kritis' : fpc[2],
        'accuracy'  : acc,
        'cm'        : cm,
        'params'    : best_p,
        'model'     : best_model,
        'scaler'    : sc,
        'fitur'     : fitur,
        'y_pred'    : yp,
        'y_test'    : y_te,
    })

# ─────────────────────────────────────────────
# TABEL PERBANDINGAN LENGKAP
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TABEL PERBANDINGAN 7 EKSPERIMEN")
print("=" * 65)
print(f"  {'Eks':>3} {'Algoritma':15} {'Lag':12} {'CW':10} "
      f"{'CV F1':>7} {'Test F1':>8} {'Gap':>7} "
      f"{'Patuh':>7} {'Waspada':>8} {'Kritis':>7}")
print("  " + "-" * 82)

for r in hasil_semua:
    stabil = '✓' if r['gap'] <= 0.10 else '⚠'
    print(f"  {r['no']:>3} {r['algoritma']:15} {r['lag']:12} "
          f"{r['cw']:10} {r['cv_f1']:>7.4f} {r['test_f1']:>8.4f} "
          f"{r['gap']:>7.4f}{stabil} {r['f1_patuh']:>6.3f} "
          f"{r['f1_waspada']:>8.3f} {r['f1_kritis']:>7.3f}")

# ─────────────────────────────────────────────
# ANALISIS PERBANDINGAN
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ANALISIS — JAWABAN PERTANYAAN PENELITIAN")
print("=" * 65)

def get(no):
    return next(r for r in hasil_semua if r['no'] == no)

# Q1: Apakah lag meningkatkan RF?
r2, r3 = get(2), get(3)
delta_rf_lag = r3['cv_f1'] - r2['cv_f1']
print(f"\n  Q1: Apakah lag features meningkatkan Random Forest?")
print(f"      Eks.2 (tanpa lag) CV F1 = {r2['cv_f1']:.4f}")
print(f"      Eks.3 (dengan lag) CV F1 = {r3['cv_f1']:.4f}")
print(f"      Perubahan: {delta_rf_lag:+.4f} "
      f"({'✅ MENINGKAT' if delta_rf_lag > 0 else '❌ Tidak meningkat / sama'})")

# Q2: Apakah class weight meningkatkan RF?
r3, r4 = get(3), get(4)
delta_rf_cw = r4['cv_f1'] - r3['cv_f1']
delta_rf_cw_kritis = r4['f1_kritis'] - r3['f1_kritis']
print(f"\n  Q2: Apakah class weight meningkatkan Random Forest?")
print(f"      Eks.3 (tanpa CW)  CV F1 = {r3['cv_f1']:.4f} | "
      f"F1 Kritis = {r3['f1_kritis']:.3f}")
print(f"      Eks.4 (dengan CW) CV F1 = {r4['cv_f1']:.4f} | "
      f"F1 Kritis = {r4['f1_kritis']:.3f}")
print(f"      Perubahan CV F1  : {delta_rf_cw:+.4f} "
      f"({'✅ MENINGKAT' if delta_rf_cw > 0 else '→ sama/tidak signifikan'})")
print(f"      Perubahan F1 Kritis: {delta_rf_cw_kritis:+.3f} "
      f"({'✅ MENINGKAT' if delta_rf_cw_kritis > 0 else '→ sama'})")

# Q3: Apakah lag meningkatkan XGBoost?
r5, r6 = get(5), get(6)
delta_xgb_lag = r6['cv_f1'] - r5['cv_f1']
print(f"\n  Q3: Apakah lag features meningkatkan XGBoost?")
print(f"      Eks.5 (tanpa lag)  CV F1 = {r5['cv_f1']:.4f}")
print(f"      Eks.6 (dengan lag) CV F1 = {r6['cv_f1']:.4f}")
print(f"      Perubahan: {delta_xgb_lag:+.4f} "
      f"({'✅ MENINGKAT' if delta_xgb_lag > 0 else '❌ Tidak meningkat / sama'})")

# Q4: Apakah class weight meningkatkan XGBoost?
r6, r7 = get(6), get(7)
delta_xgb_cw = r7['cv_f1'] - r6['cv_f1']
delta_xgb_cw_kritis = r7['f1_kritis'] - r6['f1_kritis']
print(f"\n  Q4: Apakah class weight meningkatkan XGBoost?")
print(f"      Eks.6 (tanpa CW)  CV F1 = {r6['cv_f1']:.4f} | "
      f"F1 Kritis = {r6['f1_kritis']:.3f}")
print(f"      Eks.7 (dengan CW) CV F1 = {r7['cv_f1']:.4f} | "
      f"F1 Kritis = {r7['f1_kritis']:.3f}")
print(f"      Perubahan CV F1  : {delta_xgb_cw:+.4f} "
      f"({'✅ MENINGKAT' if delta_xgb_cw > 0 else '→ sama/tidak signifikan'})")
print(f"      Perubahan F1 Kritis: {delta_xgb_cw_kritis:+.3f} "
      f"({'✅ MENINGKAT' if delta_xgb_cw_kritis > 0 else '→ sama'})")

# Q5: RF vs XGBoost konfigurasi terbaik?
r4, r7 = get(4), get(7)
print(f"\n  Q5: RF vs XGBoost pada konfigurasi terbaik?")
print(f"      Eks.4 RF+lag+CW    CV F1 = {r4['cv_f1']:.4f} | "
      f"Test F1 = {r4['test_f1']:.4f}")
print(f"      Eks.7 XGB+lag+CW   CV F1 = {r7['cv_f1']:.4f} | "
      f"Test F1 = {r7['test_f1']:.4f}")

# Pilih model terbaik (stabil + CV F1 tertinggi)
stabil = [r for r in hasil_semua if r['gap'] <= 0.10]
best = max(stabil, key=lambda r: r['cv_f1'])
print(f"\n  ★ MODEL TERBAIK: Eks.{best['no']} — {best['algoritma']} "
      f"| {best['lag']} | {best['cw']}")
print(f"    CV Macro F1  = {best['cv_f1']:.4f}")
print(f"    Test Macro F1= {best['test_f1']:.4f}")
print(f"    F1 per kelas → Patuh={best['f1_patuh']:.3f} | "
      f"Waspada={best['f1_waspada']:.3f} | Kritis={best['f1_kritis']:.3f}")
print(f"    Best params  : {best['params']}")

# Feature importance
if hasattr(best['model'], 'feature_importances_'):
    print(f"\n  Feature Importance — Eks.{best['no']}:")
    fi = sorted(zip(best['fitur'], best['model'].feature_importances_),
                key=lambda x: x[1], reverse=True)
    for feat, imp in fi[:8]:
        bar = '█' * int(imp * 40)
        print(f"    {feat:35s} {imp:.4f} {bar}")

# ─────────────────────────────────────────────
# SIMPAN HASIL
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MENYIMPAN HASIL")
print("=" * 65)

# CSV perbandingan
rows = []
for r in hasil_semua:
    rows.append({
        'Eksperimen'  : f"Eks.{r['no']}",
        'Algoritma'   : r['algoritma'],
        'Lag Features': r['lag'],
        'Class Weight': r['cw'],
        'Jumlah Fitur': r['n_fitur'],
        'CV Macro F1' : round(r['cv_f1'], 4),
        'Test Macro F1': round(r['test_f1'], 4),
        'Gap CV-Test' : round(r['gap'], 4),
        'F1 Patuh'    : round(r['f1_patuh'], 3),
        'F1 Waspada'  : round(r['f1_waspada'], 3),
        'F1 Kritis'   : round(r['f1_kritis'], 3),
        'Accuracy'    : round(r['accuracy'], 4),
        'Best Params' : str(r['params']),
        'Keterangan'  : ('★ TERBAIK' if r['no'] == best['no']
                         else '⚠ Tidak stabil' if r['gap'] > 0.10
                         else ''),
    })

df_hasil = pd.DataFrame(rows)
csv_path = OUTPUT_DIR / 'perbandingan_7eksperimen.csv'
df_hasil.to_csv(csv_path, index=False)
print(f"  ✔ {csv_path}")

# Simpan model terbaik
model_path = OUTPUT_DIR / 'best_model.pkl'
artifact = {
    'model'        : best['model'],
    'model_name'   : f"Eks.{best['no']} — {best['algoritma']}",
    'scaler'       : best['scaler'],
    'label_map'    : LABEL_MAP,
    'label_inv'    : LABEL_INV,
    'feature_names': best['fitur'],
    'best_params'  : best['params'],
    'cv_f1'        : best['cv_f1'],
    'test_f1'      : best['test_f1'],
    'all_results'  : {
        f"Eks.{r['no']}": {
            'algoritma' : r['algoritma'],
            'lag'       : r['lag'],
            'cw'        : r['cw'],
            'cv_f1'     : r['cv_f1'],
            'test_f1'   : r['test_f1'],
            'gap'       : r['gap'],
            'f1_per'    : [r['f1_patuh'], r['f1_waspada'], r['f1_kritis']],
        }
        for r in hasil_semua
    }
}
with open(model_path, 'wb') as f:
    pickle.dump(artifact, f)
print(f"  ✔ {model_path}")

# ─────────────────────────────────────────────
# RINGKASAN AKHIR
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RINGKASAN AKHIR")
print("=" * 65)
print(f"  Total eksperimen  : 7")
print(f"  Model terbaik     : Eks.{best['no']} — {best['algoritma']} "
      f"({best['lag']}, {best['cw']})")
print(f"  CV Macro F1       : {best['cv_f1']:.4f}")
print(f"  Test Macro F1     : {best['test_f1']:.4f}")
print()
print(f"  Lag features meningkatkan RF?      : "
      f"{'Ya ✅' if delta_rf_lag > 0 else 'Tidak / sama ➡'} "
      f"({delta_rf_lag:+.4f})")
print(f"  Class weight meningkatkan RF?      : "
      f"{'Ya ✅' if delta_rf_cw > 0 else 'Tidak / sama ➡'} "
      f"({delta_rf_cw:+.4f})")
print(f"  Lag features meningkatkan XGBoost? : "
      f"{'Ya ✅' if delta_xgb_lag > 0 else 'Tidak / sama ➡'} "
      f"({delta_xgb_lag:+.4f})")
print(f"  Class weight meningkatkan XGBoost? : "
      f"{'Ya ✅' if delta_xgb_cw > 0 else 'Tidak / sama ➡'} "
      f"({delta_xgb_cw:+.4f})")
print()
print(f"  Output tersimpan di: ./{OUTPUT_DIR}/")
print(f"    - perbandingan_7eksperimen.csv")
print(f"    - best_model.pkl")
print("=" * 65)
print("  ✅ 7 Eksperimen selesai! Siap dilanjutkan ke dashboard.")
print("=" * 65)
