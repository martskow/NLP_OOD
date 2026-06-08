import os
import pandas as pd
import numpy as np
from scipy import stats

PATH_EXP_1_2 = r"C:\Users\marts\Downloads\SNIPS_OOD_Project_exp1_2\SNIPS_OOD_Project_exp1_2"
PATH_EXP_3   = r"C:\Users\marts\Downloads\SNIPS_OOD_Project-exp3\SNIPS_OOD_Project"
PATH_EXP_4_5 = r"C:\Users\marts\Downloads\SNIPS_OOD_Project-ex4_5\SNIPS_OOD_Project"

def load_and_clean(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"OSTRZEŻENIE: Brak pliku {filename} w {folder}")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()
    return df


data = {
    # SCENARIUSZ LOO (Leave-One-Out)
    ('LOO', 'Exp1'): load_and_clean(PATH_EXP_1_2, "wyniki_mahalanobis_loo.csv"),
    ('LOO', 'Exp2'): load_and_clean(PATH_EXP_1_2, "wyniki_rangle_loo.csv"),
    ('LOO', 'Exp3'): load_and_clean(PATH_EXP_3,   "wyniki_loo_per_class.csv"),
    ('LOO', 'Exp4'): load_and_clean(PATH_EXP_4_5, "wyniki_loo_mahalanobis.csv"),
    ('LOO', 'Exp5'): load_and_clean(PATH_EXP_4_5, "wyniki_loo_rangle.csv"),

    # SCENARIUSZ LTO (Leave-Two-Out / OVR)
    ('LTO', 'Exp1'): load_and_clean(PATH_EXP_1_2, "wyniki_mahalanobis_lto.csv"),
    ('LTO', 'Exp2'): load_and_clean(PATH_EXP_1_2, "wyniki_rangle_lto.csv"),
    ('LTO', 'Exp3'): load_and_clean(PATH_EXP_3,   "wyniki_szczegolowe_per_fold_exp3.csv"),
    ('LTO', 'Exp4'): load_and_clean(PATH_EXP_4_5, "wyniki_mahalanobis_lto.csv"),
    ('LTO', 'Exp5'): load_and_clean(PATH_EXP_4_5, "wyniki_rangle_lto.csv"),
}


# 2. CAŁKOWITY PLAN BADANIA

experimental_plan = [
    # --- RQ1 ---
    ('RQ1', 'LOO', 'Exp 3 (UNK)', 'Exp 1 (CE + Mahal.)', 'Exp3', 'Exp1'),
    ('RQ1', 'LOO', 'Exp 3 (UNK)', 'Exp 2 (CE + R-Angle)', 'Exp3', 'Exp2'),
    ('RQ1', 'LOO', 'Exp 3 (UNK)', 'Exp 4 (SupCon + Mahal.)', 'Exp3', 'Exp4'),
    ('RQ1', 'LOO', 'Exp 3 (UNK)', 'Exp 5 (SupCon + R-Angle)', 'Exp3', 'Exp5'),
    
    ('RQ1', 'LTO', 'Exp 3 (UNK)', 'Exp 1 (CE + Mahal.)', 'Exp3', 'Exp1'),
    ('RQ1', 'LTO', 'Exp 3 (UNK)', 'Exp 2 (CE + R-Angle)', 'Exp3', 'Exp2'),
    ('RQ1', 'LTO', 'Exp 3 (UNK)', 'Exp 4 (SupCon + Mahal.)', 'Exp3', 'Exp4'),
    ('RQ1', 'LTO', 'Exp 3 (UNK)', 'Exp 5 (SupCon + R-Angle)', 'Exp3', 'Exp5'),

    # --- RQ2 ---
    ('RQ2', 'LOO', 'Exp 1 (CE)', 'Exp 4 (SupCon) [Mahal.]', 'Exp1', 'Exp4'),
    ('RQ2', 'LOO', 'Exp 2 (CE)', 'Exp 5 (SupCon) [R-Angle]', 'Exp2', 'Exp5'),
    ('RQ2', 'LTO', 'Exp 1 (CE)', 'Exp 4 (SupCon) [Mahal.]', 'Exp1', 'Exp4'),
    ('RQ2', 'LTO', 'Exp 2 (CE)', 'Exp 5 (SupCon) [R-Angle]', 'Exp2', 'Exp5'),

    # --- RQ3 ---
    ('RQ3', 'LOO', 'Exp 1 (Mahal.)', 'Exp 2 (R-Angle) [CE]', 'Exp1', 'Exp2'),
    ('RQ3', 'LOO', 'Exp 4 (Mahal.)', 'Exp 5 (R-Angle) [SupCon]', 'Exp4', 'Exp5'),
    ('RQ3', 'LTO', 'Exp 1 (Mahal.)', 'Exp 2 (R-Angle) [CE]', 'Exp1', 'Exp2'),
    ('RQ3', 'LTO', 'Exp 4 (Mahal.)', 'Exp 5 (R-Angle) [SupCon]', 'Exp4', 'Exp5')
]


# 3. ANALIZA STATYSTYCZNA I RAPORTOWANIE
results_report = []

for rq, scenario, label_a, label_b, key_a, key_b in experimental_plan:
    df_a = data.get((scenario, key_a))
    df_b = data.get((scenario, key_b))
    
    if df_a is None or df_b is None:
        continue
        
    for metric in ['AUROC', 'FPR95']:
        try:
            vec_a = df_a[metric].values
            vec_b = df_b[metric].values
            
            # Wektor różnic dla testu normalności
            diff = vec_a - vec_b
            if len(diff) < 3: # zabezpieczenie przed pustymi/za krótkimi danymi
                continue
                
            _, p_shapiro = stats.shapiro(diff)
            
            # Wybór testu na podstawie normalności (Shapiro-Wilk)
            if p_shapiro > 0.05:
                stat, p_val = stats.ttest_rel(vec_a, vec_b)
                test_type = 't-Student'
            else:
                stat, p_val = stats.wilcoxon(vec_a, vec_b, zero_method='pratt')
                test_type = 'Wilcoxon'
                
            results_report.append({
                'RQ': rq, 'Scenariusz': scenario, 'Para': f"{label_a} vs {label_b}",
                'Metryka': metric, 'Średnia A': np.mean(vec_a), 'Średnia B': np.mean(vec_b),
                'Test': test_type, 'Statystyka': stat, 'p-value': p_val, 'Istotny': 'Tak' if p_val < 0.05 else 'Nie'
            })
        except Exception as e:
            print(f"Błąd przetwarzania: {rq} | {scenario} | {metric} | Szczegóły: {e}")


df_report = pd.DataFrame(results_report)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
    
print(df_report[['RQ', 'Scenariusz', 'Para', 'Metryka', 'Test', 'Statystyka', 'p-value', 'Istotny']].to_string(index=False))
df_report.to_csv(r"C:\Users\marts\Downloads\ostateczny_raport_statystyczny.csv", index=False)