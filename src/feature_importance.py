"""
Feature-Wichtigkeits-Analyse: SHAP + PCA.

Einfach ausfuehren mit:
    python src/feature_importance.py

Voraussetzung: mindestens ein trainiertes Vintage unter models/vintage_*/
(siehe train.py) sowie data/training_table.parquet. Standardmaessig wird das
NEUESTE Vintage analysiert (per Ordnername sortiert - ISO-Datumsformat
'vintage_YYYY-MM-DD' sortiert lexikalisch korrekt chronologisch).

ZWEI GRUNDVERSCHIEDENE FRAGEN, ZWEI GRUNDVERSCHIEDENE WERKZEUGE:

  SHAP (ueberwacht - nutzt Modell + Zielvariable):
    "Welches Feature hat die Vorhersage fuer einen konkreten Trade in
    welche Richtung getrieben?" - das ist echte PRAEDIKTIVE Wichtigkeit.
    Wird hier fuer die beiden Baummodelle (LightGBM, Random Forest) per
    TreeExplainer exakt berechnet (kein Sampling-Approximationsfehler wie
    bei model-agnostischen SHAP-Varianten). Fuer die logistische Regression
    wird stattdessen der standardisierte Koeffizient (Beta * Feature-Std)
    verwendet - das ist die lineare Entsprechung, faengt aber keine
    Interaktionen/Nichtlinearitaeten ein, die die Baummodelle sehen wuerden.

  PCA (unueberwacht - kennt die Zielvariable NICHT):
    "Welche Features tragen redundante/korrelierte Information?" - sagt
    NICHTS darueber aus, was fuers Modell wichtig ist (z.B. koennte eine PCA-
    Hauptkomponente von acht hoch korrelierten HV-Fenstern dominiert werden,
    obwohl nur eines davon fuer die Vorhersage tatsaechlich gebraucht wird).
    Nuetzlich, um Redundanz zu erkennen (z.B. "brauchen wir wirklich alle 8
    HV-Fenster?"), NICHT als Ersatz fuer SHAP.

WICHTIG - SHAP-Werte von LightGBM und Random Forest sind NICHT direkt in
absoluten Einheiten vergleichbar (unterschiedliche Rohausgabe-Raeume: RF ist
nativ in Wahrscheinlichkeits-Raum, LightGBM im Margin-/Log-Odds-Raum vor
Softmax). Deshalb wird JEDES Modell zuerst separat auf eine relative
Wichtigkeit normiert (Summe = 1 ueber alle Features), und erst DANACH ueber
die Modelle gemittelt - so ist der kombinierte Rang fair, auch wenn die
Rohwerte unterschiedliche Skalen haben.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import load_vintage
from labels import CLASS_LABELS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TRAINING_TABLE_FILE = DATA_DIR / "training_table.parquet"

SHAP_SAMPLE_SIZE = 4000   # Stichprobengroesse fuer die SHAP-Berechnung (Performance)
RANDOM_STATE = 42


def pick_latest_vintage() -> Path:
    vintage_dirs = sorted(MODELS_DIR.glob("vintage_*"))
    if not vintage_dirs:
        print(f"[fehler] keine vintage_*-Ordner unter {MODELS_DIR} gefunden.")
        print("[info] Erst ausfuehren: python src/train.py")
        sys.exit(1)
    return vintage_dirs[-1]   # ISO-Datum im Ordnernamen -> lexikalisch = chronologisch


def _tree_shap_importance(model, X: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """Mittlerer |SHAP|-Wert je Feature, gemittelt ueber alle Klassen UND
    alle Stichproben-Zeilen - das ist die uebliche globale SHAP-Wichtigkeit.
    Gibt zusaetzlich den mittleren VORZEICHENBEHAFTETEN SHAP-Wert je Klasse
    zurueck (zeigt Richtung: treibt das Feature Richtung 'stark_positiv'
    oder 'stark_negativ'?)."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    # Aeltere shap-Versionen geben eine Liste (eine Matrix je Klasse) zurueck,
    # neuere ein einzelnes Array der Form (n_samples, n_features, n_classes) -
    # hier auf ein einheitliches (n_samples, n_features, n_classes) bringen.
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)

    mean_abs = np.mean(np.abs(sv), axis=(0, 2))          # global, ueber Klassen+Zeilen gemittelt
    mean_signed_per_class = np.mean(sv, axis=0)            # (n_features, n_classes), vorzeichenbehaftet

    out = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    for j, cls in enumerate(model.classes_):
        out[f"mean_signed_shap_{cls}"] = mean_signed_per_class[:, j]
    return out


def _logreg_importance(pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """Standardisierter Koeffizient (Beta * Feature-Std) je Klasse - die
    lineare Entsprechung zur SHAP-Wichtigkeit. `pipeline` ist ein
    sklearn Pipeline(StandardScaler, LogisticRegression)."""
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    # coef_ hat Form (n_classes, n_features) bei multinomialer Regression
    standardized_coef = clf.coef_ * scaler.scale_[None, :]

    out = pd.DataFrame({"feature": feature_cols})
    for j, cls in enumerate(clf.classes_):
        out[f"logreg_coef_{cls}"] = standardized_coef[j, :]
    out["mean_abs_logreg_coef"] = np.mean(np.abs(standardized_coef), axis=0)
    return out


def run_shap_analysis(vintage_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_vintage(vintage_dir)
    feature_cols = bundle["metadata"]["feature_columns"]
    models = bundle["models"]

    sample_df = df[feature_cols].dropna()
    n = min(SHAP_SAMPLE_SIZE, len(sample_df))
    sample_df = sample_df.sample(n=n, random_state=RANDOM_STATE)
    X = sample_df.to_numpy(dtype=float)
    print(f"[ok] SHAP wird auf einer Stichprobe von {n:,} Zeilen berechnet "
          f"(Vintage: {vintage_dir.name}, {len(feature_cols)} Features).")

    combined = pd.DataFrame({"feature": feature_cols})

    tree_norm_cols = []
    for name in ("lgbm", "rf"):
        if name not in models:
            continue
        print(f"[ok] berechne SHAP fuer '{name}' ...")
        tree_df = _tree_shap_importance(models[name], X, feature_cols)
        norm_col = f"{name}_importance_norm"
        tree_df[norm_col] = tree_df["mean_abs_shap"] / tree_df["mean_abs_shap"].sum()
        combined = combined.merge(
            tree_df[["feature", "mean_abs_shap", norm_col]].rename(
                columns={"mean_abs_shap": f"{name}_mean_abs_shap"}),
            on="feature",
        )
        tree_norm_cols.append(norm_col)

        # Richtungs-Detail fuer die beiden Extremklassen separat speichern
        signed_path = MODELS_DIR / f"feature_importance_{name}_signed_by_class.csv"
        tree_df.to_csv(signed_path, index=False)
        print(f"      -> vorzeichenbehaftete Klassen-Details gespeichert unter {signed_path}")

    if "logreg" in models:
        print("[ok] berechne standardisierte Koeffizienten fuer 'logreg' ...")
        lr_df = _logreg_importance(models["logreg"], feature_cols)
        lr_df["logreg_importance_norm"] = lr_df["mean_abs_logreg_coef"] / lr_df["mean_abs_logreg_coef"].sum()
        combined = combined.merge(
            lr_df[["feature", "mean_abs_logreg_coef", "logreg_importance_norm"]], on="feature",
        )
        lr_df.to_csv(MODELS_DIR / "feature_importance_logreg_signed_by_class.csv", index=False)
        norm_cols_for_combined = tree_norm_cols + ["logreg_importance_norm"]
    else:
        norm_cols_for_combined = tree_norm_cols

    combined["combined_importance"] = combined[norm_cols_for_combined].mean(axis=1)
    combined = combined.sort_values("combined_importance", ascending=False).reset_index(drop=True)
    return combined


def run_pca_analysis(df: pd.DataFrame, feature_cols: list[str], n_components: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_df = df[feature_cols].dropna()
    print(f"[ok] PCA auf {len(X_df):,} vollstaendigen Zeilen (NaN-Zeilen, v.a. "
          f"Warmup der langen Fenster wie ret_5y, ausgeschlossen).")

    X_scaled = StandardScaler().fit_transform(X_df.to_numpy(dtype=float))
    n_components = min(n_components, len(feature_cols))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(X_scaled)

    variance_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(n_components)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })

    loadings_df = pd.DataFrame(
        pca.components_.T, index=feature_cols,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    return variance_df, loadings_df


def print_top_loadings(loadings_df: pd.DataFrame, n_pcs: int = 5, top_k: int = 6):
    for i in range(min(n_pcs, loadings_df.shape[1])):
        col = f"PC{i+1}"
        top = loadings_df[col].abs().sort_values(ascending=False).head(top_k)
        print(f"\n  {col} - dominierende Features (Betrag der Ladung):")
        for feat in top.index:
            print(f"    {feat:30s} {loadings_df.loc[feat, col]:+.3f}")


def main():
    if not TRAINING_TABLE_FILE.exists():
        print(f"[fehler] {TRAINING_TABLE_FILE} nicht gefunden.")
        print("[info] Erst ausfuehren: python src/build_dataset.py")
        sys.exit(1)

    print(f"[ok] lade {TRAINING_TABLE_FILE} ...")
    df = pd.read_parquet(TRAINING_TABLE_FILE)

    vintage_dir = pick_latest_vintage()
    print(f"[ok] verwende Vintage: {vintage_dir.name}")

    print("\n" + "=" * 70)
    print("SHAP - PRAEDIKTIVE FEATURE-WICHTIGKEIT (ueberwacht)")
    print("=" * 70)
    shap_result = run_shap_analysis(vintage_dir, df)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print("\nTop 20 Features (kombiniert, gleichgewichtetes Mittel der normierten "
              "Wichtigkeit ueber alle verfuegbaren Modelle):")
        print(shap_result.head(20).to_string(index=False))

    shap_out_path = MODELS_DIR / "feature_importance_combined.csv"
    shap_result.to_csv(shap_out_path, index=False)
    print(f"\n[ok] gespeichert unter: {shap_out_path}")

    print("\n" + "=" * 70)
    print("PCA - REDUNDANZ/KORRELATIONSSTRUKTUR (unueberwacht, KEINE Wichtigkeit!)")
    print("=" * 70)
    bundle = load_vintage(vintage_dir)
    feature_cols = bundle["metadata"]["feature_columns"]
    variance_df, loadings_df = run_pca_analysis(df, feature_cols, n_components=15)

    print("\nErklaerte Varianz je Hauptkomponente:")
    print(variance_df.to_string(index=False))
    print(f"\n-> Die ersten {(variance_df['cumulative_variance_ratio'] < 0.80).sum() + 1} "
          f"Komponenten erklaeren bereits 80% der Gesamtvarianz der {len(feature_cols)} Features "
          f"- ein Hinweis auf deutliche Redundanz im Feature-Set.")

    print_top_loadings(loadings_df, n_pcs=5, top_k=6)

    variance_path = MODELS_DIR / "pca_explained_variance.csv"
    loadings_path = MODELS_DIR / "pca_loadings.csv"
    variance_df.to_csv(variance_path, index=False)
    loadings_df.to_csv(loadings_path)
    print(f"\n[ok] gespeichert unter:\n  {variance_path}\n  {loadings_path}")


if __name__ == "__main__":
    main()