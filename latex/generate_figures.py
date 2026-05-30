"""Generate CL paper tables and figures from real mismatch CSV results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LATEX_DIR = ROOT / "latex"
FIG_DIR = LATEX_DIR / "figures"
DATA_DIR = ROOT / "mismatch_results"
SNRS = [1, 4, 7, 10, 13]
BOUNDED_TAU = 3

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"

PRIMARY_RESULTS = [
    ("Original", "Original", "original_cifar10_awgn_C32_msssim_cpu.csv", True),
    (
        "D0",
        "Perfect-SNR FT (D0)",
        [
            "ua-d0-eval_CIFAR10_awgn_SwinJSCC_w-SA_C32_ckpt-*_EP10_msssim_cpu.csv",
            "ua-d0-eval_CIFAR10_awgn_SwinJSCC_w-SA_C32_ckpt-*_EP5_msssim_cpu.csv",
        ],
        True,
    ),
    ("D1", "UA-D1", "ua-d1-eval_CIFAR10_awgn_SwinJSCC_w-SA_C32_ckpt-*_EP10_msssim_cpu.csv", True),
    ("D3", "UA-D3", "ua_delta3_cifar10_awgn_C32_msssim_cpu.csv", True),
    ("D6", "UA-D6", "ua-d6-eval_CIFAR10_awgn_SwinJSCC_w-SA_C32_ckpt-*_EP10_msssim_cpu.csv", True),
    ("RandomHat", "Random-hat", "random-hat-eval_CIFAR10_awgn_SwinJSCC_w-SA_C32_ckpt-*_EP10_msssim_cpu.csv", True),
]

ROBUST_RESULTS = [
    ("Tail", "Tail-UA", "tail_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv"),
    ("Cons", "Cons-UA", "cons_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv"),
    ("TailCons", "Tail+Cons-UA", "tail_cons_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv"),
    ("TailLt1", "Tail-UA ($\\lambda_t=1$)", "tail-ua-d3-lt1-eval_CIFAR10_awgn_SwinJSCC_w-SA_C32_ckpt-*_EP10_msssim_cpu.csv"),
]

RAYLEIGH_RESULTS = [
    ("RayOrig", "Original", "rayleigh-original-eval_CIFAR10_rayleigh_SwinJSCC_w-SA_C32_ckpt-*_msssim_cpu.csv"),
    ("RayUA", "UA-D3", "rayleigh-ua-d3-eval_CIFAR10_rayleigh_SwinJSCC_w-SA_C32_ckpt-*_EP10_msssim_cpu.csv"),
]

FIG3_METHODS = ["Original", "D0", "D1", "D3", "D6", "RandomHat"]
FIG4_METHODS = ["D1", "D3", "D6", "RandomHat"]
MAGNITUDES = [3, 6, 9, 12]
QUAL_CASES = [(1, 7), (4, 10), (13, 1)]

COLORS = {
    "matched": "#222222",
    "bounded": "#0072B2",
    "off": "#009E73",
    "worst": "#D55E00",
    "D1": "#0072B2",
    "D3": "#D55E00",
    "D6": "#009E73",
    "RandomHat": "#CC79A7",
}


def _find_csv(pattern: str) -> Path | None:
    exact = DATA_DIR / pattern
    if exact.exists():
        return exact
    matches = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    return None


def resolve_csv(pattern: str | list[str], required: bool) -> Path | None:
    if isinstance(pattern, list):
        for candidate in pattern:
            path = _find_csv(candidate)
            if path is not None:
                return path
        if required:
            raise FileNotFoundError(f"Missing required result CSV matching any of {pattern!r} in {DATA_DIR}")
        print(f"WARNING: missing optional result CSV matching any of {pattern!r}; emitting TODO row.")
        return None

    path = _find_csv(pattern)
    if path is not None:
        return path
    if required:
        raise FileNotFoundError(f"Missing required result CSV matching {pattern!r} in {DATA_DIR}")
    print(f"WARNING: missing optional result CSV matching {pattern!r}; emitting TODO row.")
    return None


def read_result(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"SNR_true", "SNR_hat", "PSNR", "MS_SSIM", "MS_SSIM_dB"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(df) != 25:
        raise ValueError(f"{path} should contain a 5x5 mismatch matrix, got {len(df)} rows")
    return df


def summarize(df: pd.DataFrame) -> dict[str, float]:
    diag = df["SNR_true"].eq(df["SNR_hat"])
    off = ~diag
    bounded = off & ((df["SNR_hat"] - df["SNR_true"]).abs() <= BOUNDED_TAU)
    return {
        "diag_psnr": df.loc[diag, "PSNR"].mean(),
        "bounded_psnr": df.loc[bounded, "PSNR"].mean(),
        "off_psnr": df.loc[off, "PSNR"].mean(),
        "worst_psnr": df["PSNR"].min(),
        "diag_msd": df.loc[diag, "MS_SSIM_dB"].mean(),
        "bounded_msd": df.loc[bounded, "MS_SSIM_dB"].mean(),
        "off_msd": df.loc[off, "MS_SSIM_dB"].mean(),
        "worst_msd": df["MS_SSIM_dB"].min(),
    }


def matrix(df: pd.DataFrame, metric: str) -> np.ndarray:
    pivot = df.pivot(index="SNR_true", columns="SNR_hat", values=metric)
    return pivot.loc[SNRS, SNRS].to_numpy()


def load_primary() -> tuple[dict[str, pd.DataFrame], dict[str, str], set[str]]:
    data: dict[str, pd.DataFrame] = {}
    labels: dict[str, str] = {}
    missing: set[str] = set()
    for key, label, pattern, required in PRIMARY_RESULTS:
        labels[key] = label
        path = resolve_csv(pattern, required)
        if path is None:
            missing.add(key)
            continue
        data[key] = read_result(path)
    return data, labels, missing


def load_rayleigh() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    data: dict[str, pd.DataFrame] = {}
    labels: dict[str, str] = {}
    for key, label, pattern in RAYLEIGH_RESULTS:
        labels[key] = label
        path = resolve_csv(pattern, required=False)
        if path is not None:
            data[key] = read_result(path)
    return data, labels


def load_auxiliary() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    data: dict[str, pd.DataFrame] = {}
    labels: dict[str, str] = {}
    for key, label, pattern in ROBUST_RESULTS:
        labels[key] = label
        path = resolve_csv(pattern, required=False)
        if path is not None:
            data[key] = read_result(path)
    return data, labels


def fmt(value: float) -> str:
    return f"{value:.2f}"


def gain(value: float, reference: float) -> str:
    return f"{value - reference:+.2f}"


def strip_svg(path: Path) -> None:
    svg_text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")


def write_tradeoff_table(data: dict[str, pd.DataFrame], labels: dict[str, str], missing: set[str]) -> dict[str, dict[str, float]]:
    summaries = {key: summarize(df) for key, df in data.items()}
    original = summaries["Original"]
    rows = ["% Auto-generated by latex/generate_figures.py"]

    for key, label, _, _ in PRIMARY_RESULTS:
        if key in missing:
            rows.append(f"{label} & \\multicolumn{{7}}{{c}}{{TODO: result CSV missing}} \\\\")
            continue
        vals = summaries[key]
        rows.append(
            " & ".join(
                [
                    labels[key],
                    fmt(vals["diag_psnr"]),
                    fmt(vals["bounded_psnr"]),
                    fmt(vals["off_psnr"]),
                    fmt(vals["worst_psnr"]),
                    gain(vals["bounded_psnr"], original["bounded_psnr"]),
                    gain(vals["off_psnr"], original["off_psnr"]),
                    fmt(vals["bounded_msd"]),
                ]
            )
            + r" \\"
        )
    (LATEX_DIR / "table_tradeoff_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summaries


def write_rayleigh_table(data: dict[str, pd.DataFrame], labels: dict[str, str]) -> dict[str, dict[str, float]]:
    rows = ["% Auto-generated by latex/generate_figures.py"]
    summaries = {key: summarize(df) for key, df in data.items()}
    if {"RayOrig", "RayUA"}.issubset(summaries):
        orig = summaries["RayOrig"]
        for key in ["RayOrig", "RayUA"]:
            vals = summaries[key]
            rows.append(
                " & ".join(
                    [
                        labels[key],
                        fmt(vals["diag_psnr"]),
                        fmt(vals["off_psnr"]),
                        fmt(vals["worst_psnr"]),
                        gain(vals["off_psnr"], orig["off_psnr"]),
                        fmt(vals["off_msd"]),
                    ]
                )
                + r" \\"
            )
    else:
        rows.append("\\multicolumn{6}{c}{Rayleigh CPU results unavailable} \\\\")
    (LATEX_DIR / "table_rayleigh_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summaries


def write_auxiliary_table(
    data: dict[str, pd.DataFrame],
    labels: dict[str, str],
    original: dict[str, float],
) -> dict[str, dict[str, float]]:
    rows = ["% Auto-generated by latex/generate_figures.py"]
    summaries = {key: summarize(df) for key, df in data.items()}
    if summaries:
        for key, label, _ in ROBUST_RESULTS:
            if key not in summaries:
                continue
            vals = summaries[key]
            rows.append(
                " & ".join(
                    [
                        labels[key],
                        fmt(vals["diag_psnr"]),
                        fmt(vals["off_psnr"]),
                        fmt(vals["worst_psnr"]),
                        gain(vals["off_psnr"], original["off_psnr"]),
                        gain(vals["worst_psnr"], original["worst_psnr"]),
                    ]
                )
                + r" \\"
            )
    else:
        rows.append("\\multicolumn{6}{c}{Auxiliary robustness results unavailable} \\\\")
    (LATEX_DIR / "table_auxiliary_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summaries


def write_metrics(
    primary: dict[str, dict[str, float]],
    auxiliary: dict[str, dict[str, float]],
    rayleigh: dict[str, dict[str, float]],
) -> None:
    orig = primary["Original"]
    d0 = primary["D0"]
    d3 = primary["D3"]
    d6 = primary["D6"]
    rand = primary["RandomHat"]
    lines = [
        "% Auto-generated by latex/generate_figures.py",
        f"\\newcommand{{\\DZeroBoundedGain}}{{{d0['bounded_psnr'] - orig['bounded_psnr']:.2f}}}",
        f"\\newcommand{{\\DZeroOffGain}}{{{d0['off_psnr'] - orig['off_psnr']:.2f}}}",
        f"\\newcommand{{\\DZeroDiagGain}}{{{d0['diag_psnr'] - orig['diag_psnr']:.2f}}}",
        f"\\newcommand{{\\DThreeBoundedGain}}{{{d3['bounded_psnr'] - orig['bounded_psnr']:.2f}}}",
        f"\\newcommand{{\\DThreeOffGain}}{{{d3['off_psnr'] - orig['off_psnr']:.2f}}}",
        f"\\newcommand{{\\DThreeDiagGain}}{{{d3['diag_psnr'] - orig['diag_psnr']:.2f}}}",
        f"\\newcommand{{\\DSixBoundedGain}}{{{d6['bounded_psnr'] - orig['bounded_psnr']:.2f}}}",
        f"\\newcommand{{\\DSixOffGain}}{{{d6['off_psnr'] - orig['off_psnr']:.2f}}}",
        f"\\newcommand{{\\DSixDiagGain}}{{{d6['diag_psnr'] - orig['diag_psnr']:.2f}}}",
        f"\\newcommand{{\\RandomBoundedGain}}{{{rand['bounded_psnr'] - orig['bounded_psnr']:.2f}}}",
        f"\\newcommand{{\\RandomOffGain}}{{{rand['off_psnr'] - orig['off_psnr']:.2f}}}",
        f"\\newcommand{{\\RandomDiagGain}}{{{rand['diag_psnr'] - orig['diag_psnr']:.2f}}}",
    ]
    if "TailCons" in auxiliary:
        lines.append(
            f"\\newcommand{{\\TailConsWorstGain}}{{{auxiliary['TailCons']['worst_psnr'] - orig['worst_psnr']:.2f}}}"
        )
    else:
        lines.append("\\newcommand{\\TailConsWorstGain}{TODO}")
    if "TailLt1" in auxiliary:
        lines.append(
            f"\\newcommand{{\\TailLtOneWorstGain}}{{{auxiliary['TailLt1']['worst_psnr'] - orig['worst_psnr']:.2f}}}"
        )
    else:
        lines.append("\\newcommand{\\TailLtOneWorstGain}{TODO}")
    if {"RayOrig", "RayUA"}.issubset(rayleigh):
        lines.append(f"\\newcommand{{\\RayleighOffGain}}{{{rayleigh['RayUA']['off_psnr'] - rayleigh['RayOrig']['off_psnr']:.2f}}}")
    else:
        lines.append("\\newcommand{\\RayleighOffGain}{TODO}")
    lines.append("")
    (LATEX_DIR / "generated_metrics.tex").write_text("\n".join(lines), encoding="utf-8")


def setup_heatmap_axes(ax: plt.Axes) -> None:
    ax.set_xticks(np.arange(len(SNRS)))
    ax.set_xticklabels(SNRS)
    ax.set_yticks(np.arange(len(SNRS)))
    ax.set_yticklabels(SNRS)
    ax.set_xlabel(r"Estimated SNR $\hat{\gamma}$ (dB)")
    ax.set_ylabel(r"True SNR $\gamma$ (dB)")


def annotate_values(ax: plt.Axes, values: np.ndarray, gain_mode: bool, threshold: float = 25.0) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if gain_mode:
                label = f"{values[i, j]:+.1f}"
                if "." not in label:
                    raise ValueError(f"Gain label lost decimal point: {label}")
                color = "black"
            else:
                label = f"{values[i, j]:.1f}"
                color = "white" if values[i, j] < threshold else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.4, color=color)


def plot_psnr_heatmap(data: dict[str, pd.DataFrame]) -> None:
    orig = matrix(data["Original"], "PSNR")
    d3 = matrix(data["D3"], "PSNR")
    diff = d3 - orig
    if np.any(np.abs(diff) >= 10):
        raise ValueError("Unexpected gain magnitude >= 10 dB; check heatmap labels before export.")

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), constrained_layout=True)
    vmin = min(orig.min(), d3.min())
    vmax = max(orig.max(), d3.max())
    text_threshold = vmin + 0.45 * (vmax - vmin)

    for ax, values, title in [(axes[0], orig, "Original"), (axes[1], d3, "UA-D3")]:
        im = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        setup_heatmap_axes(ax)
        annotate_values(ax, values, gain_mode=False, threshold=text_threshold)
    fig.colorbar(im, ax=axes[:2], shrink=0.82, pad=0.015, label="PSNR (dB)")

    gmax = max(abs(diff.min()), abs(diff.max()))
    im_gain = axes[2].imshow(diff, cmap="RdBu_r", vmin=-gmax, vmax=gmax)
    axes[2].set_title("UA-D3 - Original", fontsize=9)
    setup_heatmap_axes(axes[2])
    annotate_values(axes[2], diff, gain_mode=True)
    fig.colorbar(im_gain, ax=axes[2], shrink=0.82, pad=0.015, label="Gain (dB)")

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"fig2_psnr_heatmaps.{ext}", bbox_inches="tight", dpi=450)
    plt.close(fig)
    strip_svg(FIG_DIR / "fig2_psnr_heatmaps.svg")

    wrapper = "\n".join(
        [
            "% Auto-generated by latex/generate_figures.py",
            "\\begin{figure*}[t]",
            "\\centering",
            "\\includegraphics[width=\\textwidth]{figures/fig2_psnr_heatmaps}",
            "\\caption{PSNR heatmaps under SNR mismatch on CIFAR10/AWGN/C=32. Rows denote the true physical SNR $\\gamma$, columns denote the estimated modulation SNR $\\hat{\\gamma}$, and the right panel reports the UA-D3 gain over the original checkpoint.}",
            "\\label{fig:psnr-heatmaps}",
            "\\end{figure*}",
            "",
        ]
    )
    (FIG_DIR / "fig2_psnr_heatmaps.tex").write_text(wrapper, encoding="utf-8")


def save_line_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=450)
    plt.close(fig)
    strip_svg(FIG_DIR / f"{stem}.svg")


def draw_delta_tradeoff(ax: plt.Axes, primary: dict[str, dict[str, float]], labels: dict[str, str]) -> None:
    x = np.arange(len(FIG3_METHODS))
    series = [
        ("Matched", [primary[k]["diag_psnr"] for k in FIG3_METHODS], "-", "o", COLORS["matched"]),
        ("Full off.", [primary[k]["off_psnr"] for k in FIG3_METHODS], "--", "s", COLORS["off"]),
        ("Worst", [primary[k]["worst_psnr"] for k in FIG3_METHODS], "-.", "^", COLORS["worst"]),
    ]
    for name, values, linestyle, marker, color in series:
        ax.plot(x, values, label=name, linestyle=linestyle, marker=marker, linewidth=1.35, markersize=4.0, color=color)

    tick_labels = [labels[k].replace("Perfect-SNR FT ", "").replace("Random-hat", "Rand.") for k in FIG3_METHODS]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.28, linewidth=0.6)
    ax.legend(fontsize=6.7, frameon=False, ncol=2)
    ax.set_title("(a) Robustness--fidelity trade-off", fontsize=9)


def plot_delta_tradeoff(primary: dict[str, dict[str, float]], labels: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.35), constrained_layout=True)
    draw_delta_tradeoff(ax, primary, labels)
    save_line_figure(fig, "fig3_delta_tradeoff")

    wrapper = "\n".join(
        [
            "% Auto-generated by latex/generate_figures.py",
            "\\begin{figure}[t]",
            "\\centering",
            "\\includegraphics[width=\\columnwidth]{figures/fig3_delta_tradeoff}",
            "\\caption{PSNR trade-off across SNR-conditioning strategies. Line styles denote metrics, not methods.}",
            "\\label{fig:legacy-delta-tradeoff}",
            "\\end{figure}",
            "",
        ]
    )
    (FIG_DIR / "fig3_delta_tradeoff.tex").write_text(wrapper, encoding="utf-8")


def magnitude_gain(df: pd.DataFrame, original: pd.DataFrame, magnitude: int) -> float:
    merged = df.merge(
        original[["SNR_true", "SNR_hat", "PSNR"]],
        on=["SNR_true", "SNR_hat"],
        suffixes=("", "_orig"),
    )
    mask = (merged["SNR_hat"] - merged["SNR_true"]).abs().eq(magnitude)
    return (merged.loc[mask, "PSNR"] - merged.loc[mask, "PSNR_orig"]).mean()


def draw_mismatch_magnitude(ax: plt.Axes, data: dict[str, pd.DataFrame], labels: dict[str, str]) -> None:
    original = data["Original"]
    styles = {
        "D1": {"linestyle": ":", "marker": "^", "linewidth": 1.35, "color": COLORS["D1"]},
        "D3": {"linestyle": "-", "marker": "D", "linewidth": 1.9, "color": COLORS["D3"]},
        "D6": {"linestyle": "-.", "marker": "v", "linewidth": 1.35, "color": COLORS["D6"]},
        "RandomHat": {"linestyle": (0, (6, 3)), "marker": "x", "linewidth": 1.35, "color": COLORS["RandomHat"]},
    }
    for key in FIG4_METHODS:
        values = [magnitude_gain(data[key], original, mag) for mag in MAGNITUDES]
        ax.plot(MAGNITUDES, values, label=labels[key], markersize=4.2, **styles[key])

    ax.axhline(0, color="#555555", linewidth=0.75)
    ax.set_xticks(MAGNITUDES)
    ax.set_xlabel("SNR mismatch magnitude (dB)")
    ax.set_ylabel("PSNR gain over original (dB)")
    ax.grid(True, alpha=0.28, linewidth=0.6)
    ax.legend(fontsize=6.7, frameon=False)
    ax.set_title("(b) Robustness by mismatch magnitude", fontsize=9)


def plot_mismatch_magnitude(data: dict[str, pd.DataFrame], labels: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.35), constrained_layout=True)
    draw_mismatch_magnitude(ax, data, labels)
    save_line_figure(fig, "fig4_mismatch_magnitude")

    wrapper = "\n".join(
        [
            "% Auto-generated by latex/generate_figures.py",
            "\\begin{figure}[t]",
            "\\centering",
            "\\includegraphics[width=\\columnwidth]{figures/fig4_mismatch_magnitude}",
            "\\caption{Average PSNR gain over the original checkpoint at each absolute SNR mismatch magnitude. Line styles denote methods.}",
            "\\label{fig:legacy-mismatch-magnitude}",
            "\\end{figure}",
            "",
        ]
    )
    (FIG_DIR / "fig4_mismatch_magnitude.tex").write_text(wrapper, encoding="utf-8")


def plot_tradeoff_magnitude(primary: dict[str, dict[str, float]], data: dict[str, pd.DataFrame], labels: dict[str, str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.6), constrained_layout=True)
    draw_delta_tradeoff(axes[0], primary, labels)
    draw_mismatch_magnitude(axes[1], data, labels)
    save_line_figure(fig, "fig3_tradeoff_magnitude")

    wrapper = "\n".join(
        [
            "% Auto-generated by latex/generate_figures.py",
            "\\begin{figure*}[t]",
            "\\centering",
            "\\includegraphics[width=\\textwidth]{figures/fig3_tradeoff_magnitude}",
            "\\caption{Complementary views of the robustness--fidelity trade-off. The left panel uses line styles to distinguish evaluation metrics across methods. The right panel uses line styles to distinguish methods and groups PSNR gains by absolute SNR mismatch magnitude.}",
            "\\label{fig:tradeoff-magnitude}",
            "\\end{figure*}",
            "",
        ]
    )
    (FIG_DIR / "fig3_tradeoff_magnitude.tex").write_text(wrapper, encoding="utf-8")


def crop_grid(path: Path, rows: int = 3) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing qualitative grid image: {path}")
    image = Image.open(path).convert("RGB")
    row_height = max(1, round(image.height / 128))
    crop = image.crop((0, 0, image.width, row_height * rows))
    return np.asarray(crop)


def plot_qualitative_recon() -> None:
    vis_dir = ROOT / "mismatch_vis_cifar10"
    fig, axes = plt.subplots(2, len(QUAL_CASES), figsize=(7.15, 2.15))
    for col, (true_snr, hat_snr) in enumerate(QUAL_CASES):
        case_dir = vis_dir / f"true{true_snr}_hat{hat_snr}_C32"
        original = crop_grid(case_dir / "batch0_original_grid.png")
        recon = crop_grid(case_dir / "batch0_recon_grid.png")
        axes[0, col].imshow(original)
        axes[1, col].imshow(recon)
        axes[0, col].set_title(rf"$\gamma={true_snr}$ dB, $\hat{{\gamma}}={hat_snr}$ dB", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=0.03, wspace=0.08, hspace=0.22)
    fig.text(0.027, 0.64, "Input", rotation=90, va="center", ha="center", fontsize=8)
    fig.text(0.027, 0.24, "Recon.", rotation=90, va="center", ha="center", fontsize=8)
    save_line_figure(fig, "fig4_qualitative_recon")

    wrapper = "\n".join(
        [
            "% Auto-generated by latex/generate_figures.py",
            "\\begin{figure*}[t]",
            "\\centering",
            "\\includegraphics[width=\\textwidth]{figures/fig4_qualitative_recon}",
            "\\caption{Representative CIFAR10 input and reconstruction grids under selected SNR-mismatch cases. The figure illustrates the visual form of mismatch degradation; method ranking is based on the quantitative mismatch matrices.}",
            "\\label{fig:qualitative-recon}",
            "\\end{figure*}",
            "",
        ]
    )
    (FIG_DIR / "fig4_qualitative_recon.tex").write_text(wrapper, encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    primary_data, primary_labels, missing = load_primary()
    auxiliary_data, auxiliary_labels = load_auxiliary()
    rayleigh_data, rayleigh_labels = load_rayleigh()
    primary_summary = write_tradeoff_table(primary_data, primary_labels, missing)
    auxiliary_summary = write_auxiliary_table(auxiliary_data, auxiliary_labels, primary_summary["Original"])
    rayleigh_summary = write_rayleigh_table(rayleigh_data, rayleigh_labels)
    write_metrics(primary_summary, auxiliary_summary, rayleigh_summary)
    plot_psnr_heatmap(primary_data)
    plot_delta_tradeoff(primary_summary, primary_labels)
    plot_mismatch_magnitude(primary_data, primary_labels)
    plot_tradeoff_magnitude(primary_summary, primary_data, primary_labels)
    plot_qualitative_recon()
    print(f"Generated CL paper assets in {FIG_DIR}")


if __name__ == "__main__":
    main()
