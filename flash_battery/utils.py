from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.manifold import TSNE

from .distributed_mode import is_main_process


def compute_dqdv(voltage, capacity, window_length: int = 31, polyorder: int = 2):
    if len(voltage) > window_length:
        v_smooth = savgol_filter(voltage, window_length, polyorder)
        q_smooth = savgol_filter(capacity, window_length, polyorder)
    else:
        v_smooth = voltage
        q_smooth = capacity
    dv = np.gradient(v_smooth)
    dq = np.gradient(q_smooth)
    dv[np.abs(dv) < 1e-6] = 1e-6
    dqdv = dq / dv
    if len(dqdv) > window_length:
        dqdv = savgol_filter(dqdv, window_length, polyorder)
    return v_smooth, dqdv


def safe_pearsonr(x, y):
    if len(x) < 2:
        return 0.0
    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return 0.0
    return pearsonr(x, y)[0]


def analyze_and_plot_results(run_dir, epoch, all_results):
    if is_main_process():
        print(f"Start analyzing {len(all_results)} samples...")

    save_dir = Path(run_dir) / f"epoch{epoch + 1}"
    save_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in all_results:
        r_i, r_u, r_qc = sample["real_i"], sample["real_u"], sample["real_qc"]
        g_i, g_u, g_qc = sample["gen_i"], sample["gen_u"], sample["gen_qc"]
        soh = sample["soh"]
        raw_protocol = sample["raw_protocol"]
        rmse_i = np.sqrt(np.mean((r_i - g_i) ** 2))
        rmse_u = np.sqrt(np.mean((r_u - g_u) ** 2))
        rmse_qc = np.sqrt(np.mean((r_qc - g_qc) ** 2))
        corr_i = safe_pearsonr(r_i, g_i)
        corr_u = safe_pearsonr(r_u, g_u)
        corr_qc = safe_pearsonr(r_qc, g_qc)
        try:
            _, r_dqdv = compute_dqdv(r_u, r_qc)
            _, g_dqdv = compute_dqdv(g_u, g_qc)
            crop = min(10, len(r_dqdv) // 5)
            if len(r_dqdv) > 2 * crop:
                rmse_dqdv = np.sqrt(
                    np.mean((r_dqdv[crop:-crop] - g_dqdv[crop:-crop]) ** 2)
                )
                rmse_dqdv = min(rmse_dqdv, 10.0)
            else:
                rmse_dqdv = 0.0
        except Exception:
            rmse_dqdv = 0.0

        records.append(
            {
                "soh": soh,
                "protocol_key": raw_protocol,
                "rmse_i": rmse_i,
                "rmse_u": rmse_u,
                "rmse_qc": rmse_qc,
                "corr_i": corr_i,
                "corr_u": corr_u,
                "corr_qc": corr_qc,
                "rmse_dqdv": rmse_dqdv,
            }
        )

    def plot_metric_vs_soh(metrics_keys, titles, ylabel, save_name):
        sohs = [record["soh"] for record in records]
        colors = ["black", "blue", "green"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for idx, metric in enumerate(metrics_keys):
            values = [record[metric] for record in records]
            ax = axes[idx]
            ax.scatter(sohs, values, alpha=0.6, s=15, c=colors[idx], edgecolors="none")
            if len(sohs) > 1:
                coeff = np.polyfit(sohs, values, 1)
                ax.plot(sohs, np.poly1d(coeff)(sohs), "k--", alpha=0.5, linewidth=1)
            ax.set_xlabel("SOH")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{titles[idx]} vs SOH")
            ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / save_name, dpi=150)
        plt.close()

    def plot_metric_vs_protocol(metrics_keys, titles, ylabel, save_name):
        groups = {}
        for record in records:
            key = record["protocol_key"]
            groups.setdefault(key, {metric: [] for metric in metrics_keys})
            for metric in metrics_keys:
                groups[key][metric].append(record[metric])
        protocols = sorted(groups.keys())
        colors = ["black", "blue", "green"]
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        for idx, metric in enumerate(metrics_keys):
            means = [np.mean(groups[key][metric]) for key in protocols]
            stds = [np.std(groups[key][metric]) for key in protocols]
            axes[idx].bar(range(len(protocols)), means, yerr=stds, capsize=5, color=colors[idx], alpha=0.7)
            axes[idx].set_ylabel(titles[idx])
            axes[idx].set_xticks(range(len(protocols)))
            axes[idx].set_xticklabels(protocols, rotation=45, ha="right", fontsize=9)
            axes[idx].grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / save_name, dpi=150)
        plt.close()

    plot_metric_vs_soh(
        ["rmse_i", "rmse_u", "rmse_qc"],
        ["Current RMSE", "Voltage RMSE", "Capacity RMSE"],
        "RMSE",
        "analysis_rmse_vs_soh.png",
    )
    plot_metric_vs_protocol(
        ["rmse_i", "rmse_u", "rmse_qc"],
        ["Current RMSE", "Voltage RMSE", "Capacity RMSE"],
        "RMSE",
        "analysis_rmse_vs_protocol.png",
    )
    plot_metric_vs_soh(
        ["corr_i", "corr_u", "corr_qc"],
        ["Current PCC", "Voltage PCC", "Capacity PCC"],
        "Pearson Corr",
        "analysis_pcc_vs_soh.png",
    )
    plot_metric_vs_protocol(
        ["corr_i", "corr_u", "corr_qc"],
        ["Current PCC", "Voltage PCC", "Capacity PCC"],
        "Pearson Corr",
        "analysis_pcc_vs_protocol.png",
    )

    set_wd_records = {}
    for sample in all_results:
        key = sample["raw_protocol"]
        set_wd_records.setdefault(key, {"real_u": [], "gen_u": []})
        set_wd_records[key]["real_u"].extend(sample["real_u"][::5])
        set_wd_records[key]["gen_u"].extend(sample["gen_u"][::5])
    protocols = sorted(set_wd_records.keys())
    wd_values = [
        wasserstein_distance(
            set_wd_records[key]["real_u"], set_wd_records[key]["gen_u"]
        )
        for key in protocols
    ]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(range(len(protocols)), wd_values, color="purple", alpha=0.7)
    ax.set_ylabel("Set-wise Voltage Wasserstein Distance")
    ax.set_xticks(range(len(protocols)))
    ax.set_xticklabels(protocols, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "analysis_setwise_wd_protocol.png", dpi=150)
    plt.close()

    tsne_features = []
    tsne_labels = []
    for sample in all_results:
        label = sample["label"]
        if not all(
            key in sample
            for key in [
                "real_i_padded",
                "real_u_padded",
                "real_qc_padded",
                "gen_i_padded",
                "gen_u_padded",
                "gen_qc_padded",
            ]
        ):
            continue

        def combine_features(i_arr, u_arr, qc_arr):
            return np.concatenate([i_arr, u_arr, qc_arr])

        tsne_features.append(
            combine_features(
                sample["real_i_padded"], sample["real_u_padded"], sample["real_qc_padded"]
            )
        )
        tsne_labels.append(0 if label == 0 else 1)
        tsne_features.append(
            combine_features(
                sample["gen_i_padded"], sample["gen_u_padded"], sample["gen_qc_padded"]
            )
        )
        tsne_labels.append(2 if label == 0 else 3)
        if sample.get("gen_u_padded_scale2") is not None:
            tsne_features.append(
                combine_features(
                    sample["gen_i_padded_scale2"],
                    sample["gen_u_padded_scale2"],
                    sample["gen_qc_padded_scale2"],
                )
            )
            tsne_labels.append(4 if label == 0 else 5)
        if sample.get("gen_u_padded_scale3") is not None:
            tsne_features.append(
                combine_features(
                    sample["gen_i_padded_scale3"],
                    sample["gen_u_padded_scale3"],
                    sample["gen_qc_padded_scale3"],
                )
            )
            tsne_labels.append(6 if label == 0 else 7)

    tsne_features = np.array(tsne_features)
    tsne_labels = np.array(tsne_labels)
    if len(tsne_features) > 5000:
        idx = np.random.choice(len(tsne_features), 5000, replace=False)
        tsne_features = tsne_features[idx]
        tsne_labels = tsne_labels[idx]
    if len(tsne_features) > 5:
        tsne_model = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(50, len(tsne_features) - 1),
        )
        features_2d = tsne_model.fit_transform(tsne_features)
        plt.figure(figsize=(14, 11))
        names = [
            "Real Healthy",
            "Real Faulty",
            "Gen Healthy (CFG=1)",
            "Gen Faulty (CFG=1)",
            "Gen Healthy (CFG=4)",
            "Gen Faulty (CFG=4)",
            "Gen Healthy (CFG=8)",
            "Gen Faulty (CFG=8)",
        ]
        colors = [
            "#008080",
            "#800080",
            "#87CEFA",
            "#FFA07A",
            "#1E90FF",
            "#FF4500",
            "#00008B",
            "#8B0000",
        ]
        markers = ["o", "^", "o", "^", "o", "^", "o", "^"]
        for idx in range(8):
            mask = tsne_labels == idx
            if np.any(mask):
                plt.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=colors[idx],
                    label=names[idx],
                    marker=markers[idx],
                    alpha=0.7,
                    s=40 if idx < 2 else 35,
                    edgecolors="w",
                    linewidths=0.5,
                )
        plt.legend(fontsize=10, loc="best", ncol=2)
        plt.title("t-SNE: Real vs Generated Joint I-V-Qc Distribution")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "analysis_tsne_distribution_multimodal.png", dpi=150)
        plt.close()

    protocol_groups = {}
    for sample in all_results:
        protocol_groups.setdefault(sample["raw_protocol"], []).append(sample)
    proto_save_dir = save_dir / "by_protocol"
    proto_save_dir.mkdir(parents=True, exist_ok=True)
    for protocol, samples in sorted(protocol_groups.items()):
        safe_name = (
            str(protocol)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "pct")
        )
        plot_trajectories(
            data_i=[sample["real_i"] for sample in samples],
            data_u=[sample["real_u"] for sample in samples],
            data_i_gen=[sample.get("gen_i_dict", sample["gen_i"]) for sample in samples],
            data_u_gen=[sample.get("gen_u_dict", sample["gen_u"]) for sample in samples],
            data_qc=[sample["real_qc"] for sample in samples],
            data_qc_gen=[sample["gen_qc"] for sample in samples],
            data_soh=[sample["soh"] for sample in samples],
            labels=[sample["label"] for sample in samples],
            save_path=proto_save_dir / f"protocol_{safe_name}.png",
        )


def plot_trajectories(
    data_i,
    data_u,
    data_i_gen=None,
    data_u_gen=None,
    data_qc=None,
    data_qc_gen=None,
    data_soh=None,
    labels=None,
    save_path="packIU.png",
):
    pages = (len(data_i) + 36 - 1) // 36
    for page in range(pages):
        start_idx = page * 36
        end_idx = min((page + 1) * 36, len(data_i))
        _plot_trajectories_single(
            data_i[start_idx:end_idx],
            data_u[start_idx:end_idx],
            data_i_gen[start_idx:end_idx] if data_i_gen is not None else None,
            data_u_gen[start_idx:end_idx] if data_u_gen is not None else None,
            data_qc[start_idx:end_idx] if data_qc is not None else None,
            data_qc_gen[start_idx:end_idx] if data_qc_gen is not None else None,
            data_soh[start_idx:end_idx] if data_soh is not None else None,
            labels[start_idx:end_idx] if labels is not None else None,
            str(save_path).replace(".png", f"_{page}.png"),
        )


def _plot_trajectories_single(
    data_i,
    data_u,
    data_i_gen=None,
    data_u_gen=None,
    data_qc=None,
    data_qc_gen=None,
    data_soh=None,
    labels=None,
    save_path="packIU.png",
):
    fig, axes = plt.subplots(6, 6, figsize=(18, 18))
    axes_flat = axes.flatten()
    legend_handles = []
    legend_labels = []
    scale_configs = {
        1.0: {"color": "#9C1DD5", "ls": "--"},
        4.0: {"color": "#C62828", "ls": "--"},
        8.0: {"color": "#5D4037", "ls": "--"},
    }

    for idx, ax in enumerate(axes_flat):
        if idx >= len(data_i):
            ax.axis("off")
            continue

        i_data = np.asarray(data_i[idx])
        u_data = np.asarray(data_u[idx])
        qc_data = np.asarray(data_qc[idx]) if data_qc is not None else np.zeros_like(i_data)
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        line_i = ax.plot(i_data, color="#333333", linewidth=1, alpha=0.6)
        line_u = ax2.plot(u_data, color="blue", linewidth=1, alpha=0.6)
        line_qc = ax3.plot(qc_data, color="#A5D6A7", linewidth=1, alpha=0.6)

        gen_handles = []
        gen_labels = []
        if data_i_gen is not None and data_u_gen is not None and data_qc_gen is not None:
            i_data_gen_item = data_i_gen[idx]
            u_data_gen_item = data_u_gen[idx]
            qc_data_gen = np.asarray(data_qc_gen[idx])
            l_qg = ax3.plot(qc_data_gen, color="#2CA02C", linestyle="--", alpha=0.5)
            if idx == 0:
                gen_handles.append(l_qg[0])
                gen_labels.append("Gen Capacity")
            if isinstance(i_data_gen_item, dict):
                for scale in sorted(i_data_gen_item.keys()):
                    cfg = scale_configs.get(scale, {"color": "black", "ls": "-"})
                    l_ig = ax.plot(
                        i_data_gen_item[scale],
                        color=cfg["color"],
                        linestyle=cfg["ls"],
                        linewidth=1.2,
                    )
                    if idx == 0 and scale == sorted(i_data_gen_item.keys())[0]:
                        gen_handles.append(l_ig[0])
                        gen_labels.append("Gen Current")
            else:
                l_ig = ax.plot(i_data_gen_item, color="orange", linestyle="--", linewidth=1.2)
                if idx == 0:
                    gen_handles.append(l_ig[0])
                    gen_labels.append("Gen Current")

            if isinstance(u_data_gen_item, dict):
                for scale in sorted(u_data_gen_item.keys()):
                    cfg = scale_configs.get(scale, {"color": "black", "ls": "-"})
                    l_ug = ax2.plot(
                        u_data_gen_item[scale],
                        color=cfg["color"],
                        linestyle=cfg["ls"],
                        linewidth=1.2,
                    )
                    if idx == 0 and scale == sorted(u_data_gen_item.keys())[0]:
                        gen_handles.append(l_ug[0])
                        gen_labels.append("Gen Voltage")
            else:
                l_ug = ax2.plot(u_data_gen_item, color="purple", linestyle="--", linewidth=1.2)
                if idx == 0:
                    gen_handles.append(l_ug[0])
                    gen_labels.append("Gen Voltage")

        if idx == 0:
            legend_handles = [line_i[0], line_u[0], line_qc[0]] + gen_handles
            legend_labels = ["Real Current", "Real Voltage", "Real Capacity"] + gen_labels

        label_text = "Healthy"
        if labels is not None and labels[idx] != 0:
            label_text = "Faulty"
        soh_text = f"{data_soh[idx]:.2f}" if data_soh is not None else "N/A"
        ax.set_xlabel(f"Step | SOH {soh_text} | {label_text}", fontsize=7)
        if idx % 6 == 0:
            ax.set_ylabel("Current", fontsize=10)
        if (idx + 1) % 6 == 0:
            ax2.set_ylabel("Voltage", fontsize=10)
            ax3.set_ylabel("Capacity", fontsize=10)
        ax2.yaxis.set_label_coords(1.15, 0.7)
        ax3.yaxis.set_label_coords(1.15, 0.3)
        ax.tick_params(axis="both", labelsize=6)
        if (idx + 1) % 6 != 0:
            ax2.yaxis.set_visible(False)
            ax3.yaxis.set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.3)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_handles),
            fontsize=12,
            frameon=True,
        )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
