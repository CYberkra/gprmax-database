"""Compare matched gprMax SFCW lines acquired 0.6 m and 15 m above the slope.

Run python analysis/compare_heights.py to analyse the two supplied result sets.
Each input directory contains merged.h5 and sfcw.h5 for its flight height.
By default all 33 stations are required. --partial compares only co-located stations.
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1]
X = np.arange(8, 41)
TONES = 10_000_000 + 300_000 * np.arange(534)
C = 299_792_458.0
EXTRA_AIR_NS = 2 * (15 - 0.6) / C * 1e9
# Gates describe time intervals; their peaks may belong to different arrivals.
GATES = {
    "early": ((0, 20), (0, 20)),
    "layer_window": ((175, 255), (175 + EXTRA_AIR_NS, 255 + EXTRA_AIR_NS)),
    "deep_window": ((520, 700), (520 + EXTRA_AIR_NS, 700 + EXTRA_AIR_NS)),
}


@dataclass
class Line:
    x: np.ndarray
    t_ns: np.ndarray
    amplitude: np.ndarray
    signature: tuple
    weights: np.ndarray
    tail_db: float
    bin_spacing_ns: float


def read_line(folder, height):
    with h5py.File(folder / "merged.h5") as merged:
        np.testing.assert_allclose(
            merged.attrs["dx_dy_dz"], [1 / 23, 0.04, 0.04], rtol=0, atol=1e-9
        )
        expected_nz = 1400 if height == 0.6 else 1760
        np.testing.assert_array_equal(
            merged.attrs["nx_ny_nz"], [1104, 500, expected_nz]
        )
        rx = merged["trace_metadata/rxs/rx1/Position"][:]
        tx = merged["trace_metadata/srcs/src1/Position"][:]
        x = np.rint(rx[:, 0]).astype(int)
        if not x.size or np.unique(x).size != x.size or not np.isin(x, X).all():
            raise ValueError(
                f"{folder}: station positions must be distinct members of x=8..40 m"
            )
        for positions, y in [(rx, 10.6), (tx, 9.4)]:
            expected = np.column_stack([x, np.full(x.size, y), 41.6 + 0.2 * x + height])
            np.testing.assert_allclose(
                positions,
                expected,
                atol=1e-5,
                rtol=0,
                err_msg=f"{folder}: unexpected geometry for {height:g} m",
            )
        ey = merged["rxs/rx1/Ey"]
        dt = float(ey.attrs["SampleInterval"])
        t0 = float(ey.attrs["TimeSampleOffset"])
        end_ns = 1e9 * (t0 + (ey.shape[0] - 1) * dt)
        if ey.ndim != 2 or ey.shape[1] != len(x) or abs(end_ns - 1600) > 1:
            raise ValueError(f"{folder}: unexpected waveform dimensions or FDTD window")
        raw = ey[:]
        if not np.isfinite(raw).all():
            raise ValueError(f"{folder}: nonfinite raw Ey")
    with h5py.File(folder / "sfcw.h5") as s:
        np.testing.assert_array_equal(s["frequency"][:], TONES)
        g = s["time_response"]
        if (
            s.attrs["Method"] != "direct"
            or s.attrs["SourcePath"] != "/srcs/src1"
            or not str(s.attrs["ReceiverPath"]).endswith("/rxs/rx1/Ey")
            or g.attrs["Window"] != "hann"
            or g.attrs["ZeroPadFactor"] != 4
            or g.attrs["TimeShift"] != 0
            or s.attrs["TailTaperFraction"] != 0
        ):
            raise ValueError(
                f"{folder}: incompatible source, receiver or reconstruction parameters"
            )
        freq_response = s["response"]
        if (
            freq_response.shape != (534, len(x))
            or not np.isfinite(freq_response[:]).all()
            or not s["source_valid"][:].all()
        ):
            raise ValueError(f"{folder}: missing or invalid frequency response")
        # Bind the spectral file to the actual merged trace order and dataset. Three
        # independently evaluated Fourier coefficients suffice to catch file swaps.
        bins = np.array([0, 233, 533])
        phase = np.exp(
            -2j * np.pi * TONES[bins, None] * (t0 + dt * np.arange(raw.shape[0]))
        )
        received = dt * (phase @ raw)
        np.testing.assert_allclose(
            received,
            s["receiver_spectrum"][bins],
            rtol=1e-7,
            atol=np.max(np.abs(received)) * 1e-10,
            err_msg=f"{folder}: receiver spectra do not match merged.h5",
        )
        clock = g["time"][:] * 1e9
        np.testing.assert_allclose(
            clock, np.arange(4 * 534) * 1e9 / (4 * 534 * 300_000), rtol=1e-13
        )
        stop = np.searchsorted(clock, end_ns, side="right")
        a = np.abs(g["complex_envelope"][:stop])
        if (
            a.shape != (stop, len(x))
            or not np.isfinite(a).all()
            or not np.all(a.max(axis=0) > 0)
        ):
            raise ValueError(f"{folder}: invalid envelope")
        signature = tuple(
            s.attrs[k]
            for k in (
                "SourceQuantity",
                "SourceUnits",
                "ReceiverQuantity",
                "ReceiverUnits",
                "EngineeringConvention",
                "SourceSpatialScale",
            )
        )
        if signature[:4] != ("electric_current", "A", "Ey", "V/m"):
            raise ValueError(
                f"{folder}: expected source current in A and receiver Ey in V/m"
            )
        return Line(
            x,
            clock[:stop],
            a,
            signature,
            g["weights"][:],
            float(s.attrs["ReceiverTailRelativeDB"]),
            float(clock[1] - clock[0]),
        )


def configure_fonts():
    # The same OFL-licensed subset keeps the Chinese labels portable.
    font = Path(__file__).with_name("fonts") / "SlopeLabels-Regular.ttf"
    font_manager.fontManager.addfont(str(font))
    mpl.rcParams.update(
        {
            "font.family": "Slope Labels",
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.65,
            "text.color": "#162e39",
            "axes.labelcolor": "#243b46",
        }
    )


def save_figure(file, t_ns, x, arrays, reference):
    configure_fonts()
    rows = [(0, 1600, -120, 0), (460, 850, -140, -95)]
    cmap = mpl.colormaps["magma"].copy()
    cmap.set_bad("#d4d9db")
    fig = plt.figure(figsize=(11.8, 9.2))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 0.034],
        hspace=0.43,
        wspace=0.24,
        top=0.85,
        bottom=0.11,
        left=0.072,
        right=0.92,
    )
    fig.text(0.072, 0.956, "飞行高度对比  ·  山坡下拱形洞室", size=20)
    status = (
        "完整测线，33 道"
        if x.size == 33
        else f"共同完成 {x.size}/33 道，灰列为未配对测点"
    )
    fig.text(
        0.072,
        0.912,
        f"{status}    /    10–169.9 MHz    /    1 m 道距    /    Hann 频窗",
        color="#355564",
    )
    dt = np.median(np.diff(t_ns))
    for r, (start, end, lo, hi) in enumerate(rows):
        use = (t_ns >= start) & (t_ns <= end)
        times = t_ns[use]
        for c, (a, h) in enumerate(zip(arrays, [0.6, 15])):
            ax = fig.add_subplot(gs[r, c])
            db = np.full((len(times), 33), np.nan)
            db[:, x - 8] = 20 * np.log10(
                np.maximum(a[use, :] / reference, np.finfo(float).tiny)
            )
            im = ax.imshow(
                np.ma.masked_invalid(db),
                origin="upper",
                interpolation="nearest",
                aspect="auto",
                extent=(7.5, 40.5, times[-1] + dt / 2, times[0] - dt / 2),
                cmap=cmap,
                vmin=lo,
                vmax=hi,
            )
            ax.set(
                xlim=(7.5, 40.5),
                ylim=(end, start),
                xticks=np.arange(8, 41, 4),
                xlabel="测线水平位置 x（m）",
                ylabel="双程走时（ns）" if c == 0 else "",
            )
            ax.set_title(
                f"离地 {h:g} m   ·   {start}–{end} ns",
                loc="left",
                fontsize=12,
            )
        color = fig.colorbar(im, cax=fig.add_subplot(gs[r, 2]), extend="both")
        color.set_label("包络幅值（dB）")
    fig.text(
        0.072,
        0.035,
        f"共同幅值参考为 {reference:.5g} (V/m)/A。每行采用相同色阶，深部图仅放大显示范围。",
        color="#3a5561",
    )
    fig.savefig(file, dpi=180)
    plt.close(fig)


def window_metrics(x, t, arrays, reference):
    rows = []
    for name, intervals in GATES.items():
        values = []
        for a, (start, end) in zip(arrays, intervals):
            mask = (t >= start) & (t <= end)
            selected = a[mask]
            index = np.argmax(selected, axis=0)
            peak = selected[index, np.arange(x.size)]
            if np.any(peak <= 0):
                raise ValueError(f"{name}: zero peak prevents a finite dB comparison")
            rms = np.sqrt(np.mean(selected**2, axis=0))
            values.append((t[mask][index], 20 * np.log10(peak / reference), rms))
        (t0, d0, r0), (t1, d1, r1) = values
        for j, loc in enumerate(x):
            rows.append(
                dict(
                    gate=name,
                    station=int(loc - 7),
                    x_m=int(loc),
                    low_start_ns=intervals[0][0],
                    low_stop_ns=intervals[0][1],
                    high_start_ns=intervals[1][0],
                    high_stop_ns=intervals[1][1],
                    low_peak_ns=t0[j],
                    high_peak_ns=t1[j],
                    high_minus_low_ns=t1[j] - t0[j],
                    low_peak_db=d0[j],
                    high_peak_db=d1[j],
                    high_minus_low_peak_db=d1[j] - d0[j],
                    high_minus_low_rms_db=20 * np.log10(r1[j] / r0[j]),
                )
            )
    return rows


def summarize(rows):
    result = []
    for gate in GATES:
        for name, lo, hi in [
            ("center", 24, 24),
            ("opening_projection", 18, 30),
            ("line", 8, 40),
        ]:
            group = [r for r in rows if r["gate"] == gate and lo <= r["x_m"] <= hi]
            if not group:
                continue
            item = dict(
                gate=gate,
                region=name,
                x_start_m=lo,
                x_stop_m=hi,
                paired_count=len(group),
            )
            for key in [
                "low_peak_ns",
                "high_peak_ns",
                "high_minus_low_ns",
                "low_peak_db",
                "high_peak_db",
                "high_minus_low_peak_db",
                "high_minus_low_rms_db",
            ]:
                item[f"median_{key}"] = np.median([r[key] for r in group])
            result.append(item)
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--low",
        type=Path,
        default=MODEL_DIR / "data" / "0.6m" / "results",
        help="Low-altitude results (default: data/0.6m/results)",
    )
    parser.add_argument(
        "--high",
        type=Path,
        default=MODEL_DIR / "data" / "15m" / "results",
        help="High-altitude results (default: data/15m/results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_DIR / "runs" / "comparison",
        help="Output directory (default: runs/comparison)",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Compare the intersection of available stations",
    )
    args = parser.parse_args()
    low, high = read_line(args.low, 0.6), read_line(args.high, 15)
    if low.signature != high.signature:
        raise ValueError(
            f"Source or receiver definitions differ: {low.signature} / {high.signature}"
        )
    # Hann weights generated by different math libraries can differ by a few ULPs.
    np.testing.assert_allclose(low.weights, high.weights, rtol=0, atol=1e-14)
    ntime = min(low.t_ns.size, high.t_ns.size)
    t = low.t_ns[:ntime]
    np.testing.assert_allclose(t, high.t_ns[:ntime], rtol=0, atol=1e-10)
    x, idx_low, idx_high = np.intersect1d(low.x, high.x, return_indices=True)
    if not len(x) or not args.partial and x.size != 33:
        raise ValueError(
            f"Matched stations {len(x)}/33. --partial is required for paired previews."
        )
    arrays = [low.amplitude[:ntime, idx_low], high.amplitude[:ntime, idx_high]]
    reference = max(a.max() for a in arrays)
    rows = window_metrics(x, t, arrays, reference)
    suffix = "" if len(x) == 33 else f"_partial_{len(x)}of33"
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        low_height_m=0.6,
        high_height_m=15.0,
        paired_station_ids=(x - 7).tolist(),
        reference_peak=float(reference),
        source_units=low.signature[1],
        receiver_units=low.signature[3],
        fft_time_sample_ns=low.bin_spacing_ns,
        nominal_time_resolution_ns=float(1e9 / (TONES[-1] - TONES[0])),
        air_roundtrip_gate_shift_ns=EXTRA_AIR_NS,
        low_receiver_tail_db=low.tail_db,
        high_receiver_tail_db=high.tail_db,
        method="Peak time is the time of the strongest measured sample in each gate. "
        "Regional values are medians across matched stations. Each difference is high minus low. "
        "All amplitudes use a single reference over both lines, paired traces and the common record.",
    )
    with tempfile.TemporaryDirectory(prefix=".compare-", dir=args.output) as directory:
        tmp = Path(directory)
        save_figure(tmp / f"bscan_heights{suffix}.png", t, x, arrays, reference)
        write_csv(tmp / f"station_metrics{suffix}.csv", rows)
        write_csv(tmp / f"region_metrics{suffix}.csv", summarize(rows))
        (tmp / f"comparison{suffix}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for path in tmp.iterdir():
            path.replace(args.output / path.name)
    print(f"Compared {x.size}/33 stations. Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
