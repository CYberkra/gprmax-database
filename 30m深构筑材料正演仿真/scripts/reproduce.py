"""Run one survey station or merge all 33 stations with gprMax 4.0.1."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
from gprMax.toolboxes.Utilities.outputfiles_merge import merge_files


CASE = Path(__file__).resolve().parents[1]


def grid_from_deck(deck):
    commands = dict(line.split(": ", 1) for line in deck.read_text(encoding="utf-8").splitlines()
                    if line.startswith(("#domain: ", "#dx_dy_dz: ", "#time_window: ")))
    spacing = np.fromstring(commands["#dx_dy_dz"], sep=" ")
    cells = np.rint(np.fromstring(commands["#domain"], sep=" ") / spacing).astype(int)
    return spacing, cells, float(commands["#time_window"])


def read_reference(path, positions, grid):
    """Check the case and station, and return the actual source reference."""
    with h5py.File(path, "r") as record:
        spacing, cells, duration = grid
        if (not np.array_equal(record.attrs["nx_ny_nz"], cells)
                or not np.allclose(record.attrs["dx_dy_dz"], spacing, rtol=1e-7, atol=1e-10)):
            raise ValueError(f"{path}: computational grid differs from the input model")
        dt = float(record.attrs["dt"])
        steps = int(record.attrs["Iterations"])
        if not np.isfinite(dt) or dt <= 0 or steps != int(np.ceil(duration / dt)) + 1:
            raise ValueError(f"{path}: invalid sample interval or incomplete time window")
        if set(record["srcs"]) != {"src1"} or set(record["rxs"]) != {"rx1"}:
            raise ValueError(f"{path}: expected a single source and receiver")
        src = record["srcs/src1"]
        rx = record["rxs/rx1"]
        for group, expected in zip((src, rx), positions):
            if not np.allclose(group.attrs["Position"], expected, rtol=0, atol=1e-5):
                raise ValueError(f"{path}: {group.name} is not at {expected} m")
        excitation = src["excitation"]
        samples = excitation["samples"][:]
        ey = rx["Ey"]
        if (src.attrs["Polarisation"] != "y" or excitation.attrs["WaveformType"] != "impulse"
                or samples.shape != (steps,) or samples.shape != ey.shape or not np.isfinite(samples[0])
                or samples[0] == 0 or np.any(samples[1:])):
            raise ValueError(f"{path}: expected a y-polarised impulse and matching Ey record")
        values = ey[:]
        if not np.isfinite(values).all() or not np.any(values):
            raise ValueError(f"{path}: nonfinite or empty Ey response")
        clock = tuple(obj.attrs[key] for obj in (excitation, ey) for key in ("SampleInterval", "TimeSampleOffset"))
        if not np.allclose(clock, (dt, dt / 2, dt, 0), rtol=1e-10, atol=dt * 1e-10):
            raise ValueError(f"{path}: source/receiver sampling clocks differ from the Yee time lattice")
        clock += (excitation.attrs["SpatialScale"],)
    return samples, clock


def run_station(deck, target, station, positions, grid, gpu):
    if target.is_file():
        read_reference(target, positions, grid)
        print(f"Reusing {target}")
        return
    with tempfile.TemporaryDirectory(prefix=f".{target.stem}-", dir=target.parent) as temporary:
        output = Path(temporary) / "response.h5"
        subprocess.run([
            sys.executable, "-m", "gprMax", str(deck), "-i", str(station), "-n", "1",
            "-gpu", str(gpu), "-gpu_precision", "single", "-o", str(output),
        ], check=True)
        read_reference(output, positions, grid)
        os.replace(output, target)
    print(f"Saved {target}")


def verify_existing(candidate, published):
    with h5py.File(candidate, "r") as fresh, h5py.File(published, "r") as existing:
        def compare(key, item):
            saved = existing[key]
            for attr, value in item.attrs.items():
                # Source and receiver file paths depend on the working directory.
                if attr not in ("SourceFile", "ReceiverFile") and not np.array_equal(value, saved.attrs[attr]):
                    raise ValueError(f"{published}: attribute {key}/{attr} differs from the new calculation")
            if isinstance(item, h5py.Dataset):
                a, b = item[...], saved[...]
                same = np.array_equal(a, b)
                if not same and a.shape == b.shape and a.dtype.kind in "fc" and fresh.attrs.get("Method") == "direct":
                    # The CZT/FFT backend can differ by round-off between machines.
                    same = np.allclose(a, b, rtol=1e-12, atol=1e-13 * np.max(np.abs(a)))
                if not same:
                    raise ValueError(f"{published}: {key} differs; choose a separate --output-root")

        compare("/", fresh)
        fresh.visititems(compare)


def process_line(work, files, positions, grid):
    reference_samples = reference_clock = None
    for path, expected in zip(files, positions):
        samples, clock = read_reference(path, expected, grid)
        if reference_samples is None:
            reference_samples, reference_clock = samples, clock
        elif clock != reference_clock or not np.array_equal(samples, reference_samples):
            raise ValueError(f"{path}: source reference differs from the first station")

    frequencies = np.loadtxt(CASE / "models/frequencies.csv", delimiter=",", skiprows=1, usecols=1, dtype=int)
    results = work / "results"
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=work) as temporary:
        staging = Path(temporary)
        merge_files(files, merged_outputfile=staging / "merged.h5")
        subprocess.run([
            sys.executable, "-m", "gprMax.toolboxes.SFCW", "process", "merged.h5",
            "--source-file", str(Path("..") / "raw" / files[0].name), "--source", "/srcs/src1",
            "--receiver", "/rxs/rx1", "--component", "Ey", "--method", "direct",
            "--f-start", str(frequencies[0]), "--f-stop", str(frequencies[-1]), "--steps", str(len(frequencies)),
            "--window", "hann", "--zero-pad", "4", "--output", "sfcw.h5",
        ], cwd=staging, check=True)
        with h5py.File(staging / "sfcw.h5", "r") as output:
            if (not np.array_equal(frequencies, output["frequency"][:])
                    or output["response"].shape != (len(frequencies), len(files))):
                raise ValueError("Unexpected SFCW frequency axis or response dimensions")
        if results.exists():
            # Existing results are immutable. Verify both files against the fresh
            # calculation before reusing them, including all per-station metadata.
            for name in ("merged.h5", "sfcw.h5"):
                verify_existing(staging / name, results / name)
            print(f"Verified existing {results}")
            return
        staging.rename(results)
    print(f"Saved {results / 'merged.h5'} and {results / 'sfcw.h5'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=CASE / "runs", help="parent of the height-specific runs")
    commands = parser.add_subparsers(dest="command", required=True)
    station = commands.add_parser("station", help="simulate one station on CUDA")
    station.add_argument("height", choices=["0.6m", "15m"])
    station.add_argument("number", type=int, help="station number (1–33)")
    station.add_argument("--gpu", type=int, default=0, help="visible CUDA device index (default: 0)")
    merge = commands.add_parser("merge", help="combine the completed stations and compute SFCW on the CPU")
    merge.add_argument("height", choices=["0.6m", "15m"])
    args = parser.parse_args()
    model = CASE / "models" / args.height
    deck = model / "sweep.in"
    grid = grid_from_deck(deck)
    table = np.loadtxt(model / "stations.csv", delimiter=",", skiprows=1)
    positions = table[:, 1:].reshape(-1, 2, 3)
    work = args.output_root.resolve() / args.height
    files = [work / "raw" / f"trace_{int(number):03d}.h5" for number in table[:, 0]]
    if args.command == "station":
        if not 1 <= args.number <= len(files):
            parser.error(f"station must be between 1 and {len(files)}")
        (work / "raw").mkdir(parents=True, exist_ok=True)
        i = args.number - 1
        run_station(deck, files[i], args.number, positions[i], grid, args.gpu)
    else:
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            parser.error("missing station files: " + ", ".join(missing))
        process_line(work, files, positions, grid)


if __name__ == "__main__":
    main()
