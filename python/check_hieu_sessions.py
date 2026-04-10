from pathlib import Path


SAMPLE_RATE_HZ = 50
WINDOW_SIZE_SAMPLES = 200
EXPECTED_PERSONS = ["An", "Hao", "Hieu", "Kien", "Quan", "Tien"]


def count_sessions(label_dir: Path) -> int:
    """Count sessions in one label folder (Normal/Fall).

    A session is counted when a directory either:
    - directly contains accel.csv or gyro.csv, or
    - has child directories that contain accel.csv or gyro.csv.
    """
    if not label_dir.exists() or not label_dir.is_dir():
        return 0

    total = 0
    for activity_dir in sorted(label_dir.iterdir()):
        if not activity_dir.is_dir():
            continue

        has_direct_sensor_file = (
            (activity_dir / "accel.csv").exists()
            or (activity_dir / "gyro.csv").exists()
        )
        if has_direct_sensor_file:
            total += 1
            continue

        for session_dir in sorted(activity_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            if (session_dir / "accel.csv").exists() or (session_dir / "gyro.csv").exists():
                total += 1

    return total


def count_csv_samples(csv_path: Path) -> int:
    """Count data rows in a csv file, excluding header."""
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def count_windows(label_dir: Path, sensor_file: str) -> tuple[int, int, int]:
    """Return (n_files, total_samples, total_windows) for one sensor type.

    A non-overlapping window has WINDOW_SIZE_SAMPLES samples.
    """
    if not label_dir.exists() or not label_dir.is_dir():
        return 0, 0, 0

    n_files = 0
    total_samples = 0
    total_windows = 0

    for csv_path in label_dir.rglob(sensor_file):
        n_files += 1
        n_samples = count_csv_samples(csv_path)
        total_samples += n_samples
        total_windows += n_samples // WINDOW_SIZE_SAMPLES

    return n_files, total_samples, total_windows


def find_activity_dir(label_dir: Path, activity_name: str) -> Path | None:
    """Find activity directory by case-insensitive name match."""
    if not label_dir.exists() or not label_dir.is_dir():
        return None

    target = activity_name.lower()
    for activity_dir in label_dir.iterdir():
        if activity_dir.is_dir() and activity_dir.name.lower() == target:
            return activity_dir
    return None


def count_activity_windows(person_dir: Path, label: str, activity_name: str) -> dict:
    """Count sessions/samples/windows for one person's specific activity folder."""
    label_dir = person_dir / label
    activity_dir = find_activity_dir(label_dir, activity_name)
    if activity_dir is None:
        return {
            "sessions": 0,
            "accel_files": 0,
            "gyro_files": 0,
            "accel_samples": 0,
            "gyro_samples": 0,
            "accel_windows": 0,
            "gyro_windows": 0,
        }

    accel_files, accel_samples, accel_windows = count_windows(activity_dir, "accel.csv")
    gyro_files, gyro_samples, gyro_windows = count_windows(activity_dir, "gyro.csv")

    # Each accel.csv (or direct file folder) corresponds to one captured session in current structure.
    return {
        "sessions": accel_files,
        "accel_files": accel_files,
        "gyro_files": gyro_files,
        "accel_samples": accel_samples,
        "gyro_samples": gyro_samples,
        "accel_windows": accel_windows,
        "gyro_windows": gyro_windows,
    }


def gather_person_stats(person_dir: Path, person: str) -> dict:
    normal_count = count_sessions(person_dir / "Normal")
    fall_count = count_sessions(person_dir / "Fall")

    normal_accel = count_windows(person_dir / "Normal", "accel.csv")
    fall_accel = count_windows(person_dir / "Fall", "accel.csv")
    normal_gyro = count_windows(person_dir / "Normal", "gyro.csv")
    fall_gyro = count_windows(person_dir / "Fall", "gyro.csv")

    return {
        "person": person,
        "normal_sessions": normal_count,
        "fall_sessions": fall_count,
        "normal_accel_windows": normal_accel[2],
        "fall_accel_windows": fall_accel[2],
        "normal_gyro_windows": normal_gyro[2],
        "fall_gyro_windows": fall_gyro[2],
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "Person",
        "Normal Sess",
        "Fall Sess",
        "Normal Win(acc)",
        "Fall Win(acc)",
        "Normal Win(gyr)",
        "Fall Win(gyr)",
    ]
    keys = [
        "person",
        "normal_sessions",
        "fall_sessions",
        "normal_accel_windows",
        "fall_accel_windows",
        "normal_gyro_windows",
        "fall_gyro_windows",
    ]

    widths = []
    for h, k in zip(headers, keys):
        max_val = max((len(str(r[k])) for r in rows), default=0)
        widths.append(max(len(h), max_val))

    def fmt_line(values: list[str]) -> str:
        return " | ".join(v.ljust(w) for v, w in zip(values, widths))

    print(fmt_line(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt_line([str(r[k]) for k in keys]))


def main() -> None:
    data_root = Path(__file__).resolve().parents[1] / "server" / "data"
    available = [p for p in EXPECTED_PERSONS if (data_root / p).is_dir()]

    rows = [gather_person_stats(data_root / p, p) for p in available]

    print(f"Sampling rate: {SAMPLE_RATE_HZ} Hz")
    print(f"Window size: {WINDOW_SIZE_SAMPLES} samples ({WINDOW_SIZE_SAMPLES / SAMPLE_RATE_HZ:.1f} s)")
    print()
    print(f"Subjects found: {', '.join(available)}")
    print()
    print_table(rows)

    total_normal_sessions = sum(r["normal_sessions"] for r in rows)
    total_fall_sessions = sum(r["fall_sessions"] for r in rows)
    total_normal_windows = sum(r["normal_accel_windows"] for r in rows)
    total_fall_windows = sum(r["fall_accel_windows"] for r in rows)

    print()
    print("Totals (accel windows):")
    print(f"  Normal sessions: {total_normal_sessions} | Fall sessions: {total_fall_sessions}")
    print(f"  Normal windows : {total_normal_windows} | Fall windows : {total_fall_windows}")

    # Focused check requested: Hieu, LonXon folder only.
    person = "Hieu"
    activity = "LonXon"
    person_dir = data_root / person
    if person_dir.is_dir():
        lonxon_normal = count_activity_windows(person_dir, "Normal", activity)
        lonxon_fall = count_activity_windows(person_dir, "Fall", activity)

        print()
        print(f"{person} | activity folder: {activity}")
        print(
            f"  Normal: sessions={lonxon_normal['sessions']} | "
            f"accel_windows={lonxon_normal['accel_windows']} | gyro_windows={lonxon_normal['gyro_windows']}"
        )
        print(
            f"  Fall  : sessions={lonxon_fall['sessions']} | "
            f"accel_windows={lonxon_fall['accel_windows']} | gyro_windows={lonxon_fall['gyro_windows']}"
        )


if __name__ == "__main__":
    main()
