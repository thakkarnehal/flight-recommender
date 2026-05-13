"""Download BTS On-Time Performance data for 2024-2025 from transtats.bts.gov."""

import os
import zipfile
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)


def download_month(year: int, month: int) -> Path | None:
    url = BASE_URL.format(year=year, month=month)
    zip_path = RAW_DIR / f"bts_{year}_{month:02d}.zip"
    csv_path = RAW_DIR / f"bts_{year}_{month:02d}.csv"

    if csv_path.exists():
        print(f"  [skip] {csv_path.name} already exists")
        return csv_path

    print(f"  Downloading {year}-{month:02d}...")
    try:
        r = requests.get(url, timeout=180, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    except Exception as e:
        print(f"  ERROR downloading {year}-{month}: {e}")
        zip_path.unlink(missing_ok=True)
        return None

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        csv_inside = next(n for n in names if n.endswith(".csv"))
        z.extract(csv_inside, RAW_DIR)
        extracted = RAW_DIR / csv_inside
        extracted.rename(csv_path)

    zip_path.unlink()
    size_mb = csv_path.stat().st_size / 1e6
    print(f"  Done: {csv_path.name} ({size_mb:.1f} MB)")
    return csv_path


def main():
    targets = [(y, m) for y in [2024, 2025] for m in range(1, 13)]
    print(f"Downloading {len(targets)} monthly files (2024-2025)...")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_month, y, m): (y, m) for y, m in targets}
        results = []
        for fut in as_completed(futures):
            path = fut.result()
            if path:
                results.append(path)

    print(f"\nDownloaded {len(results)}/{len(targets)} files to {RAW_DIR}")
    return sorted(results)


if __name__ == "__main__":
    main()
