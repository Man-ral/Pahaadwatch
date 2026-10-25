
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
import os
import re

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "imerg_daily"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://jsimpsonhttps.pps.eosdis.nasa.gov"
    "/imerg/gis/early/"
)

DATE_RE = re.compile(r"3IMERG\.(\d{8})-")
VERSION_RE = re.compile(
    r"\.(V\d+[A-Z]?)\.1day\.tif$",
    re.IGNORECASE,
)


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def main():
    load_dotenv(ROOT / ".env")
    email = os.getenv("PPS_EMAIL")

    if not email:
        raise RuntimeError(
            "PPS_EMAIL is missing from the project's .env file."
        )

    email = email.strip().lower()
    session = requests.Session()
    session.auth = (email, email)

    print("Checking NASA IMERG Early Run directory...")

    response = session.get(BASE_URL, timeout=60)

    if response.status_code in (401, 403):
        raise RuntimeError(
            "NASA PPS access denied. Verify your registered email "
            "and ensure Near-Realtime Products access is enabled."
        )

    response.raise_for_status()

    parser = LinkParser()
    parser.feed(response.text)

    # Collect daily TIFFs and their product versions.
    remote_files = {}

    for href in parser.links:
        filename = Path(urlparse(href).path).name

        if not filename.lower().endswith(".1day.tif"):
            continue

        date_match = DATE_RE.search(filename)
        version_match = VERSION_RE.search(filename)

        if not date_match or not version_match:
            continue

        file_date = datetime.strptime(
            date_match.group(1), "%Y%m%d"
        ).date()

        remote_files[file_date] = (
            filename,
            urljoin(BASE_URL, href),
            version_match.group(1).upper(),
        )

    if not remote_files:
        raise RuntimeError(
            "No daily TIFF links were found in the NASA directory. "
            "Check PPS access and the directory listing before retrying."
        )

    # Keep the same product version as your existing files.
    local_versions = []

    for path in DATA_DIR.glob("*.1day.tif"):
        match = VERSION_RE.search(path.name)
        if match:
            local_versions.append(match.group(1).upper())

    if local_versions:
        expected_version = max(
            set(local_versions), key=local_versions.count
        )
    else:
        expected_version = max(
            remote_files.values(), key=lambda item: item[0]
        )[2]

    today = datetime.now(timezone.utc).date()
    earliest = today - timedelta(days=45)

    existing_names = {
        path.name
        for path in DATA_DIR.glob("*.1day.tif")
    }

    downloaded = 0

    for file_date, (filename, file_url, version) in sorted(
        remote_files.items()
    ):
        if not earliest <= file_date <= today:
            continue

        if version != expected_version:
            continue

        if filename in existing_names:
            continue

        print("Downloading:", filename)

        result = session.get(file_url, timeout=120)

        if result.status_code in (401, 403):
            raise RuntimeError(
                f"NASA PPS denied access to {filename}."
            )

        result.raise_for_status()
        content = result.content

        # Reject HTML error pages and invalid TIFF downloads.
        if not (
            content[:4] in (
                b"II*\x00", b"MM\x00*",
                b"II+\x00", b"MM\x00+",
            )
        ):
            raise RuntimeError(
                f"{filename} did not download as a valid TIFF."
            )

        destination = DATA_DIR / filename
        temporary = destination.with_suffix(".tmp")

        try:
            temporary.write_bytes(content)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

        downloaded += 1

    # Check whether the latest 30 dates are all present.
    available_dates = set()

    for path in DATA_DIR.glob("*.1day.tif"):
        match = DATE_RE.search(path.name)
        version_match = VERSION_RE.search(path.name)

        if match and version_match:
            if version_match.group(1).upper() == expected_version:
                available_dates.add(
                    datetime.strptime(
                        match.group(1), "%Y%m%d"
                    ).date()
                )

    print(f"\nNew files downloaded: {downloaded}")
    print(f"Product version: {expected_version}")

    if available_dates:
        latest = max(available_dates)
        required = {
            latest - timedelta(days=i) for i in range(30)
        }
        missing = sorted(required - available_dates)

        print(f"Latest available date: {latest}")

        if missing:
            print("Missing dates in the latest 30-day window:")
            for day in missing:
                print(" ", day)
        else:
            print("A complete 30-day window is available.")
    else:
        print("No usable daily files are present.")


if __name__ == "__main__":
    main()
