import tarfile
from pathlib import Path

base = Path("data/email_dataset")
destination = base / "extracted"
destination.mkdir(parents=True, exist_ok=True)

archives = [
    base / "easy_ham.tar.bz2",
    base / "spam.tar.bz2",
]

for archive in archives:
    if not archive.exists() or archive.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty archive: {archive}")

    with tarfile.open(archive, "r:bz2") as tar:
        members = [
            member for member in tar.getmembers()
            if member.isfile()
            and not member.issym()
            and not member.islnk()
        ]
        tar.extractall(destination, members=members, filter="data")
        print(f"Extracted {len(members)} files from {archive.name}")

print("\nExtraction completed.")
print("Destination:", destination.resolve())
