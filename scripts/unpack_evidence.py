"""Verify the submitted archive and restore only its artifacts/ and output/ trees."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'occupancy-perception-pipeline/'


def main():
    archive = ROOT / 'submission.zip'
    with zipfile.ZipFile(archive) as bundle:
        hashes = json.loads(bundle.read(PREFIX + 'SHA256SUMS.json'))
        selected = []
        for name, digest in hashes.items():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unsafe archive path: {name}')
            data = bundle.read(PREFIX + name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'Checksum mismatch: {name}')
            if path.parts[0] in ('artifacts', 'output'):
                selected.append((path, data))
        for path, data in selected:
            destination = ROOT.joinpath(*path.parts)
            if not destination.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError(f'Unsafe destination: {destination}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    print(f'Verified {len(hashes)} checksums and restored {len(selected)} evidence files.')


if __name__ == '__main__':
    main()
