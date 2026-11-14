"""Explicit compatibility bindings for previously audited native traces."""
import json
from pathlib import Path


def audited_compatible(trace, cache, workload, scene, recording_key, *, independent=False):
    if cache is None:
        return False
    path = Path(cache) / 'pim' / 'trace-compatibility.json'
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 1:
        raise ValueError(f'unsupported trace compatibility manifest: {path}')
    for binding in manifest.get('bindings', []):
        if (binding['workload'] == workload and binding['scene'] == scene and
                binding['independent'] is independent and
                binding['current_recording_sha256'] == recording_key and
                binding['original_recording_sha256'] == trace.metadata.get('recording_sha256') and
                binding['trace_sha256'] == trace.sha256 and binding.get('evidence')):
            return True
    return False
