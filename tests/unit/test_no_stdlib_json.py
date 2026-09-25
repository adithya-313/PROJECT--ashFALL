import subprocess
from pathlib import Path

def test_no_stdlib_json():
    """
    Test that 'import json' is not used anywhere in api/ or core/.
    This ensures we stick to orjson for performance.
    """
    repo_root = Path(__file__).parent.parent.parent
    result = subprocess.run(
        ["grep", "-r", "import json", "api/", "core/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    # grep returns 1 if no matches are found, 0 if matches are found.
    assert result.returncode == 1, f"Found 'import json' in the codebase:\n{result.stdout}"
