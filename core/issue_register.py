"""
File: core/issue_register.py
Purpose: Checks the issue register in config/issues.yaml against reality.
Why it exists: We need to ensure no review issue is silently forgotten. This checker enforces that 
every built issue has concrete evidence in the project, every planned issue has future work noted, 
and no unknown IDs are introduced.
"""

import sys
from pathlib import Path

from core.config_validator import load_yaml

EXPECTED_IDS = {
    "G1", "G2", "G3", "G4", "G5", "G6",
    "L1", "L2", "L3", "L4", "L5",
    "A1", "A2", "A3",
    "N1", "N2", "N3", "N4",
    "E1", "E2", "E3", "E4",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"
}

ALLOWED_STATUSES = ["built", "planned", "deferred"]
ALLOWED_KINDS = ["config", "amendment", "doc", "test"]
CONFIG_FILES_WITH_ISSUE_IDS = ["config/world_ranges.yaml", "config/evaluation.yaml"]


def load_register(path):
    """Load the register YAML file and return it as a dict."""
    try:
        return load_yaml(path)
    except Exception:
        return {}


def check_ids(register):
    """Check that all expected IDs are present and no unknown IDs are included."""
    errors = []
    found_ids = set(register.keys())
    missing = EXPECTED_IDS - found_ids
    unknown = found_ids - EXPECTED_IDS
    if missing:
        errors.append(f"Missing expected IDs in register: {sorted(list(missing))}")
    if unknown:
        errors.append(f"Unknown IDs found in register: {sorted(list(unknown))}")
    return errors


def check_config_evidence(issue_id, item, root):
    """Check that the issue ID appears in the given config file block."""
    errors = []
    try:
        file_path = root / item.get("file", "")
        if not file_path.exists():
            return [f"File {item.get('file')} not found"]
        data = load_yaml(file_path)
        blocks = data.get("parameters", data)
        name = item.get("name", "")
        if name not in blocks:
            return [f"Block {name} not found in {item.get('file')}"]
        issue_ids = blocks[name].get("issue_ids", [])
        if issue_id not in issue_ids:
            return [f"Issue ID {issue_id} not in issue_ids for {name}"]
    except Exception as e:
        return [f"Error reading config evidence: {e}"]
    return errors


def find_amendment_issue_ids(root):
    """Parse docs/research/amendments.md to extract issue IDs for each A-nnn amendment."""
    result = {}
    amendments_file = root / "docs" / "research" / "amendments.md"
    if not amendments_file.exists():
        return result
    try:
        content = amendments_file.read_text(encoding="utf-8")
        current_amendment = None
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("## A-"):
                current_amendment = line[3:].strip()
            elif current_amendment and line.startswith("| Issue IDs |"):
                parts = line.split("|")
                if len(parts) >= 3:
                    ids_str = parts[2].strip()
                    ids = [i.strip() for i in ids_str.split(",") if i.strip()]
                    result[current_amendment] = ids
                current_amendment = None
    except Exception:
        pass
    return result


def check_amendment_evidence(issue_id, item, root):
    """Check that the issue ID appears in the specified amendment."""
    amendments = find_amendment_issue_ids(root)
    amendment_id = item.get("id", "")
    if amendment_id not in amendments:
        return [f"Amendment {amendment_id} not found"]
    if issue_id not in amendments[amendment_id]:
        return [f"Issue ID {issue_id} not in amendment {amendment_id}"]
    return []


def check_doc_evidence(issue_id, item, root):
    """Check that the specified document contains the required text."""
    file_path = root / item.get("file", "")
    text = item.get("text", "")
    if not file_path.exists():
        return [f"File {item.get('file')} not found"]
    try:
        content = file_path.read_text(encoding="utf-8")
        if text not in content:
            return [f"Text '{text}' not found in {item.get('file')}"]
    except Exception as e:
        return [f"Error reading doc evidence: {e}"]
    return []


def check_test_evidence(issue_id, item, root):
    """Check that the specified test function exists in the file."""
    file_path = root / item.get("file", "")
    name = item.get("name", "")
    if not file_path.exists():
        return [f"File {item.get('file')} not found"]
    try:
        content = file_path.read_text(encoding="utf-8")
        expected_def = "def " + name + "("
        if expected_def not in content:
            return [f"Test function {name} not found in {item.get('file')}"]
    except Exception as e:
        return [f"Error reading test evidence: {e}"]
    return []


def check_evidence(issue_id, item, root):
    """Route evidence to the correct checker based on its kind."""
    kind = item.get("kind")
    if kind not in ALLOWED_KINDS:
        return [f"Unknown evidence kind '{kind}'"]
    if kind == "config":
        return check_config_evidence(issue_id, item, root)
    if kind == "amendment":
        return check_amendment_evidence(issue_id, item, root)
    if kind == "doc":
        return check_doc_evidence(issue_id, item, root)
    if kind == "test":
        return check_test_evidence(issue_id, item, root)
    return []


def check_entry(issue_id, entry, root):
    """Check one issue entry for valid title, status, reason, later_work, and evidence."""
    errors = []
    
    if not entry.get("title") or not isinstance(entry["title"], str):
        errors.append("Title is missing or empty")
        
    status = entry.get("status")
    if status not in ALLOWED_STATUSES:
        errors.append(f"Unknown status '{status}'")
        
    if status == "deferred":
        if not entry.get("reason"):
            errors.append("Deferred issue is missing a reason")
            
    if status == "built":
        evidence_list = entry.get("evidence", [])
        if not evidence_list:
            errors.append("Built issue must have at least one evidence item")
        for item in evidence_list:
            item_errors = check_evidence(issue_id, item, root)
            errors.extend(item_errors)
            
    if status == "planned":
        later_work = entry.get("later_work", [])
        if not later_work:
            errors.append("Planned issue must have non-empty later_work list")
            
    for note in entry.get("later_work", []):
        if not isinstance(note, str) or not note.startswith("Level "):
            errors.append(f"Bad later_work note: '{note}'")
            
    return [f"{issue_id}: {err}" for err in errors]


def find_config_issue_ids(root):
    """Find all issue IDs used in the issue_ids lists of the main config files."""
    found = set()
    for file_name in CONFIG_FILES_WITH_ISSUE_IDS:
        file_path = root / file_name
        if file_path.exists():
            try:
                data = load_yaml(file_path)
                blocks = data.get("parameters", data)
                if isinstance(blocks, dict):
                    for k, v in blocks.items():
                        if isinstance(v, dict):
                            for i in v.get("issue_ids", []):
                                found.add(i)
            except Exception:
                pass
    return found


def check_config_ids_known(config_ids):
    """Ensure no unknown issue ID appears in any config file."""
    unknown = config_ids - EXPECTED_IDS
    if unknown:
        return [f"Unknown ID in config files: {i}" for i in sorted(list(unknown))]
    return []


def check_register(register, root):
    """Run all checks on the loaded register and return a list of errors."""
    errors = []
    errors.extend(check_ids(register))
    for issue_id, entry in register.items():
        if isinstance(entry, dict):
            errors.extend(check_entry(issue_id, entry, root))
    
    config_ids = find_config_issue_ids(root)
    errors.extend(check_config_ids_known(config_ids))
    return errors


def main(argv=None):
    """Load the issue register, check it, print any errors, and return 1 if there are any."""
    args = argv if argv is not None else sys.argv[1:]
    root = Path(__file__).resolve().parent.parent
    path = Path(args[0]) if args else root / "config" / "issues.yaml"
    
    if not path.exists():
        print(f"Register file {path} not found")
        return 1
        
    register = load_register(path)
    if not register:
        print("Register file is empty or invalid YAML")
        return 1
        
    errors = check_register(register, root)
    if errors:
        for err in errors:
            print(err)
        return 1
        
    built = sum(1 for k, v in register.items() if isinstance(v, dict) and v.get("status") == "built")
    planned = sum(1 for k, v in register.items() if isinstance(v, dict) and v.get("status") == "planned")
    deferred = sum(1 for k, v in register.items() if isinstance(v, dict) and v.get("status") == "deferred")
    print(f"OK: 31 issues checked (built {built}, planned {planned}, deferred {deferred})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
