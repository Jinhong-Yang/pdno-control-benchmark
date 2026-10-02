"""Build all tables from archived evidence, normalize labels, add new diagnostics."""
from pathlib import Path
import subprocess,sys
S=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(S/'build_legacy_tables.py')],check=True)
subprocess.run([sys.executable,str(S/'build_submission_tables.py')],check=True)
