"""``python -m loop --help`` must exit 0 without the FATAL crash banner (argparse raises SystemExit)."""
import subprocess
import sys


def _run(*args):
    return subprocess.run([sys.executable, "-m", "loop", *args], capture_output=True, text=True, timeout=120)


def test_help_exits_zero_without_fatal():
    for args in (("--help",), ("run", "--help")):
        r = _run(*args)
        assert r.returncode == 0, r.stderr
        assert "FATAL" not in r.stderr and "uncaught" not in r.stderr


def test_usage_error_keeps_argparse_code():
    r = _run("run", "--no-such-flag")
    assert r.returncode == 2
    assert "FATAL" not in r.stderr
