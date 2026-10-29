"""Exercise shell entrypoints without installing a training stack or probing GPUs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(os.name == 'nt' or not shutil.which('bash'), 'Linux shell entrypoints')
class ScriptPaths(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='conventional-paths-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith(('LAPLACE_', 'XDG_'))}
        self.environment['HOME'] = str(self.root / 'home')

    def paths(self, **values):
        result = subprocess.run(['bash', '-eu', '-c',
            'source "$1"; printf "%s\\0" "$VENV" "$STATE_DEFAULT" "$OUTPUT_DEFAULT"',
            'paths', str(ROOT / 'scripts/paths.sh')],
            env=dict(self.environment, **values), check=True, capture_output=True, text=True)
        return result.stdout.rstrip('\0').split('\0')

    def test_declared_roots_are_shared_across_checkouts_without_creating_outputs(self):
        paths = self.paths(LAPLACE_BUILD=str(self.root / 'build with spaces'),
                           LAPLACE_WORK=str(self.root / 'work'), LAPLACE_MODELS=str(self.root / 'models'))
        self.assertEqual(paths, [str(self.root / 'build with spaces/Laplace-Conventional/venv'),
                                 str(self.root / 'work/conventional'),
                                 str(self.root / 'models/conventional/checkpoints')])
        self.assertFalse(any(Path(path).exists() for path in paths))
        self.assertFalse(any(Path(path).is_relative_to(ROOT) for path in paths))

    def test_standalone_paths_follow_xdg_and_explicit_prior_run_overrides(self):
        self.assertEqual(self.paths(), [str(self.root / 'home/.cache/laplace/build/Laplace-Conventional/venv'),
                                        str(self.root / 'home/.local/state/laplace/conventional'),
                                        str(self.root / 'home/.local/share/laplace/models/conventional/checkpoints')])
        self.assertEqual(self.paths(LAPLACE_CONVENTIONAL_VENV=str(self.root / 'old/.venv'),
                                    LAPLACE_CONVENTIONAL_STATE=str(self.root / 'old/.state'),
                                    LAPLACE_CONVENTIONAL_OUTPUT=str(self.root / 'old/checkpoints')),
                         [str(self.root / name) for name in ('old/.venv', 'old/.state', 'old/checkpoints')])

    def test_training_entrypoint_uses_declared_state_and_preserves_argument_override(self):
        environment = dict(self.environment, LAPLACE_WORK=str(self.root / 'work'))
        for arguments, state in (([], self.root / 'work/conventional'),
                                 ([str(self.root / 'prior run')], self.root / 'prior run')):
            result = subprocess.run(['bash', str(ROOT / 'scripts/train.sh'), *arguments],
                                     env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn(str(state / 'training.json'), result.stderr)
        self.assertFalse((ROOT / '.state').exists())


if __name__ == '__main__':
    unittest.main()
