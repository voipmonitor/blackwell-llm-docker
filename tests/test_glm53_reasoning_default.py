"""GLM launcher reasoning defaults without model loading or CUDA access."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "recipes/glm53/serve-glm53-flash-nvfp4-dflash2.sh"
)


class ReasoningDefaultTest(unittest.TestCase):
    def command(self, mode, *args):
        env = dict(os.environ, DRY_RUN="1", SPECULATOR=mode, MTP_DEPTH="3")
        result = subprocess.run(
            ["bash", str(LAUNCHER), *args],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return shlex.split(result.stdout.split(" launch:", 1)[1])

    def defaults(self, command):
        # argparse uses the last complete occurrence of the option, allowing
        # operator-supplied JSON to override launcher defaults.
        values = [
            json.loads(command[i + 1])
            for i, arg in enumerate(command)
            if arg == "--default-chat-template-kwargs"
        ]
        self.assertTrue(values)
        return values[-1]

    def test_high_for_both_speculator_families(self):
        for mode in ("mtp", "dflash2"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    self.defaults(self.command(mode)), {"reasoning_effort": "high"}
                )

    def test_operator_can_override_default_json(self):
        override = {"reasoning_effort": "max", "clear_thinking": True}
        self.assertEqual(
            self.defaults(
                self.command("mtp", "--default-chat-template-kwargs", json.dumps(override))
            ),
            override,
        )


if __name__ == "__main__":
    unittest.main()
