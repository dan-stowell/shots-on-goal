import argparse
import dataclasses
import os
import subprocess

import llm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="openrouter/z-ai/glm-4.5-air", description="model to use")
    parser.add_argument("--tool-call-limit", type=int, default=50, description="limit on number of tool calls to allow per prompt")
    parser.add_argument("--prompt", description="prompt to pass to the model")
    parser.add_argument("--workdir", default=os.getcwd(), description="working directory")
    return parser.parse_args()


@dataclasses.dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class ContainerToolbox(llm.Toolbox):
    def __init__(self, container_id: str):
        self.container_id = container_id

    def ls(self, args: tuple[str, ...]) -> CommandResult:
        """
        Run `ls` with the input args.
        """
        result = subprocess.Run(("container", "exec", self.container_id, "ls",) + args, capture_output=True, text=True)
        return CommandResult(result.returncode, result.stdout, result.stderr)

    def cat(self, paths: tuple[str, ...]) -> CommandResult:
        """
        `cat` all the input paths.
        """
        result = subprocess.Run(("container", "exec", self.container_id, "cat",) + paths, capture_output=True, text=True)
        return CommandResult(result.returncode, result.stdout, result.stderr)

    def find(self, args: tuple[str, ...]) -> CommandResult:
        """
        Run `find` with the input args.
        """
        result = subprocess.Run(("container", "exec", self.container_id, "find",) + args, capture_output=True, text=True)
        return CommandResult(result.returncode, result.stdout, result.stderr)

    def ripgrep(self, args: tuple[str, ...]) -> CommandResult:
        """
        Run `ripgrep` with the input args.
        """
        result = subprocess.Run(("container", "exec", self.container_id, "find",) + args, capture_output=True, text=True)
        return CommandResult(result.returncode, result.stdout, result.stderr)

def main():
    args = parse_args()
    model = llm.get_model(args.model_name)
    conversation = model.conversation(tools=[], chain_limit=3)

    try:
        result = conversation.chain("Do something complex").text()
    except Exception as e:
        pass
