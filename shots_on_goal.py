import argparse
import os
import subprocess

import llm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="openrouter/z-ai/glm-4.5-air", help="model to use")
    parser.add_argument("--tool-call-limit", type=int, default=50, help="limit on number of tool calls to allow per prompt")
    parser.add_argument("--prompt", required=True, help="prompt to pass to the model")
    parser.add_argument("--workdir", default=os.getcwd(), help="directory to mount into /workspace inside the container")
    parser.add_argument("--image", default="shots-on-goal:latest", help="container image to run tools inside")
    args = parser.parse_args()
    args.workdir = os.path.abspath(args.workdir)
    return args


def start_container(image: str, workdir: str) -> str:
    result = subprocess.run(
        (
            "container",
            "run",
            "-d",
            "--rm",
            "-v",
            f"{workdir}:/workspace",
            "-w",
            "/workspace",
            image,
            "sleep",
            "infinity",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def stop_container(container_id: str) -> None:
    try:
        subprocess.run(
            ("container", "stop", container_id),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Failed to stop container {container_id}: {exc}")


class ContainerToolbox(llm.Toolbox):
    def __init__(self, container_id: str):
        self.container_id = container_id

    def _exec(self, *command: str) -> str:
        result = subprocess.run(
            ("container", "exec", self.container_id) + command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(error)
        return result.stdout

    def ls(self, *args: str) -> str:
        """
        Run `ls` with the input args.
        """
        return self._exec("ls", *args)

    def cat(self, *paths: str) -> str:
        """
        `cat` all the input paths.
        """
        return self._exec("cat", *paths)

    def find(self, *args: str) -> str:
        """
        Run `find` with the input args.
        """
        return self._exec("find", *args)

    def ripgrep(self, *args: str) -> str:
        """
        Run `ripgrep` with the input args.
        """
        return self._exec("rg", *args)

def main():
    args = parse_args()
    container_id = None

    try:
        container_id = start_container(args.image, args.workdir)
        toolbox = ContainerToolbox(container_id)
        model = llm.get_model(args.model_name)
        conversation = model.conversation(tools=[toolbox], chain_limit=args.tool_call_limit)
        response = conversation.chain(args.prompt).text()
        print(response)
    except Exception as exc:
        print(f"Error: {exc}")
        raise
    finally:
        if container_id:
            stop_container(container_id)


if __name__ == "__main__":
    main()
