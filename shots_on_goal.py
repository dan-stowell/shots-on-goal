import argparse
import logging
import os
import subprocess

import llm


logger = logging.getLogger(__name__)


def _preview(value: object, limit: int = 80) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


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
    logger.info("Starting container from image '%s' with workdir '%s'", image, workdir)
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
    container_id = result.stdout.strip()
    logger.info("Started container %s", container_id)
    return container_id


def stop_container(container_id: str) -> None:
    try:
        logger.info("Stopping container %s", container_id)
        subprocess.run(
            ("container", "stop", container_id),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Failed to stop container {container_id}: {exc}")


class ContainerToolbox(llm.Toolbox):
    def __init__(self, container_id: str):
        self.container_id = container_id

    def _exec(self, command_and_args: list[str]) -> str:
        logger.info("Executing tool command: %s", " ".join(command_and_args))
        result = subprocess.run(
            ("container", "exec", self.container_id) + tuple(command_and_args),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(error)
        output = result.stdout
        logger.info("Tool command succeeded: %s", " ".join(command_and_args))
        return output

    def ls(self, args: list[str]) -> str:
        """
        Run `ls` with the input args.
        """
        return self._exec(["ls", *args])

    def cat(self, paths: list[str]) -> str:
        """
        `cat` all the input paths.
        """
        return self._exec(["cat", *paths])

    def find(self, args: list[str]) -> str:
        """
        Run `find` with the input args.
        """
        return self._exec(["find", *args])

    def rg(self, args: list[str]) -> str:
        """
        Run `rg` with the input args.
        """
        return self._exec(["rg", *args])


def _log_before_call(tool, tool_call):
    tool_name = getattr(tool, "name", getattr(tool, "__name__", "unknown"))
    logger.info("Before tool call: %s args=%s", tool_name, tool_call.arguments)


def _log_after_call(tool, tool_call, tool_result):
    tool_name = getattr(tool, "name", getattr(tool, "__name__", "unknown"))
    preview = _preview(tool_result.output)
    logger.info("After tool call: %s result=%r", tool_name, preview)

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    args = parse_args()
    for key, value in vars(args).items():
        logger.info("Arg %s: %s", key, _preview(value))
    container_id = None

    try:
        container_id = start_container(args.image, args.workdir)
        toolbox = ContainerToolbox(container_id)
        model = llm.get_model(args.model_name)
        conversation = model.conversation(
            tools=[toolbox],
            chain_limit=args.tool_call_limit,
            before_call=_log_before_call,
            after_call=_log_after_call,
        )
        logger.info("Prompt: %r", _preview(args.prompt))
        response = conversation.chain(args.prompt).text()
        logger.info("Model response: %r", _preview(response))
        print(response)
    except Exception as exc:
        logger.error("Error: %s", exc)
        raise
    finally:
        if container_id:
            stop_container(container_id)


if __name__ == "__main__":
    main()
