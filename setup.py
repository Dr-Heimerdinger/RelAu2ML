import subprocess
import sys


def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError:
        print(f"Failed to run: {command}")
        sys.exit(1)


def main():
    print("Installing dependencies...")
    print("Available installation options:")
    print("  uv sync                           # Default installation")
    print("  uv sync --group dev               # Include dev dependencies")
    print("  uv sync --extra all               # Full installation with deep learning + chatui")
    print("  uv sync --extra transformers      # Only deep learning dependencies")
    print("  uv sync --extra chatui            # Only chatui dependencies")
    run_command("uv sync")

    print("Installing pre-commit hooks...")
    run_command("pre-commit install")

    print("Setup complete!")


if __name__ == "__main__":
    main()
