import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Unified entry for all methods (PrivSTD, UG, AG, AHP, HB-Striped, PrivBayes, MWEM)"
    )

    # Positional argument: method name
    parser.add_argument(
        "method",
        type=str,
        help="Method name, e.g. AG, MWEM, PrivBayes, HB-Striped"
    )

    # Whether to use the User version
    parser.add_argument(
        "--user",
        action="store_true",
        help="Use User version (e.g. AG_User.py)"
    )

    # Parse known arguments; keep the remaining arguments unchanged
    args, remaining_args = parser.parse_known_args()

    method = args.method

    # Construct script name
    if args.user:
        script_name = f"{method}_User.py"
    else:
        script_name = f"{method}.py"

    # Check whether the script exists
    if not os.path.exists(script_name):
        print(f"[ERROR] Script not found: {script_name}")
        sys.exit(1)

    # Assemble the final command
    cmd = ["python", script_name] + remaining_args

    print("[INFO] Running command:")
    print(" ".join(cmd))

    # Execute the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
