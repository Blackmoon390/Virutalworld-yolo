import sys

def install_modules():
    import subprocess
    

    modules = [
        "opencv-python",
        "numpy",
        "torch --index-url https://download.pytorch.org/whl/cu118",
        "torchvision --index-url https://download.pytorch.org/whl/cu118",
        "torchaudio --index-url https://download.pytorch.org/whl/cu118",
    ]

    for module in modules:
        try:
            sys.stdout.write(f"\rInstalling: {module}...")
            sys.stdout.flush()

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + module.split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            sys.stdout.write(f"\rInstalled: {module} ✔️{' ' * 20}")
            sys.stdout.flush()

        except Exception as e:
            sys.stdout.write(f"\rError installing {module}: {e}{' ' * 20}")
            sys.stdout.flush()

    print("\nAll installations attempted.\n")


with open("settings.txt", "r") as file:
    lines = file.readlines()

new_lines = []

for line in lines:
    if line.startswith("Installedlibraries"):
        checklib = line.strip().split("=")[1]

        if checklib == "True":
            
            sys.stdout.write("\rLibraries already installed. Skipping.      ")
            sys.stdout.flush()
            new_lines.append(line)

        else:
            # Install only if False
            install_modules()
            new_lines.append("Installedlibraries=True\n")

    else:
        new_lines.append(line)

with open("settings.txt", "w") as file:
    file.writelines(new_lines)
