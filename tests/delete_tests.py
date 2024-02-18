import os
import shutil
import sysconfig

import torch.cuda


def is_ubuntu() -> bool:
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("NAME") and "Ubuntu" in line:
                    return True
    except FileNotFoundError:
        pass
    return False


def is_github_action() -> bool:
    return is_ubuntu() and not torch.cuda.is_available()


def main():
    tests_dir = os.path.join(sysconfig.get_path('purelib'), 'tests')
    print(is_github_action())
    print(sysconfig.get_path('purelib'))
    if os.path.exists(tests_dir):
        shutil.rmtree(tests_dir)
    else:
        print("Directory does not exist")


if __name__ == "__main__":
    main()
