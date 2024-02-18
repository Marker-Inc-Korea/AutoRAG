import os
import shutil
import sysconfig


def main():
    tests_dir = os.path.join(sysconfig.get_path('purelib'), 'tests')
    print(f"Github Actions: {os.getenv('CI')}")
    print(sysconfig.get_path('purelib'))
    if os.path.exists(tests_dir):
        shutil.rmtree(tests_dir)
    else:
        print("Directory does not exist")


if __name__ == "__main__":
    main()
