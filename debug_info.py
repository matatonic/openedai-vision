#!/usr/bin/env python

import torch
import re
import sys
import pkg_resources

def get_cuda_info():
    print("---- cuda")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    for i in range(0, torch.cuda.device_count()):
        print(f"CUDA device[{i}]{torch.cuda.get_device_capability(i)}: {torch.cuda.get_device_name(i)}")

def get_python_version():
    print("---- python")
    print(sys.version)

def get_pip_packages():
    print("---- pip")
    try:
        packages = set(["transformers"])
        with open("requirements.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("http:") or line.startswith("https:") or line.startswith("git+"):
                    continue
                package = re.split(r"[=<>;#\[ ]", line)[0]
                packages.add(package)

        for package in sorted(list(packages)):
            try:
                version = pkg_resources.get_distribution(package).version
                print(f"{package}=={version}")
            except pkg_resources.DistributionNotFound:
                print(f"{package}: Not found")

    except FileNotFoundError:
        print("requirements.txt not found")

get_cuda_info()
get_python_version()
get_pip_packages()
