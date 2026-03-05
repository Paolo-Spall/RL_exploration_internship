#!/usr/bin/python3

import sys

print("sys.argv: ")

for i, arg in enumerate(sys.argv):
    print(f"Argument {i}: {arg}")