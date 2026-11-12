#!/usr/bin/env python3
"""Make DeepStream 9.2's pyservicemaker bindings compile with pybind11 3 (the first pybind11 that supports Python 3.14): it refuses
keep_alive on def_property*, so the three getters that use it become py::cpp_function, as its error message asks.

    python3 patch_pyservicemaker.py <copy of service-maker/sources/python/src>
"""
import os, re, sys

path = os.path.join(sys.argv[1], "bindings", "metadata.h")
src = open(path).read()
src, n = re.subn(r'\.def_property_readonly\("(\w+)", (\[\]\(BatchMetadata& self\)[^\n]*\{\n(?:(?!\n        \.def).*\n)*?        \}, py::keep_alive<0, 1>\(\))\)',
                 r'.def_property_readonly("\1", py::cpp_function(\2))', src)
assert n == 3, f"expected 3 getters with keep_alive, found {n} (another DeepStream version?)"
open(path, "w").write(src)
print("patched bindings/metadata.h: 3 getters")
