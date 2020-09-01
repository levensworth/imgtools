# dummy file for now
# the import problem was solved formm:
# https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
import os
import sys

test_module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, test_module_path + "/../")


def test_dummy():
    return True
