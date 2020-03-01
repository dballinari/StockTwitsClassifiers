"""
Module import helper: Modifies PATH in order to allow us to import other directories of the project.
"""
import sys
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))