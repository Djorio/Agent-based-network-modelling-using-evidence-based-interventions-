"""
mov_utils.py
Shared import utility to load all pipeline modules cleanly.
Avoids Python's restriction on modules starting with digits.
"""

import importlib.util
import os


def _load_module(filename: str, name: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod  # register so pickle/joblib can resolve classes defined in it
    spec.loader.exec_module(mod)
    return mod


_data_prep = None
_network = None
_abc_smc = None
_sim = None
_val = None
_sa = None
_cea = None
_figs = None
_writeup = None


def data_prep():
    global _data_prep
    if _data_prep is None:
        _data_prep = _load_module("00_data_preparation.py", "data_prep")
    return _data_prep


def baseline():
    return _load_module("01_baseline_prevalence.py", "baseline")


def network():
    global _network
    if _network is None:
        _network = _load_module("02_network_model.py", "network_model")
    return _network


def abc_smc():
    global _abc_smc
    if _abc_smc is None:
        _abc_smc = _load_module("03_abc_smc_calibration.py", "abc_smc")
    return _abc_smc


def simulation():
    global _sim
    if _sim is None:
        _sim = _load_module("04_simulation.py", "simulation")
    return _sim


def validation():
    global _val
    if _val is None:
        _val = _load_module("05_calibration_validation.py", "validation")
    return _val


def sensitivity():
    global _sa
    if _sa is None:
        _sa = _load_module("06_sensitivity_analysis.py", "sensitivity")
    return _sa


def cea():
    global _cea
    if _cea is None:
        _cea = _load_module("07_cost_effectiveness.py", "cea")
    return _cea


def figures():
    global _figs
    if _figs is None:
        _figs = _load_module("08_figures.py", "figures")
    return _figs


def writeup():
    global _writeup
    if _writeup is None:
        _writeup = _load_module("09_statistical_writeup.py", "writeup")
    return _writeup
