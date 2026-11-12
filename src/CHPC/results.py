from pathlib import Path

import numpy as np



# Results are saved as .npz files, such that running the tool (main.py) and post processing (postprocess.py) are 
# separated. Dictionaries are stored with keys of the form "<dictionary>/<key>", e.g. "cycle_data/p_ref_1", and 
# nested dictionaries such as ts_data in the same way, e.g. "ts_data/major/s". None values are not stored. 

def save_results(file_path, **results):
    """Save dictionaries of results to a single .npz file.

    Args:
        file_path: Path of the .npz file, parent folders are created if necessary.
        **results: Dictionaries to save, e.g. cycle_data=cycle_data, perf=perf.

    Returns:
        Path of the saved file.
    """
    flat_results = {}

    def _flatten(prefix, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                _flatten(f"{prefix}/{key}", value)
            elif value is not None:
                flat_results[f"{prefix}/{key}"] = np.asarray(value)

    for name, dictionary in results.items():
        _flatten(name, dictionary)
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(file_path, **flat_results)
    return file_path


def load_results(file_path):
    """Load the dictionaries of results saved with save_results.

    Args:
        file_path: Path of the .npz file.

    Returns:
        Dictionary of the saved dictionaries, e.g. results["cycle_data"]["p_ref_1"]. Scalars are returned 
        as Python floats, booleans or strings, and lists as numpy arrays.
    """
    results = {}
    with np.load(file_path, allow_pickle=False) as data:
        for flat_key in data.files:
            *names, key = flat_key.split("/")
            dictionary = results
            for name in names:
                dictionary = dictionary.setdefault(name, {})
            value = data[flat_key]
            dictionary[key] = value.item() if value.ndim == 0 else value
    return results


def method_suffix(calculation_method, separator="_"):
    """Suffix added to the result file names of calculation methods other than full_optimization.

    Args:
        calculation_method: full_optimization, three_temp_fix or fixed_PR.
        separator: Separator between the file name and the calculation method.

    Returns:
        Empty string for full_optimization, otherwise the separator followed by the calculation method.
    """
    return "" if calculation_method == "full_optimization" else f"{separator}{calculation_method}"
