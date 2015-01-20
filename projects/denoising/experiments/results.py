import os
import re
import json
import numpy as np
from bunch import Bunch, bunchify


RESULT_FORMAT = '.json'


class _ResultSet(Bunch):
    """
    Represents single .json result file of one run with:
    - Iteration statistics (best/average population fitness)
    - Run parameters
    - Results (best solution, total run time)

    The file is not actually read until one of result attributes
    are accessed.
    """
    def __init__(self, filepath, result_dir=None):
        self.filepath = filepath
        self.result_dir = result_dir
        self.id = int(os.path.splitext(
            os.path.basename(self.filepath))[0])
        self._data_initialized = False

    def __getitem__(self, key):
        """
        Do actual data reading on first result set attribute access
        """
        if key not in [
                '_data_initialized',
                'filepath',
                'run_dir',
                'id'
        ]:
            if self._data_initialized is False:
                self._read_results_file()
                self._data_initialized = True
        return super(Bunch, self).__getitem__(key)

    def _read_results_file(self):
        """
        Read single JSON-formatted result file
        """
        with open(self.filepath, 'r') as f:
            json_results = json.load(f)
        self.update(bunchify(json_results))
        return None


class _ResultDir(object):
    """
    Represents directory of one single run with a number
    of .json result files
    """
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self._result_sets = []
        self._data_initialized = False

        # Get run ID from directory name
        self.dirname = os.path.basename(self.dirpath)
        try:
            self._run_id = int(re.search(
                r'(?P<run_id>\d+)$', self.dirname).groups()[0])
        except:
            raise ValueError(
                "Run ID could not be determined from: %s" % self.dirpath)

    @property
    def run_id(self):
        return self._run_id

    def __iter__(self):
        # Lazy-load directory contents
        if self._data_initialized is False:
            self._read_result_sets()
            self._data_initialized = True
        return self._result_sets.__iter__()

    def _read_result_sets(self):
        for filename in os.listdir(self.dirpath):
            filepath = os.path.join(self.dirpath, filename)
            if os.path.isfile(filepath) and filename.endswith(RESULT_FORMAT):
                result_set = _ResultSet(filepath, self)
                self._result_sets.append(result_set)
        return None


class _ResultRuns(object):
    """
    Represents a list of result sets acquired with the same
    parameter set but during several runs.
    """
    def __init__(self, result_dir):
        self.result_dir = result_dir
        self._result_sets = {}

    def __setitem__(self, key, value):
        self._result_sets[key] = value

    def __getitem__(self, key):
        return self._result_sets[key]

    def __iter__(self):
        return self._result_sets.values().__iter__()

    @property
    def result_sets(self):
        return self._result_sets.values()

    def mean_best_fitnesses(self):
        """
        Averaged numpy array of best fitnesses per iteration
        for given list of result sets.
        """
        return np.average(np.array([
            [iteration.best_fitness for iteration in result.iterations]
            for result in self.result_sets
        ]), axis=0)

    def var_best_fitnesses(self):
        """
        Variance between best fitnesses per iteration
        among result sets
        """
        return np.var(np.array([
            [iteration.best_fitness for iteration in result.iterations]
            for result in self.result_sets
        ]), axis=0)

    def mean_avg_fitnesses(self):
        """
        Same as mean_best_fitness but for average fitnesses
        """
        return np.average(np.array([
            [iteration.average_fitness for iteration in result.iterations]
            for result in self.result_sets
        ]), axis=0)

    def var_avg_fitnesses(self):
        """
        Same as var_best_fitness but for average fitnesses
        """
        return np.var(np.array([
            [iteration.average_fitness for iteration in result.iterations]
            for result in self.result_sets
        ]), axis=0)


class Results(object):
    """
    Represents highest-level result directory which contains
    separate result directories for a number of runs
    """
    def __init__(self, dirpath):
        self.dirpath = dirpath

        # Read run directories
        self._result_dirs = []
        for run_dirname in os.listdir(self.dirpath):
            run_dirpath = os.path.join(self.dirpath, run_dirname)
            if os.path.isdir(run_dirpath) and run_dirname.startswith('run'):
                result_dir = _ResultDir(run_dirpath)
                self._result_dirs.append(result_dir)

        # Read IDs of result files from all runs and structure them
        self._result_runs = {}
        for result_dir in self._result_dirs:
            for result_set in result_dir:
                if result_set.id not in self._result_runs:
                    self._result_runs[result_set.id] = _ResultRuns(result_dir)
                self._result_runs[result_set.id][result_dir.run_id] = result_set

    def __iter__(self):
        for result_set_id, result_runs in self._result_runs.items():
            yield result_set_id, result_runs

    def __getitem__(self, key):
        return self._result_runs[key]

"""
# Getting all run results for param ID range
filtered = [
    results_runs
    for id, results_runs in results
    if param_id_start <= id <= param_id_end
]
"""
