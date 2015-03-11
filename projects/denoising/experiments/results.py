import os
import re
import json
import numpy as np
from bunch import Bunch, bunchify
# from multiprocessing.pool import ThreadPool


RESULT_FORMAT = '.json'

# Special attributes managed separately from actual result data
_RESULT_SET_SPECIAL_ATTRS = [
    'filepath',
    'result_dir',
    'id',
    'run_id',
    'load_callback',
    'is_loaded',
]


class _ResultSet(Bunch):
    """
    Represents single .json result file of one run with:
    - Iteration statistics (best/average population fitness)
    - Run parameters
    - Results (best solution, total run time)

    The file is not actually read until one of result attributes
    are accessed.
    """
    def __init__(self, filepath, result_dir=None, load_callback=None):
        self.filepath = filepath
        self.result_dir = result_dir
        if result_dir is not None:
            self.run_id = result_dir.run_id
        self.load_callback = load_callback
        self.id = int(os.path.splitext(
            os.path.basename(self.filepath))[0])
        self.is_loaded = False

    def __getitem__(self, key):
        """
        Do actual data reading on first result set attribute access
        """
        if key not in _RESULT_SET_SPECIAL_ATTRS:
            if self.is_loaded is False:
                self._read_results_file()
                self.is_loaded = True
                self.load_callback(self)
        return super(Bunch, self).__getitem__(key)

    def _read_results_file(self):
        """
        Read single JSON-formatted result file
        """
        with open(self.filepath, 'r') as f:
            json_results = json.load(f)
        self.update(bunchify(json_results))
        return None

    def unload(self):
        """
        Remove result data loaded from file
        """
        for key in self.keys():
            if key not in _RESULT_SET_SPECIAL_ATTRS:
                del self[key]
        self.is_loaded = False
        return None


class _ResultDir(object):
    """
    Represents directory of one single run with a number
    of .json result files
    """
    def __init__(self, dirpath, load_callback=None):
        self.dirpath = dirpath
        self.load_callback = load_callback
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
                result_set = _ResultSet(
                    filepath, result_dir=self, load_callback=self.load_callback)
                self._result_sets.append(result_set)
        return None


class _ResultGroup(list):
    def mean_best_fitnesses(self):
        """
        Averaged numpy array of best fitnesses per iteration
        for given list of result sets.
        """
        return np.average(np.array([
            [iteration.best_fitness for iteration in result.iterations]
            for result in self
        ]), axis=0)

    def var_best_fitnesses(self):
        """
        Variance between best fitnesses per iteration
        among result sets
        """
        return np.var(np.array([
            [iteration.best_fitness for iteration in result.iterations]
            for result in self
        ]), axis=0)

    def mean_avg_fitnesses(self):
        """
        Same as mean_best_fitness but for average fitnesses
        """
        return np.average(np.array([
            [iteration.average_fitness for iteration in result.iterations]
            for result in self
        ]), axis=0)

    def var_avg_fitnesses(self):
        """
        Same as var_best_fitness but for average fitnesses
        """
        return np.var(np.array([
            [iteration.average_fitness for iteration in result.iterations]
            for result in self
        ]), axis=0)


class Results(object):
    """
    Represents highest-level result directory which contains
    separate result directories for a number of runs.
    All interactions with results should go through instances
    of this class:

    results = Results('/path/to/results')
    results[param_id] => _ResultRuns(...)
    results[param_id][run_id] => _ResultSet(...)

    results.filter(lambda rs: rs.id == 1458) =>
        Results(with result set IDs equal to 1458)
    """
    def __init__(
            self,
            dirpath=None, result_sets=None,
            max_loaded_files=500):
        self.dirpath = dirpath

        # First, collect a flat list of result sets
        if dirpath is not None:
            # Read run directories
            self._result_dirs = []
            for run_dirname in os.listdir(self.dirpath):
                run_dirpath = os.path.join(self.dirpath, run_dirname)
                if os.path.isdir(run_dirpath) and run_dirname.startswith('run'):
                    result_dir = _ResultDir(
                        run_dirpath, load_callback=self._load_callback)
                    self._result_dirs.append(result_dir)

            # Read file info of all result files
            self._result_sets = [
                result_set
                for res_dir in self._result_dirs
                for result_set in res_dir
            ]
        elif result_sets is not None:
            # Take what has been passed
            self._result_sets = result_sets
        else:
            raise ValueError(
                "Must supply either result directory or result set list")

        # Group by param set IDs
        self._result_groups = {}
        for result_set in self._result_sets:
            if result_set.id not in self._result_groups:
                result_group = _ResultGroup()
                self._result_groups[result_set.id] = result_group
            result_group.append(result_set)

        self._max_loaded_files = max_loaded_files
        # Queue for keeping track of which files have been read
        self._loaded_result_sets = []
        for result_set in self._result_sets:
            if result_set.is_loaded is True:
                self._loaded_result_sets.append(result_set)

    def __getitem__(self, key):
        return self._result_runs[key]

    def iter_sets(self):
        """
        Iterate flat list of result sets
        """
        for value in self._result_sets:
            yield value

    def iter_groups(self):
        """
        Iterate result runs where each contains a list
        of result sets
        """
        for key, value in self._result_groups.items():
            yield key, value

    def filter(self, predicate):
        """
        Returns new instance containing result sets satisfying
        the predicate
        """
        filtered_sets = [
            result_set
            for result_set in self.iter_sets()
            if predicate(result_set) is True
        ]
        return Results(result_sets=filtered_sets)

    def _load_callback(self, result_set):
        """
        Called by _ResultSet when data is actually read
        from .json file
        """
        self._loaded_result_sets.append(result_set)
        if len(self._loaded_result_sets) >= self._max_loaded_files:
            oldest_result_set = self._loaded_result_sets.pop(0)
            # Remove result file data from memory
            oldest_result_set.unload()

    def _unload_all(self):
        for rs in self.iter_sets():
            rs.unload()
