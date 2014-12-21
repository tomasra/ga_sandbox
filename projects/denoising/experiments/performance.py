#!/usr/bin/env python
import argparse
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import numpy as np

import pylab as plt
import matplotlib
matplotlib.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import curve_fit
from projects.denoising.experiments.experiment import read_results


MIN_PROC_COUNT = 2


def f_exp(x, a, b, c):
    return a * np.exp(-b * x) + c


def f_plane(x, a, b, c):
    return a * x[0] + b * x[1] + c


def plane_z_points(mesh_x, mesh_y, *plane_params):
    """
    Calculate z coordinate for x/y plane points
    """
    zs = np.array([
        f_plane(
            [x, y],
            *plane_params)
        for x, y in zip(
            np.ravel(mesh_x),
            np.ravel(mesh_y))
    ])
    zs = zs.reshape(mesh_x.shape)
    return zs


def plot_real_predicted(
        data, model,
        population_size, chromosome_length):
    filtered_data = data.filter(
        population_size=population_size,
        chromosome_length=chromosome_length
    )
    filtered_times = [p[3] for p in filtered_data]
    fit_global = model.predict(filtered_data)

    fig = plt.figure(figsize=(14, 7))
    plt.xlabel(u'Procesu skaicius')
    plt.ylabel(u'Vykdymo trukme')
    plt.plot(filtered_times, label='Reali trukme')
    # plt.plot(fit_local, label='Lokali aproksimacija')
    plt.plot(fit_global, label='Globali aproksimacija')
    plt.show()


def plot_3d():
    # TODO

    # mesh_x, mesh_y = np.meshgrid(
    #     xrange(10, 50 + 1, 5),
    #     xrange(10, 50 + 1, 5))

    # exp1_z = plane_z_points(mesh_x, mesh_y, *exp1_popt)
    # exp2_z = plane_z_points(mesh_x, mesh_y, *exp2_popt)
    # exp3_z = plane_z_points(mesh_x, mesh_y, *exp3_popt)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(p_sizes, c_lengths, exp_param1, c='r', marker='o')
    # ax.plot_surface(mesh_x, mesh_y, exp1_z)
    # ax.set_xlabel('P size')
    # ax.set_ylabel('C length')
    # ax.set_zlabel('Exp param 1')
    # plt.show()
    pass


class PerformanceModel(object):
    def __init__(self):
        self.exp1_params = None
        self.exp2_params = None
        self.exp3_params = None

    @staticmethod
    def load(filepath):
        with open(filepath, 'r') as fp:
            unpickled = pickle.load(fp)
        return unpickled

    def save(self, filepath):
        with open(filepath, 'w') as fp:
            pickle.dump(self, fp)
        return None

    def fit(self, data):
        """
        Step #1: local exponent fit of proc_count to time
        for each fixed population size and chromosome length
        """
        popt_local, pcov_local, params_local = [], [], []
        # params_fixed = []
        for population_size in data.population_sizes:
            for chromosome_length in data.chromosome_lengths:
                filtered_data = data.filter(
                    population_size=population_size,
                    chromosome_length=chromosome_length)

                # Fit to exponent for each population size / chromo length pair
                proc_counts = [p[2] for p in filtered_data]
                times = [p[3] for p in filtered_data]
                popt, pcov = curve_fit(
                    f_exp,
                    proc_counts,
                    times,
                    # HACK - hardcoded initial values
                    p0=(0.3, 0.6, 0.03))

                popt_local.append(popt)
                pcov_local.append(pcov)
                params_local.append([population_size, chromosome_length])

        """
        Step #2: linear fit of population size and chromosome length
        to previously fitted exponent parameters
        """
        p_sizes = [p[0] for p in params_local]
        c_lengths = [p[1] for p in params_local]
        exp_param1 = [p[0] for p in popt_local]
        exp_param2 = [p[1] for p in popt_local]
        exp_param3 = [p[2] for p in popt_local]

        # Linear regression for every exponential function param
        exp1_popt, exp1_pcov = curve_fit(
            f_plane, [p_sizes, c_lengths], exp_param1)
        exp2_popt, exp2_pcov = curve_fit(
            f_plane, [p_sizes, c_lengths], exp_param2)
        exp3_popt, exp3_pcov = curve_fit(
            f_plane, [p_sizes, c_lengths], exp_param3)

        # Save parameters
        self.exp1_params = exp1_popt
        self.exp2_params = exp2_popt
        self.exp3_params = exp3_popt

    def predict(self, data):
        return [
            self.predict_one(point[0], point[1], point[2])
            for point in data
        ]

    def predict_one(
            self,
            population_size, chromosome_length, proc_count):
        """
        The main function for approximating time of one iteration
        with given parameters
        """
        val = f_exp(
            proc_count,
            f_plane(
                [population_size, chromosome_length],
                *self.exp1_params),
            f_plane(
                [population_size, chromosome_length],
                *self.exp2_params),
            f_plane(
                [population_size, chromosome_length],
                *self.exp3_params)
        )
        return val

    def optimal_proc_count(
            self, population_size, chromosome_length,
            speedup_threshold=10.0):     # percentages!
        """
        Start with a minimum number of processes,
        keep adding an additional one and check if time decrease is good enough.
        Stop this when time decrease percentage falls below speedup_threshold.
        """
        time_previous, time_current = None, None
        for proc_count in xrange(
                MIN_PROC_COUNT,
                population_size + MIN_PROC_COUNT):
            # Estimated iteration time with current process count
            time_previous = time_current
            time_current = self.predict_one(
                population_size, chromosome_length, proc_count)

            # Skip the first
            if proc_count == MIN_PROC_COUNT:
                continue

            # Time percentage decrease compared to previous proc count
            speedup_pct = (time_previous - time_current) / time_previous * 100.0
            if speedup_pct < speedup_threshold:
                # Not worth adding this additional process
                return proc_count - 1

        # Not found?
        raise RuntimeError("Process count exceeded population size")

    def r2_score(self, data):
        """
        Goodness-of-fit for new data (must include real times)
        """
        times_true = [d[3] for d in data]
        times_predicted = model.predict(data)
        score = r2_score(times_true, times_predicted)
        return score


class PerformanceData(object):
    def __init__(self, data):
        self.data = data
        self._population_sizes = list(set([
            entry[0] for entry in self.data]))
        self._chromosome_lengths = list(set([
            entry[1] for entry in self.data]))
        self._proc_counts = list(set([
            entry[2] for entry in self.data]))
        self._times = list(set([
            entry[3] for entry in self.data]))

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def filter(
            self,
            population_size=None, chromosome_length=None, proc_count=None):
        """
        Filter data entries by optionally specified values
        """
        if isinstance(population_size, int):
            population_size = [population_size]

        if isinstance(chromosome_length, int):
            chromosome_length = [chromosome_length]

        if isinstance(proc_count, int):
            proc_count = [proc_count]

        filtered_data = [
            entry
            for entry in self.data
            if (population_size is None or entry[0] in population_size)
            and (chromosome_length is None or entry[1] in chromosome_length)
            and (proc_count is None or entry[2] in proc_count)
        ]
        return PerformanceData(filtered_data)

    @property
    def population_sizes(self):
        return self._population_sizes

    @property
    def chromosome_lengths(self):
        return self._chromosome_lengths

    @property
    def proc_counts(self):
        return self._proc_counts

    @property
    def times(self):
        return self._times

    @staticmethod
    def load(directory):
        """
        Extract performance data from json files in the given directory
        """
        result_sets = sorted(
            read_results(directory),
            key=lambda rs: (
                rs.parameters.population_size,
                rs.parameters.chromosome_length,
                rs.parameters.proc_count
            ))

        data = []
        for rs in result_sets:
            data_point = (
                rs.parameters.population_size,
                rs.parameters.chromosome_length,
                rs.parameters.proc_count,
                # Deal with seconds/iteration
                rs.results.run_time / rs.results.iterations
            )
            data.append(data_point)
        return PerformanceData(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir',
        help='Directory with .json result files')
    parser.add_argument(
        'dir2',
        help='Another directory for validation')
    args = parser.parse_args()

    # Read results from json files
    data = PerformanceData.load(args.dir)
    data2 = PerformanceData.load(args.dir2)

    # Train regression model
    model = PerformanceModel()
    small_dataset = data.filter(
        population_size=[10, 15, 20, 25],
        chromosome_length=[10, 15, 20, 25])
    # model.fit(data)
    model.fit(small_dataset)

    # Now test it on run #2 data
    print model.r2_score(data2)

    # plot_real_predicted(data2, model, 40, 40)

    # Test optimal process counts
    for population_size in xrange(40, 80 + 1, 5):
        for chromosome_length in xrange(40, 80 + 1, 5):
            optimal_pc = model.optimal_proc_count(
                population_size, chromosome_length)
            print population_size, chromosome_length, optimal_pc

    # print optimal_pc
    import pdb; pdb.set_trace()
