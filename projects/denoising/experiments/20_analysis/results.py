#!/usr/bin/env python
#-*- coding: utf-8 -*-
import argparse
import os
from collections import OrderedDict
import numpy as np
from scipy.optimize import curve_fit
import pickle
import json
import pylab as plt
import matplotlib
matplotlib.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
import matplotlib.cm as cm
import projects.denoising.experiments.experiment as exp


# parser = argparse.ArgumentParser()
# parser.add_argument('dir', help='Result directory with all runs')
# args = parser.parse_args()
# # Absolute path
# directory = os.path.abspath(args.dir)


def f_exp(x, a, b, c, d):
    return -a * np.exp(-b * x + c) + d


def f_hyperbolic(x, a, b):
    return a / x + b


def read_all_runs(directory, param_set_start=None, param_set_end=None):
    """
    Read results across all runs for specified param ID range
    Returns dictionary: param_id -> [results_all_runs]
    """
    results = {}
    for dirname in os.listdir(directory):
        dirpath = os.path.join(directory, dirname)
        if os.path.isdir(dirpath) and dirname.startswith('run'):
            # Read result range from directory of single run
            for result_filepath in exp.enumerate_results(dirpath):
                param_set_id = int(
                    os.path.splitext(
                        os.path.basename(result_filepath)
                    )[0])

                # In requested range?
                if param_set_start <= param_set_id <= param_set_end:
                    if param_set_id not in results:
                        results[param_set_id] = []
                    results[param_set_id].append(
                        exp.read_results_file(result_filepath))
    return results


def read_one_run(directory):
    results = {}
    # Read result range from directory of single run
    for result_filepath in exp.enumerate_results(directory):
        param_set_id = int(
            os.path.splitext(
                os.path.basename(result_filepath)
            )[0])

        # In requested range?
        # if param_set_start <= param_set_id <= param_set_end:
        #     if param_set_id not in results:
        #         results[param_set_id] = []
        # print param_set_id
        results[param_set_id] = exp.read_results_file(result_filepath)
    return results


def load_fits(filepath):
    pass


def mean_best_fitness(results):
    """
    Averaged numpy array of best fitnesses per iteration
    for given list of result sets.
    """
    return np.average(np.array([
        [iteration.best_fitness for iteration in result.iterations]
        for result in results
    ]), axis=0)


def var_best_fitness(results):
    """
    Variance between best fitnesses per iteration
    among result sets
    """
    return np.var(np.array([
        [iteration.best_fitness for iteration in result.iterations]
        for result in results
    ]), axis=0)


def mean_avg_fitness(results):
    """
    Same as mean_best_fitness but for average fitnesses
    """
    return np.average(np.array([
        [iteration.average_fitness for iteration in result.iterations]
        for result in results
    ]), axis=0)


def var_avg_fitness(results):
    """
    Same as var_best_fitness but for average fitnesses
    """
    return np.var(np.array([
        [iteration.average_fitness for iteration in result.iterations]
        for result in results
    ]), axis=0)


def fitness_fits(results, fitness_data='best'):
    """
    Do exponential fit for mean best fitnesses of
    result group (same parameter set / all runs).
    Returns dict: param_id -> (fit_popt, fit_pcov)
    """
    fits = {}
    for param_id, results_runs in results.items():
        # Mean of best or average fitnesses across iterations
        if fitness_data == 'best':
            fitness_mean = mean_best_fitness(results_runs)
        elif fitness_data == 'average':
            fitness_mean = mean_avg_fitness(results_runs)
        else:
            raise ValueError(
                'Fitness data type must be either \'best\' or \'average\'')

        iterations = [i + 1 for i in xrange(len(fitness_mean))]
        try:
            popt, pcov = curve_fit(
                f_exp,
                iterations, fitness_mean,
                p0=(1, 1, 1, 1),
                # p0=(0.17, 0.01, 0.13, 0.91),
                maxfev=10000)
        except:
            print "Failed to fit: %i" % (param_id)
            continue
            # fitness_plot_runs(results_runs)

        # Approximated values
        fit_best = [f_exp(i, *popt) for i in iterations]

        # Relative approximation error
        error = np.sum(
            np.absolute(
                np.divide(
                    np.subtract(fitness_mean, fit_best),
                    fitness_mean
                )
            )
        ) / len(fit_best) * 100.0

        fits[param_id] = (popt, pcov, error)
    return fits


def average_error(fits):
    return np.mean([
        fit[2] for fit in fits.values()
    ])







def plot_old_vs_new(old_vs_new_pairs, titles):
    fig = plt.figure(figsize=(10, 3))

    shared_axes = None
    for idx, old_vs_new in enumerate(old_vs_new_pairs):
        if shared_axes is None:
            shared_axes = plt.subplot(1, len(old_vs_new_pairs), idx + 1)
            plt.ylabel(u'Geriausio sprendinio \nfitneso funkcijos reikšmė')
        else:
            plt.subplot(1, len(old_vs_new_pairs), idx + 1, sharey=shared_axes)

        plt.plot(old_vs_new[0], color='red', label='Seni parametrai')
        plt.plot(old_vs_new[1], color='green', label='Nauji parametrai')

        # for idx, values in enumerate(value_lists):
        #     plt.plot(values, label=str(idx))
        plt.xlabel(u'Vykdymo eil. nr.')
        plt.xticks(
            xrange(len(old_vs_new[0])),
            xrange(1, len(old_vs_new[0]) + 1))
        plt.title(titles[idx])
        plt.ylim([0.9, 1.02])
        plt.legend(loc='lower right', prop={'size': 9})
        plt.tight_layout()
        plt.grid()
    plt.show()







def fitness_plot(result):
    best_fitnesses = [it.best_fitness for it in result.iterations]
    avg_fitnesses = [it.average_fitness for it in result.iterations]

    fig = plt.figure(figsize=(14, 7))
    plt.xlabel(u'Iteracija')
    plt.ylabel(u'Fitneso reikšmė')
    plt.ylim([0, 1])
    plt.plot(
        best_fitnesses,
        label='Geriausias sprendinys',
        color='red')
    plt.plot(
        avg_fitnesses,
        label='Populiacijos vidurkis',
        color='blue')
    plt.show()


def fitness_plot_runs(results):
    fig = plt.figure(figsize=(14, 7))
    plt.xlabel(u'Iteracija')
    plt.ylabel(u'Fitneso funkcijos reikšmė')
    plt.ylim([0, 1])
    plt.plot(
        mean_best_fitness(results),
        label=u'Didžiausio fitneso vidurkis',
        color='red')
    plt.plot(
        mean_avg_fitness(results),
        label=u'Vidutinio fitneso vidurkis',
        color='blue')
    plt.legend(loc='lower right')
    plt.show()


def simple_plot_line(values, xlim=None, ylim=None):
    fig = plt.figure(figsize=(14, 7))
    plt.plot(values)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


def simple_plot_lines(value_lists):
    fig = plt.figure(figsize=(4, 3))
    for idx, values in enumerate(value_lists):
        plt.plot(values, label=str(idx))
    plt.ylabel(u'Fitneso funkcijos \nreikšmė')
    plt.xlabel(u'Eil. nr.')
    plt.ylim([0.9, 1.02])
    plt.legend(loc='top left')
    plt.tight_layout()
    plt.show()


def simple_plot_scatter(x_values, y_values):
    fig = plt.figure(figsize=(14, 7))
    plt.scatter(x_values, y_values)
    plt.show()


def simple_plot_scatters(x_value_lists, y_value_lists):
    fig = plt.figure(figsize=(14, 7))
    colors = iter(cm.rainbow(np.linspace(0, 1, len(x_value_lists))))
    for idx in xrange(len(x_value_lists)):
        plt.scatter(
            x_value_lists[idx], y_value_lists[idx],
            color=next(colors),
            label=str(idx))
    plt.ylim([0.8, 1])
    plt.xlim([0, 1])
    plt.legend(loc='bottom right')
    plt.show()



















def plot_exp_params(fits):
    # 2nd exp param - how fast the curve grows
    x_vals = [fit[0][1] for fit in fits.values()]
    # 4th exp param - value into which the curve converges
    y_vals = [fit[0][3] for fit in fits.values()]

    fig = plt.figure(figsize=(14, 7))
    plt.xlabel(u'Kreivės augimo greitis')
    plt.ylabel(u'Maksimali fitneso reikšmė')
    plt.ylim([0.8, 1])
    plt.xlim([0, 1])
    plt.scatter(x_vals, y_vals)
    plt.show()


def plot_fit(results, fit):
    # Averaged best and avg fitnesses over all runs
    mean_best = mean_best_fitness(results)
    mean_avg = mean_avg_fitness(results)
    iterations = [i + 1 for i in xrange(len(mean_best))]
    fit_best = [f_exp(i, *(fit[0])) for i in iterations]

    fig = plt.figure(figsize=(14, 7))
    plt.xlabel(u'Iteracija')
    plt.ylabel(u'Fitneso reikšmė')
    plt.ylim([0, 1.2])
    plt.plot(
        mean_best,
        label='Suvidurkintas geriausias fitnesas')
    plt.plot(
        mean_avg,
        label='Suvidurkintas vidutinis fitnesas')
    plt.plot(
        fit_best,
        label='Geriausio fitneso aproksimacija')
    plt.legend(loc='lower right')
    plt.show()


def fitness_with_fit(results):
    # Averaged best and avg fitnesses over all runs
    mean_best = mean_best_fitness(results)
    mean_avg = mean_avg_fitness(results)

    iterations = [i + 1 for i in xrange(500)]

    # Fit best fitness to exponential function
    popt, pcov = curve_fit(
        f_exp,
        iterations, mean_best, p0=(1, 1, -1, 1), maxfev=2000)
    print popt
    print pcov
    fit_best = [f_exp(i, *popt) for i in iterations]

    # popt, pcov = curve_fit(
    #     f_hyperbolic,
    #     iterations, mean_best, p0=(0, 0), maxfev=2000)
    # print popt
    # fit_best = [f_hyperbolic(i, *popt) for i in iterations]


    # import pdb; pdb.set_trace()

    fig = plt.figure(figsize=(14, 7))
    plt.xlabel(u'Iteracija')
    plt.ylabel(u'Fitneso reikšmė')
    plt.ylim([0, 1.2])
    plt.plot(
        mean_best,
        label='Suvidurkintas geriausias fitnesas')
    plt.plot(
        mean_avg,
        label='Suvidurkintas vidutinis fitnesas')
    plt.plot(
        fit_best,
        label='Geriausio fitneso aproksimacija')
    plt.legend(loc='lower right')
    plt.show()


def filter_by_noise(results_all_runs, noise_type, noise_param):
    return [
        results_runs
        for results_runs in results_all_runs.values()
        if results_runs[0].parameters.noise_type == noise_type
        and results_runs[0].parameters.noise_param == noise_param
    ]


# Population size: small + large
RESULT_GROUP_SIZE = 100
ITERATIONS = 500
# for param_id_start in xrange(500, 1600, RESULT_GROUP_SIZE):
#     results = read_all_runs(
#         directory,
#         param_id_start,
#         param_id_start + RESULT_GROUP_SIZE - 1)
#     fits = fitness_fits(results)
#     print average_error(fits)
#     # plot_exp_params(fits)



# all_fits = {}
# for param_id_start in xrange(0, 6900, RESULT_GROUP_SIZE):
#     results = read_all_runs(
#         directory,
#         param_id_start,
#         param_id_start + RESULT_GROUP_SIZE - 1)
#     fits = fitness_fits(results, fitness_data='average')
#     all_fits.update(fits)
#     print average_error(fits)
# all_results = read_all_runs(directory, 0, 6899)


# pickle_filepath = '/home/tomas/Masters/fits_average.pickle'
# with open(pickle_filepath, 'w') as fp:
#     pickle.dump(all_fits, fp)

# examples = read_all_runs(directory, 0, 99)


def param_sets_by_noise(param_sets, noise_type, noise_param):
    results = {}
    for idx, param_set in enumerate(param_sets):
        if param_set['noise_type'] == noise_type \
                and param_set['noise_param'] == noise_param:
            results[idx] = param_set
    return results


def top_fitnesses_by_noise(top_fitnesses, param_sets, noise_type, noise_param):
    results = {}
    filtered_param_sets = param_sets_by_noise(
        param_sets, noise_type, noise_param)
    for idx in filtered_param_sets.keys():
        results[idx] = top_fitnesses[idx]
    return results


# def multibar_by_noise(top_fitnesses, param_sets, param_type):


# res.parameter_options = {
#     'population_size': [
#         20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
#         250, 300, 350, 400, 450, 500
#     ],
#     'crossover_rate': [
#         0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
#         0.8, 0.85, 0.9, 0.95, 1.0
#     ],
#     'mutation_rate': [
#         0.001, 0.003, 0.005, 0.007, 0.009,
#         0.01, 0.03, 0.05, 0.07, 0.09
#     ],
#     'elite_size': [0, 1, 2, 3, 4, 5],
#     'selection': ['roulette', 'tournament']
# }


def param_sets_by_noises(param_sets):
    return [
        param_sets_by_noise(param_sets, 'snp', 0.1),
        param_sets_by_noise(param_sets, 'snp', 0.3),
        param_sets_by_noise(param_sets, 'snp', 0.5),
        param_sets_by_noise(param_sets, 'gaussian', 0.1),
        param_sets_by_noise(param_sets, 'gaussian', 0.3),
        param_sets_by_noise(param_sets, 'gaussian', 0.5),
    ]


def multibar_param_set_plot(
        param_set_lists,
        param_set_list_names,
        param_choices, param_type, param_name,
        rotate_xlabels=False):
    choice_count = len(param_choices[param_type])
    # fig = plt.figure(figsize=(len(param_set_lists) * 3, 3))
    fig = plt.figure(figsize=(9, 2.5))

    param_value_histograms = []
    for param_set_list_idx, param_set_list in enumerate(param_set_lists):
        param_value_histogram = [0] * choice_count
        for param_set_tuple in param_set_list:
            param_value = param_set_tuple[1][param_type]
            option_idx = list(param_choices[param_type]).index(param_value)
            param_value_histogram[option_idx] += 1

        param_value_histograms.append(param_value_histogram)

    y_max = max([max(vals) for vals in param_value_histograms])

    # Share y-axis of leftmost plot with other plots
    shared_axes = None
    for idx, param_value_histogram in enumerate(param_value_histograms):
        if shared_axes is None:
            shared_axes = plt.subplot(1, len(param_value_histograms), idx + 1)
            plt.ylabel(u'Parametrų rinkinių\nskaičius')
        else:
            plt.subplot(1, len(param_value_histograms), idx + 1, sharey=shared_axes)
        plt.ylim([0, y_max])
        # plt.ylabel(u'Parametrų rinkinių skaičius')
        plt.bar(
            xrange(choice_count), param_value_histogram,
            align='center')
        if rotate_xlabels is True:
            plt.xticks(
                xrange(choice_count), param_choices[param_type],
                ha='center',
                rotation=90)
        else:
            plt.xticks(
                xrange(choice_count), param_choices[param_type],
                ha='center')

        if param_set_list_names:
            plt.title(param_set_list_names[idx])

        # X label at the bottom
        plt.xlabel(param_name)
    plt.tight_layout()
    plt.show()


def top_fitnesses(result_dir):
    fitnesses = []
    for param_id_start in xrange(0, 6900, RESULT_GROUP_SIZE):
        # param_id_start = group_index * RESULT_GROUP_SIZE
        # param_id_end = (group_index + 1) * RESULT_GROUP_SIZE
        param_id_end = param_id_start + RESULT_GROUP_SIZE
        group_results = read_all_runs(
            result_dir, param_id_start, param_id_end)
        for param_id in xrange(param_id_start, param_id_end):
            # Top fitness (last iteration) of averaged best fitness data
            fitnesses.append(mean_best_fitness(
                group_results[param_id])[ITERATIONS - 1])
        print param_id_start
    return fitnesses


def fit_params(fits, group_index, fit_param_index, sort=True):
    idx_start = group_index * RESULT_GROUP_SIZE
    idx_end = (group_index + 1) * RESULT_GROUP_SIZE
    params = []
    for idx in xrange(idx_start, idx_end):
        if idx in fits:
            params.append(fits[idx][0][fit_param_index])
    if sort is True:
        params = sorted(params)
    return params


def fit_params_groups(fits, group_indexes, fit_param_index, sort=True):
    return [
        fit_params(fits, group_index, fit_param_index, sort)
        for group_index in group_indexes
    ]


def fit_params_groups_median(fits, group_indexes, fit_param_index, sort=True):
    return [
        np.median(fit_params(fits, group_index, fit_param_index, sort))
        for group_index in group_indexes
    ]


fits_best_filepath = u'/home/tomas/Masters/fits_best.pickle'
fits_average_filepath = u'/home/tomas/Masters/fits_average.pickle'
with open(fits_best_filepath, 'r') as fp:
    fits_best = pickle.load(fp)
with open(fits_average_filepath, 'r') as fp:
    fits_average = pickle.load(fp)


def plots_separate_noises(fits, parameter_sets, group_index_start, group_index_end):
    for group_idx in xrange(group_index_start, group_index_end):
        # Exp model params for that group (100 param sets)
        # params_b = fit_params(fits_best, group_idx, 1)
        # params_d = fit_params(fits_best, group_idx, 3)
        param_set_start = group_idx * RESULT_GROUP_SIZE
        param_set_end = (group_idx + 1) * RESULT_GROUP_SIZE

        snp_01_b, snp_03_b, snp_05_b = [], [], []
        snp_01_d, snp_03_d, snp_05_d = [], [], []
        gaussian_01_b, gaussian_03_b, gaussian_05_b = [], [], []
        gaussian_01_d, gaussian_03_d, gaussian_05_d = [], [], []
        # Group by noise
        for param_set_idx in xrange(param_set_start, param_set_end):
            if param_set_idx in fits:
                ps = parameter_sets[param_set_idx]
                model_param_b = fits[param_set_idx][0][1]
                model_param_d = fits[param_set_idx][0][3]
                if ps['noise_type'] == 'snp':
                    if ps['noise_param'] == 0.1:
                        snp_01_b.append(model_param_b)
                        snp_01_d.append(model_param_d)
                    elif ps['noise_param'] == 0.3:
                        snp_03_b.append(model_param_b)
                        snp_03_d.append(model_param_d)
                    elif ps['noise_param'] == 0.5:
                        snp_05_b.append(model_param_b)
                        snp_05_d.append(model_param_d)
                    else:
                        raise ValueError
                elif ps['noise_type'] == 'gaussian':
                    if ps['noise_param'] == 0.1:
                        gaussian_01_b.append(model_param_b)
                        gaussian_01_d.append(model_param_d)
                    elif ps['noise_param'] == 0.3:
                        gaussian_03_b.append(model_param_b)
                        gaussian_03_d.append(model_param_d)
                    elif ps['noise_param'] == 0.5:
                        gaussian_05_b.append(model_param_b)
                        gaussian_05_d.append(model_param_d)
                    else:
                        raise ValueError
                else:
                    raise ValueError

        for_plot_x = [
            snp_01_b,
            snp_03_b,
            snp_05_b,
            gaussian_01_b,
            gaussian_03_b,
            gaussian_05_b
        ]
        for_plot_y = [
            snp_01_d,
            snp_03_d,
            snp_05_d,
            gaussian_01_d,
            gaussian_03_d,
            gaussian_05_d
        ]
        simple_plot_scatters(for_plot_x, for_plot_y)


def quartile_plot(
        fits,
        group_index_start, group_index_end,
        model_param_index,
        ylim=None,
        log=True,
        xlabel=None,
        ylabel=None,
        labels=None):
    model_param_values = [
        fit_params(fits, group_index, model_param_index)
        for group_index in xrange(
            group_index_start, group_index_end)
    ]
    fig = plt.figure(figsize=(len(model_param_values), 7))
    if log is True:
        plt.yscale('log')
    if ylim is not None:
        plt.ylim(ylim)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.boxplot(
        model_param_values,
        labels=labels,
        showmeans=True)
    plt.grid()
    plt.show()



# def best_param_sets(fits, count=100):



# pop_size_20 = read_all_runs(directory, 0, 99)
# pop_20_fits = fitness_fits(pop_size_20)
# print average_error(pop_20_fits)
# plot_exp_params(pop_20_fits)

# param1_median = np.median(
#     [f[0][0] for f in pop_20_fits.values()])
# param2_median = np.median(
#     [f[0][1] for f in pop_20_fits.values()])
# param3_median = np.median(
#     [f[0][2] for f in pop_20_fits.values()])
# param4_median = np.median(
#     [f[0][3] for f in pop_20_fits.values()])



# noise_snp_01 = filter_by_noise(pop_size_20, 'snp', 0.1)
# noise_snp_03 = filter_by_noise(pop_size_20, 'snp', 0.3)
# noise_snp_05 = filter_by_noise(pop_size_20, 'snp', 0.5)
# noise_gaussian_01 = filter_by_noise(pop_size_20, 'gaussian', 0.1)
# noise_gaussian_03 = filter_by_noise(pop_size_20, 'gaussian', 0.3)
# noise_gaussian_05 = filter_by_noise(pop_size_20, 'gaussian', 0.5)

# fitness_plot(pop_size_20[0][0])
# fitness_with_fit(pop_size_20[0])





# import pdb; pdb.set_trace()
