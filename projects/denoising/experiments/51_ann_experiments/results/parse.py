#!/usr/bin/env python
#-*- coding: utf-8 -*-
import re
import os
import json
import csv
import numpy as np
import itertools
from scipy.optimize import curve_fit
from scipy import stats
from fann2 import libfann
from skimage import io, util
from projects.denoising.neural.filtering import filter_fann
from projects.denoising.imaging.metrics import q_py, ocr_accuracy, mse

import pylab as plt
import matplotlib as mpl
mpl.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D


OCR_LEVEL_FILES = {
    'high': '/home/tomas/Masters/4_semester/synthetic_tests/high_ocr.txt',
    'medium': '/home/tomas/Masters/4_semester/synthetic_tests/medium_ocr.txt',
    'low': '/home/tomas/Masters/4_semester/synthetic_tests/low_ocr.txt',
}


def results_to_initial(results):
    initial = []
    for result in results:
        if result['image']['name'].startswith('noisy-00-00-'):
            continue
        try:
            result_initial = next(
                r for r in initial if r['image']['name'] == result['image']['name'])
        except StopIteration:
            result_initial = {
                'image': result['image'],
                'metrics_initial': result['metrics_initial'],
                'metrics_absolute': {
                    'q': [result['metrics_initial']['q']],
                    'ocr': [result['metrics_initial']['ocr']],
                    'mse': [result['metrics_initial']['mse']],
                }
            }
            initial.append(result_initial)
    return initial


def filter_ocr_level(results, level):
    path = OCR_LEVEL_FILES[level]
    with open(path, 'r') as fp:
        image_names = json.load(fp)
    return [
        result
        for result in results
        if result['image']['name'] in image_names
    ]


def read_results(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data


def _parse_image_name(image_name):
    """
    Split image name into its parameter values
    """
    blur_values = {'00': 0.0, '05': 0.5, '10': 1.0, '15': 1.5, '20': 2.0}
    noise_values = {'00': 0.0, '001': 0.01, '002': 0.02, '003': 0.03, '004': 0.04, '005': 0.05}
    contrast_values = {'02': 0.2, '04': 0.4, '06': 0.6, '08': 0.8, '10': 1.0}
    parts = image_name.split('-')
    return blur_values[parts[1]], noise_values[parts[2]], contrast_values[parts[3]]


def _read_clear_images():
    clear_images = {}
    clear_image_dir = '/home/tomas/Masters/4_semester/synthetic_tests/clear/'
    for clear_image_file in os.listdir(clear_image_dir):
        if clear_image_file.endswith('.png'):
            clear_image_name = os.path.splitext(clear_image_file)[0]
            clear_image_path = os.path.join(clear_image_dir, clear_image_file)
            clear_image = util.img_as_float(io.imread(clear_image_path))
            clear_images[clear_image_name] = clear_image
    return clear_images


def run_initial(noisy_image_dir, output_file):
    DEFAULT_TEXT = 'Lorem ipsum\ndolor sit amet,\nconsectetur\n\nadipiscing elit.\n\nDonec vel\naliquet velit,\nid congue\nposuere.'
    fieldnames = ['image', 'q', 'ocr', 'mse']
    csv_file = open(output_file, 'w')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    clear_images = _read_clear_images()
    for noisy_image_file in sorted(os.listdir(noisy_image_dir)):
        if noisy_image_file.endswith('.png'):
            image = util.img_as_float(io.imread(
                os.path.join(noisy_image_dir, noisy_image_file)))
            image_name = os.path.splitext(noisy_image_file)[0]
            # All three metrics
            q = q_py(image)
            ocr = ocr_accuracy(image, DEFAULT_TEXT)
            _, __, contrast = _parse_image_name(image_name)
            contrast_str = str(contrast).replace('.', '')
            clear_image_name = 'clear-00-00-' + contrast_str
            mse_val = mse(clear_images[clear_image_name], image)

            result_row = {
                'image': image_name,
                'q': q,
                'ocr': ocr,
                'mse': mse_val
            }
            writer.writerow(result_row)
    csv_file.close()
    return None


def run_mse(result_dir, output_file):
    csv_file = open(output_file, 'w')
    # Read clear images
    clear_images = _read_clear_images()


    # fieldnames = ['image', 'param_set', 'final_error', 'training_time']
    fieldnames = ['image', 'param_set', 'mse']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    # Enumerate images
    for image_name in sorted(os.listdir(result_dir)):
        image_dir = os.path.join(result_dir, image_name)
        if os.path.isdir(image_dir):
            print image_name
            # Enumerate parameter sets
            for result_file in sorted(os.listdir(image_dir)):
                if result_file.endswith('.png'):
                    image_path = os.path.join(image_dir, result_file)
                    image = util.img_as_float(io.imread(image_path))
                    # q = q_py(image)
                    _, __, contrast = _parse_image_name(os.path.splitext(image_name)[0])
                    contrast_str = str(contrast).replace('.', '')
                    clear_image_name = 'clear-00-00-' + contrast_str
                    mse_val = mse(clear_images[clear_image_name], image)

                    # result_json_path = os.path.join(image_dir, result_json)
                    result_ps_name = os.path.splitext(result_file)[0]
                    # result_data = read_results(result_json_path)
                    # # last_epoch_error = float(parse_epochs(result_data)[-1]['error'])
                    # last_epoch_error = float(parse_epochs(result_data)[-1])
                    # # Write into csv file
                    result_row = {
                        'image': image_name,
                        'param_set': result_ps_name,
                        'mse': mse_val,
                    }
                    writer.writerow(result_row)
    csv_file.close()
    return None


def run_ocr(result_dir, output_file):
    DEFAULT_TEXT = 'Lorem ipsum\ndolor sit amet,\nconsectetur\n\nadipiscing elit.\n\nDonec vel\naliquet velit,\nid congue\nposuere.'
    csv_file = open(output_file, 'w')
    fieldnames = ['image', 'param_set', 'ocr']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    # Enumerate images
    for image_name in sorted(os.listdir(result_dir)):
        image_dir = os.path.join(result_dir, image_name)
        if os.path.isdir(image_dir):
            print image_name
            # Enumerate parameter sets
            for result_file in sorted(os.listdir(image_dir)):
                if result_file.endswith('.png'):
                    image_path = os.path.join(image_dir, result_file)
                    image = util.img_as_float(io.imread(image_path))
                    ocr = ocr_accuracy(image, DEFAULT_TEXT)

                    result_ps_name = os.path.splitext(result_file)[0]
                    # # Write into csv file
                    result_row = {
                        'image': image_name,
                        'param_set': result_ps_name,
                        'ocr': ocr,
                    }
                    writer.writerow(result_row)
    csv_file.close()
    return None


def run_q(result_dir, output_file):
    csv_file = open(output_file, 'w')
    # fieldnames = ['image', 'param_set', 'final_error', 'training_time']
    fieldnames = ['image', 'param_set', 'q']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    # Enumerate images
    for image_name in sorted(os.listdir(result_dir)):
        image_dir = os.path.join(result_dir, image_name)
        if os.path.isdir(image_dir):
            print image_name
            # Enumerate parameter sets
            for result_file in sorted(os.listdir(image_dir)):
                if result_file.endswith('.png'):
                    image_path = os.path.join(image_dir, result_file)
                    image = util.img_as_float(io.imread(image_path))
                    q = q_py(image)

                    # result_json_path = os.path.join(image_dir, result_json)
                    result_ps_name = os.path.splitext(result_file)[0]
                    # result_data = read_results(result_json_path)
                    # # last_epoch_error = float(parse_epochs(result_data)[-1]['error'])
                    # last_epoch_error = float(parse_epochs(result_data)[-1])
                    # # Write into csv file
                    result_row = {
                        'image': image_name,
                        'param_set': result_ps_name,
                        'q': q,
                    }
                    writer.writerow(result_row)
    csv_file.close()
    return None


def recreate_images(result_dir, noisy_image_dir):
    # Read noisy images first
    test_images = {}
    for image_name in os.listdir(noisy_image_dir):
        if image_name.endswith('.png'):
            image_path = os.path.join(noisy_image_dir, image_name)
            image = util.img_as_float(io.imread(image_path))
            image_name_noext = os.path.splitext(image_name)[0]
            test_images[image_name_noext] = image
    # Enumerate results - image directories
    for image_name in sorted(os.listdir(result_dir)):
        image_dir = os.path.join(result_dir, image_name)
        if os.path.isdir(image_dir):
            print image_name
            for result_file in sorted(os.listdir(image_dir)):
                if result_file.endswith('.net'):
                    # Instantiate trained ANN from .net file
                    net_path = os.path.join(image_dir, result_file)
                    ann = libfann.neural_net()
                    ann.create_from_file(net_path)
                    # Filter the same image which it was trained with
                    filtered_image = filter_fann(
                        test_images[image_name], ann)
                    param_set_name = os.path.splitext(result_file)[0]
                    io.imsave(
                        os.path.join(image_dir, param_set_name + '.png'),
                        filtered_image)


def parse_results(result_dir, output_file):
    csv_file = open(output_file, 'w')
    # fieldnames = ['image', 'param_set', 'final_error', 'training_time']
    fieldnames = ['image', 'param_set', 'final_error']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    # Enumerate images
    for image_name in sorted(os.listdir(result_dir)):
        image_dir = os.path.join(result_dir, image_name)
        if os.path.isdir(image_dir):
            print image_name
            # Enumerate parameter sets
            for result_json in sorted(os.listdir(image_dir)):
                if result_json.endswith('.json'):
                    result_json_path = os.path.join(image_dir, result_json)
                    result_ps_name = os.path.splitext(result_json)[0]
                    result_data = read_results(result_json_path)
                    # last_epoch_error = float(parse_epochs(result_data)[-1]['error'])
                    last_epoch_error = float(parse_epochs(result_data)[-1])
                    # Write into csv file
                    result_row = {
                        'image': image_name,
                        'param_set': result_ps_name,
                        'final_error': last_epoch_error,
                        # 'training_time': result_data['training_time']
                    }
                    writer.writerow(result_row)
    csv_file.close()
    return None


def _parse_ann_info(image_res_dir):
    """
    Input - directory of a single image
    """
    results = {}
    for result_file in os.listdir(image_res_dir):
        if result_file.endswith('.net'):
            ann_path = os.path.join(image_res_dir, result_file)
            param_set = os.path.splitext(result_file)[0]
            ann = libfann.neural_net()
            ann.create_from_file(ann_path)
            results[param_set] = {}
            results[param_set]['connection_count'] = len(ann.get_connection_array())
    return results


def aggregate_parsed_results(result_dir, param_dir, metrics_initial_path):
    results = []
    result_files = sorted(
        [
            name for name in os.listdir(result_dir)
            if name.endswith('.csv')
        ],
        key=lambda n: int(n.split('_')[0].split('.')[0]))

    for result_file in result_files:
        print result_file
        filename = os.path.splitext(result_file)[0]
        filepath = os.path.join(result_dir, result_file)
        csv_file = open(filepath, 'r')
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            # Get existing or create new result row
            try:
                result = next(
                    result_row for result_row in results
                    if result_row['image']['name'] == row['image']
                    and result_row['param_set']['name'] == row['param_set'])
            except (StopIteration, KeyError):
                blur, noise, contrast = _parse_image_name(row['image'])
                param_set_filepath = os.path.join(
                    param_dir, row['param_set'] + '.json')
                with open(param_set_filepath, 'r') as fp:
                    param_set_json = json.load(fp)
                # print row
                if 'hidden_neurons2' not in param_set_json:
                    param_set_json['hidden_neurons2'] = 0
                result = {
                    'image': {
                        'name': row['image'],
                        'blur': blur,
                        'noise': noise,
                        'contrast': contrast
                    },
                    'param_set': {
                        'name': row['param_set'],
                        'input_size': param_set_json['patch_size'],
                        'hidden_neurons': param_set_json['hidden_neurons'],
                        'hidden_neurons2': param_set_json['hidden_neurons2'],
                        'activation_function': param_set_json['activation_function']
                    },
                    'metrics_absolute': {
                        'q': [],
                        'ocr': [],
                        'mse': []
                    }
                }
                results.append(result)
            # Q
            if filename.endswith('_q'):
                result['metrics_absolute']['q'].append(float(row['q']))
            # OCR
            elif filename.endswith('_ocr'):
                result['metrics_absolute']['ocr'].append(float(row['ocr']))
            # MSE
            elif filename.endswith('_mse'):
                result['metrics_absolute']['mse'].append(float(row['mse']))
            else:
                result['metrics_absolute']['training_error'] = float(row['final_error'])
        csv_file.close()

    # Add initial metric values
    csv_file = open(metrics_initial_path, 'r')
    csv_reader = csv.DictReader(csv_file)
    metrics_initial = {}
    for row in csv_reader:
        metrics_initial[row['image']] = {
            'q': float(row['q']),
            'ocr': float(row['ocr']),
            'mse': float(row['mse'])
        }
    csv_file.close()

    clear_images = _read_clear_images()
    for result in results:
        result['metrics_initial'] = metrics_initial[result['image']['name']]
        result['metrics_relative'] = {}

        # Relative values
        q_noisy = result['metrics_initial']['q']
        clear_image_name = 'noisy-00-00-' + str(result['image']['contrast']).replace('.', '')
        q_clear = metrics_initial[clear_image_name]['q']

        result['metrics_relative']['q'] = []
        for q_filtered in result['metrics_absolute']['q']:
            q1 = q_filtered - q_noisy
            q2 = q_clear - q_noisy
            if q2 == 0.0:
                result['metrics_relative']['q'].append(None)
            else:
                result['metrics_relative']['q'].append(q1 / q2)

        ocr_noisy = result['metrics_initial']['ocr']
        result['metrics_relative']['ocr'] = [
            # (ocr - ocr_initial) / ocr_initial
            ocr - ocr_noisy
            for ocr in result['metrics_absolute']['ocr']
        ]

        # MSE is different - measure decrease, not increase!
        result['metrics_relative']['mse'] = []
        mse_noisy = result['metrics_initial']['mse']
        for mse_filtered in result['metrics_absolute']['mse']:
            if mse_noisy == 0.0:
                result['metrics_relative']['mse'].append(None)
            else:
                result['metrics_relative']['mse'].append(
                    (mse_noisy - mse_filtered) / mse_noisy)

    return results


def param_set_averages(results, metric='ocr'):
    values = {}
    for result in results:
        if result['image']['blur'] == 0.0 and result['image']['noise'] == 0.0:
            continue
        if result['param_set']['name'] not in values:
            values[result['param_set']['name']] = []
        if metric == 'ocr':
            values[result['param_set']['name']] += result['metrics_relative']['ocr']
        elif metric == 'q':
            values[result['param_set']['name']] += result['metrics_relative']['q']
        elif metric == 'mse':
            values[result['param_set']['name']] += result['metrics_relative']['mse']
        else:
            raise ValueError
    averages = {}
    for key, value_list in values.items():
        averages[key] = np.mean(value_list)
    return averages


def param_set_averages_plot(results):
    averages_ocr = [
        a[1] for a in sorted(
            param_set_averages(results, metric='ocr').items(),
            key=lambda x: int(x[0].split('-')[1]))
    ]
    averages_q = [
        a[1] for a in sorted(
            param_set_averages(results, metric='q').items(),
            key=lambda x: int(x[0].split('-')[1]))
    ]
    averages_mse = [
        a[1] for a in sorted(
            param_set_averages(results, metric='mse').items(),
            key=lambda x: int(x[0].split('-')[1]))
    ]
    fig = plt.figure(figsize=(6, 4))
    # plt.tight_layout()
    plt.plot(averages_ocr, label='OCR', linewidth=2.0)
    plt.plot(averages_q, label='Q', linewidth=2.0)
    plt.plot(averages_mse, label='MSE', linewidth=2.0)
    plt.ylim([0, 1])
    plt.xlabel(u'Paslėptų neuronų skaičius')
    plt.ylabel(u'Vidurinė Q įverčio pokyčio reikšmė')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.show()


def best_param_sets(results, metric):
    """
    Best param set for each image, determined by averaged
    specified metric of all 10 runs
    """
    image_results = {}
    for result in results:
        image_name = result['image']['name']
        if result['image']['blur'] == 0.0 and result['image']['noise'] == 0.0:
            continue
        if image_name not in image_results:
            image_results[image_name] = {}
        ps_name = result['param_set']['activation_function']
        ps_name += '-' + result['param_set']['name']
        metric_avg = np.mean(result['metrics_relative'][metric])
        image_results[image_name][ps_name] = metric_avg

    best_results = {}
    for image_name, averages in image_results.items():
        best_param_set = sorted(
            averages.items(), key=lambda p: p[1], reverse=True)[0]
        best_results[image_name] = best_param_set

    grouped = {}
    for image_name, best in best_results.items():
        ps_name = best[0]
        if ps_name not in grouped:
            grouped[ps_name] = 0
        else:
            grouped[ps_name] += 1
    # return best_results
    return grouped
    # return image_results['noisy-00-001-02']


def bad_results(results):
    bad = []
    for result in results:
        for run in xrange(10):
            q = result['metrics_relative']['q'][run]
            mse = result['metrics_relative']['mse'][run]
            ocr = result['metrics_absolute']['ocr'][run]
            if q > 1.0 and mse < 0.0:
                ps_name = result['param_set']['activation_function']
                ps_name += '-' + result['param_set']['name']
                bad.append({
                    'image_name': result['image']['name'],
                    'param_set': ps_name,
                    'run': run + 1,
                    'p_q': q,
                    'p_mse': mse,
                    'p_ocr': ocr,
                })
    return bad


def annotate_group(name, xspan, ax=None):
    """Annotates a span of the x-axis"""
    def annotate(ax, name, left, right, y, pad):
        arrow = ax.annotate(name,
                xy=(left, y), xycoords='data',
                xytext=(right, y-pad), textcoords='data',
                annotation_clip=False, verticalalignment='top',
                horizontalalignment='center', linespacing=2.0,
                arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0,
                        connectionstyle='angle,angleB=90,angleA=0,rad=5')
                )
        return arrow
    if ax is None:
        ax = plt.gca()
    ymin = ax.get_ylim()[0]
    ypad = 0.01 * np.ptp(ax.get_ylim())
    xcenter = np.mean(xspan)
    left_arrow = annotate(ax, name, xspan[0], xcenter, ymin, ypad)
    right_arrow = annotate(ax, name, xspan[1], xcenter, ymin, ypad)
    return left_arrow, right_arrow


def make_second_bottom_spine(ax=None, label=None, offset=0, labeloffset=20):
    """Makes a second bottom spine"""
    if ax is None:
        ax = plt.gca()
    second_bottom = mpl.spines.Spine(ax, 'bottom', ax.spines['bottom']._path)
    second_bottom.set_position(('outward', offset))
    ax.spines['second_bottom'] = second_bottom

    if label is not None:
        # Make a new xlabel
        ax.annotate(label, 
                xy=(0.5, 0), xycoords='axes fraction', 
                xytext=(0, -labeloffset), textcoords='offset points', 
                verticalalignment='top', horizontalalignment='center')



def plot_average_ocr(results, loc='lower left'):
    """
    Averaged actual OCR values per param set
    """
    ocr_values = {}
    for result in results:
        # Exclude clear images
        if result['image']['name'].startswith('noisy-00-00-'):
            continue
        ps_name = result['param_set']['activation_function']
        ps_name += '-' + result['param_set']['name']
        if ps_name not in ocr_values:
            ocr_values[ps_name] = []
        ocr_values[ps_name] += result['metrics_absolute']['ocr']
    ocr_averages = {}
    for ps_name, values in ocr_values.items():
        # ocr_averages[ps_name] = (np.mean(values), np.var(values))
        ocr_averages[ps_name] = np.mean(values)

    gaussian_averages, sigmoid_averages = {}, {}
    for key, avg in ocr_averages.items():
        if key.startswith('gaussian'):
            gaussian_averages[key] = avg
        else:
            sigmoid_averages[key] = avg
    gaussian_averages = sorted(
        gaussian_averages.items(),
        key=lambda x: int(x[0].split('-')[2]))
    gaussian_averages = [a[1] for a in gaussian_averages]
    sigmoid_averages = sorted(
        sigmoid_averages.items(),
        key=lambda x: int(x[0].split('-')[2]))
    sigmoid_averages = [a[1] for a in sigmoid_averages]



    initial_qs = {}
    for result in results:
        initial_qs[result['image']['name']] = result['metrics_initial']['ocr']
    initial_q_avg = np.mean(initial_qs.values())
    # baseline = [0.67694388100067615] * len(gaussian_averages)
    baseline = [initial_q_avg] * len(gaussian_averages)


    fig = plt.figure(figsize=(6, 5))
    # http://stackoverflow.com/questions/3918028/how-do-i-plot-multiple-x-or-y-axes-in-matplotlib
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_position(('outward', 40))
    make_second_bottom_spine(label='Filtravimo lango dydis')
    groups = [
        ('3x3', (0, 13)),
        ('5x5', (14, 27)),
        ('7x7', (28, 41)),
        ('9x9', (42, 55)),
    ]
    for name, xspan in groups:
        annotate_group(name, xspan)

    major_ticks = np.arange(0, 56, 14)
    minor_ticks = np.arange(0, 56, 1)
    ax.set_xticks(major_ticks)                                                       
    ax.set_xticks(minor_ticks, minor=True)

    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # ax.set_yticks(minor_ticks, minor=True)

    # and a corresponding grid                                                       

    ax.grid(which='both')                                                            

    # or if you want differnet settings for the grids:                               
    ax.grid(which='minor', alpha=0.2)                                                
    ax.grid(which='major', alpha=0.5)  

    # plt.tight_layout()
    plt.plot(gaussian_averages, label='DNT su Gauso a.f.', linewidth=2)
    plt.plot(sigmoid_averages, label='DNT su sigmoidine a.f.', linewidth=2)
    plt.plot(baseline, label='Pradinis vidurkis', linewidth=2)
    plt.legend(loc=loc)
    plt.ylim([0.0, 1])
    plt.xlabel(u'DNT topologijos eil. nr.')
    plt.ylabel(u'Vaizdų rinkinio OCR vidurkis')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return gaussian_averages, sigmoid_averages


def plot_ann_connections(result_dir):
    connections = []
    for filename in os.listdir(result_dir):
        if not filename.endswith('.net'):
            continue
        set_id = int(os.path.splitext(filename)[0].split('-')[1])
        filepath = os.path.join(result_dir, filename)
        ann = libfann.neural_net()
        ann.create_from_file(filepath)
        connections.append((set_id, len(ann.get_connection_array())))
        print set_id, len(ann.get_connection_array())
    connections = sorted(connections, key=lambda x: x[0])
    # import pdb; pdb.set_trace()

    fig = plt.figure(figsize=(6, 4))
    # http://stackoverflow.com/questions/3918028/how-do-i-plot-multiple-x-or-y-axes-in-matplotlib
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_position(('outward', 40))
    make_second_bottom_spine(label='Filtravimo lango dydis')
    groups = [
        ('3x3', (0, 13)),
        ('5x5', (14, 27)),
        ('7x7', (28, 41)),
        ('9x9', (42, 55)),
    ]
    for name, xspan in groups:
        annotate_group(name, xspan)

    major_ticks = np.arange(0, 56, 14)
    minor_ticks = np.arange(0, 56, 1)
    ax.set_xticks(major_ticks)                                                       
    ax.set_xticks(minor_ticks, minor=True)

    # ax.set_yticks(np.arange(0, 2100, 500))

    ax.grid(which='both')                                                            

    # or if you want differnet settings for the grids:                               
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    # plt.tight_layout()
    plt.plot([
        c[1] for c in connections
    ], label='DNT su Gauso a.f.', linewidth=2)
    # plt.legend(loc=loc)
    # plt.ylim([0.0, 1])
    # plt.ylim([0, 2100])
    plt.xlabel(u'DNT topologijos eil. nr.')
    plt.ylabel(u'Jungčių skaičius')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return connections


def read_initial(metrics_initial_path, exclude_clear_images=False):
    # Add initial metric values
    csv_file = open(metrics_initial_path, 'r')
    csv_reader = csv.DictReader(csv_file)
    metrics_initial = {}
    for row in csv_reader:
        b, n, c = _parse_image_name(row['image'])
        if exclude_clear_images == True:
            if b == 0.0 and n == 0.0:
                continue
        metrics_initial[row['image']] = {
            'q': float(row['q']),
            'ocr': float(row['ocr']),
            'mse': float(row['mse'])
        }
    csv_file.close()
    return metrics_initial


def hist_ocr_noisy(metrics_initial_path):
    csv_file = open(metrics_initial_path, 'r')
    csv_reader = csv.DictReader(csv_file)
    ocr_values = []
    for row in csv_reader:
        if not row['image'].startswith('noisy-00-00-'):
            ocr_values.append(float(row['ocr']))
    csv_file.close()
    # Draw histogram
    fig = plt.figure(figsize=(4, 3))
    # plt.tight_layout()
    plt.hist(ocr_values)
    plt.ylim([0, 145])
    plt.xlabel(u'OCR tikslumo įvertis')
    plt.ylabel(u'Vaizdų kiekis')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def hist_ocr_filtered(result_csv_path, param_set):
    csv_file = open(result_csv_path, 'r')
    csv_reader = csv.DictReader(csv_file)
    ocr_values = []
    for row in csv_reader:
        if not row['image'].startswith('noisy-00-00-'):
            if row['param_set'] == param_set:
                ocr_values.append(float(row['ocr']))
    csv_file.close()
    # Draw histogram
    fig = plt.figure(figsize=(4, 3))
    # plt.tight_layout()
    plt.hist(ocr_values)
    plt.ylim([0, 145])
    plt.xlabel(u'OCR tikslumo įvertis')
    plt.ylabel(u'Vaizdų kiekis')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# def connection_counts(result_image_dir):
#     results = []
#     for result_file in os.listdir(result_image_dir):
#         if os.path.splitext(result_file)[0] == '.net':



def plot1(results):
    points = []
    for result in results:
        # Get or create point
        try:
            point = next(
                p for p in points
                if p['param_set'] == result['param_set']['name'])
        except StopIteration:
            point = {
                'param_set': result['param_set']['name'],
                'patch_size': result['param_set']['input_size'],
                'hidden_neurons': result['param_set']['hidden_neurons'],
                'q': [],
                'ocr': [],
                'mse': [],
            }
            points.append(point)
        point['q'] += result['metrics_relative']['q']
        point['ocr'] += result['metrics_relative']['ocr']
        point['mse'] += result['metrics_relative']['mse']

    # Median values for each param set
    for point in points:
        point['q'] = [p for p in point['q'] if p is not None]
        point['ocr'] = [p for p in point['ocr'] if p is not None]
        point['mse'] = [p for p in point['mse'] if p is not None]
        point['q_median'] = np.median(point['q'])
        point['ocr_median'] = np.median(point['ocr'])
        point['mse_median'] = np.median(point['mse'])
        # point['q_mean'] = np.mean(point['q'])
        # point['ocr_mean'] = np.mean(point['ocr'])
        # point['mse_mean'] = np.mean(point['mse'])
        del point['q']
        del point['ocr']
        del point['mse']

    points = sorted(
        points, key=lambda p: int(p['param_set'].split('-')[1]))

    patch_size_values = sorted(list(set([p['patch_size'] for p in points])))
    # hidden_neuron_values = sorted(list(set([p['hidden_neurons'] for p in points])))

    fig = plt.figure(figsize=(6, 4))
    # plt.tight_layout()
    for patch_size in patch_size_values:
        plt.plot(
            [point['q_median'] for point in points if point['patch_size'] == patch_size],
            label='Lango dydis: ' + str(patch_size),
            linewidth=2.0)
    plt.legend(loc='lower right')
    # plt.xticks(
    #     list(xrange(len(hidden_neuron_values))),
    #     hidden_neuron_values)
    plt.ylim([0, 1])
    plt.xlabel(u'Paslėptų neuronų skaičius')
    plt.ylabel(u'Vidurinė Q įverčio pokyčio reikšmė')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    fig = plt.figure(figsize=(6, 4))
    for patch_size in patch_size_values:
        plt.plot(
            [point['mse_median'] for point in points if point['patch_size'] == patch_size],
            label='Lango dydis: ' + str(patch_size),
            linewidth=2.0)
    plt.legend(loc='lower right')
    # plt.xticks(
    #     list(xrange(len(hidden_neuron_values))),
    #     hidden_neuron_values)
    plt.ylim([0, 1])
    plt.xlabel(u'Paslėptų neuronų skaičius')
    plt.ylabel(u'Vidurinė MSE įverčio pokyčio reikšmė')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # plot_points = [
    #     # (point['patch_size'], point['hidden_neurons'], point['q_median'])
    #     (point['patch_size'], point['hidden_neurons'], point['mse_median'])
    #     for point in points
    # ]
    # x, y, z = zip(*plot_points)
    # xs = sorted(list(set(x)))
    # ys = sorted(list(set(y)))

    # X, Y = np.meshgrid(xs, ys)
    # Z = np.array([
    #     [
    #         # p['q_median']
    #         p['mse_median']
    #         for p in points 
    #         if p['patch_size'] == xi
    #         and p['hidden_neurons'] == yi
    #     ][0]
    #     for xi, yi in zip(np.ravel(X), np.ravel(Y))
    # ]).reshape(X.shape)

    # ax.plot_trisurf(
    #     x, y, z, linewidth=1, cmap=cm.hot)
    # # ax.plot_surface(
    # #     X, Y, Z,
    # #     rstride=1, cstride=1, linewidth=1, cmap=cm.hot)
    # plt.show()


def plot2(results):
    points = []
    for result in results:
        image_name = result['image']['name']
        blur, noise, contrast = _parse_image_name(image_name)
        if blur == 0.0 and noise == 0.0:
            continue
        else:
            # Get or create point
            try:
                point = next(
                    p for p in points
                    if p['image'] == image_name)
            except StopIteration:
                point = {
                    'image': image_name,
                    # 'param_set': result['param_set']['name'],
                    # 'patch_size': result['param_set']['input_size'],
                    # 'hidden_neurons': result['param_set']['hidden_neurons'],
                    'q': [],
                    'ocr': [],
                    'mse': [],
                }
                points.append(point)
            point['q'] += result['metrics_relative']['q']
            # point['ocr'] += result['metrics_relative']['ocr']
            point['ocr'] += result['metrics_absolute']['ocr']
            point['mse'] += result['metrics_relative']['mse']

            # if max(result['metrics_relative']['q']) > 5.0:
            #     print image_name, result['param_set']['name'], result['metrics_relative']['q']

    for point in points:
        point['q'] = [p for p in point['q'] if p is not None]
        point['ocr'] = [p for p in point['ocr'] if p is not None]
        point['mse'] = [p for p in point['mse'] if p is not None]
        point['q_median'] = np.median(point['q'])
        point['ocr_median'] = np.median(point['ocr'])
        point['ocr_mean'] = np.mean(point['ocr'])
        point['mse_median'] = np.median(point['mse'])
        point['q_min'] = np.min(point['q'])
        point['ocr_min'] = np.min(point['ocr'])
        point['mse_min'] = np.min(point['mse'])
        point['q_max'] = np.max(point['q'])
        point['ocr_max'] = np.max(point['ocr'])
        point['mse_max'] = np.max(point['mse'])
        print point['image']

    fig = plt.figure(figsize=(6, 4))
    # plt.tight_layout()
    # plot_points = [
    #     p['q_median'] for p in sorted(points, key=lambda a: a['image'])
    # ]
    plt.plot([
        p['q_median'] for p in sorted(points, key=lambda a: a['image'])
    ], linewidth=2.0)
    # plt.plot([
    #     p['q_min'] for p in sorted(points, key=lambda a: a['image'])
    # ], linewidth=2.0)
    # plt.plot([
    #     p['q_max'] for p in sorted(points, key=lambda a: a['image'])
    # ], linewidth=2.0)
    # for patch_size in patch_size_values:
    #     plt.plot(
    #         [point['q_median'] for point in points if point['patch_size'] == patch_size],
    #         label='Lango dydis: ' + str(patch_size),
    #         linewidth=2.0)
    plt.legend(loc='lower right')
    # plt.xticks(
    #     list(xrange(len(hidden_neuron_values))),
    #     hidden_neuron_values)
    # plt.ylim([0, 1])
    plt.xlabel(u'Vaizdo eil. nr.')
    plt.ylabel(u'Vidurinė MSE įverčio pokyčio reikšmė')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def simple_plot_line(values, xlim=None, ylim=None):
    fig = plt.figure(figsize=(14, 7))
    plt.plot(values)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


# def parse_epochs(results):
#     regex = r'\D+(?P<epochs>\d+)\D+(?P<error>\d+[.]\d+)\D+(?P<bit_fail>\d+)'
#     reports = []
#     lines = results['output'].split('\n')
#     for line in lines[1:]:
#         if len(line) > 0:
#             match = re.match(regex, line)
#             reports.append(match.groupdict())
#     return reports


def parse_epochs(results):
    # Faster version without regular expressions
    lines = results['output'].split('\n')
    reports = []
    for line in lines[1:]:
        if len(line) > 0:
            err_start = line.index(':') + 2
            err_end = err_start + line[err_start:].index(' ') - 1
            reports.append(float(line[err_start:err_end]))
    return reports


def plot_epoch_errors(results):
    epoch_errors = [float(r['error']) for r in parse_epochs(results)]
    simple_plot_line(epoch_errors)


def count_invalid(results):
    invalid = 0
    for result in results:
        invalid_result = 0
        for idx in xrange(10):
            q = result['metrics_absolute']['q'][idx]
            ocr = result['metrics_absolute']['ocr'][idx]
            if q == 0.0 and ocr == 0.0:
                invalid += 1
                invalid_result += 1
        if invalid_result > 0:
            print result['image']['name'], result['param_set']['name'], invalid_result
    return invalid


if __name__ == "__main__":
    result_run_dir = '/home/tomas/Masters/4_semester/synthetic_tests/runs/'
    noisy_image_dir = '/home/tomas/Masters/4_semester/synthetic_tests/noisy/1/'
    for run in xrange(1, 10 + 1):
        print "### RUN " + str(run) + " ###"
        # Parse final epoch errors
        # parse_results(
        #     result_run_dir + str(run),
        #     result_run_dir + str(run) + '.csv')

        # Run neural filters
        # recreate_images(result_run_dir + str(run), noisy_image_dir)
        
        # Q metric
        # run_q(
        #     result_run_dir + str(run),
        #     result_run_dir + str(run) + '_q.csv')

        # Tesseract OCR
        # run_ocr(
        #     result_run_dir + str(run),
        #     result_run_dir + str(run) + '_ocr.csv')

        # MSE values
        run_mse(
            result_run_dir + str(run),
            result_run_dir + str(run) + '_mse.csv')
