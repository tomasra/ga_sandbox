#!/usr/bin/env python
import sys
import pickle
import projects.denoising.experiments.experiment as exp
from projects.denoising.imaging.utils import render_image


def filter_image(result_filepath):
    """
    Generate noisified image according to parameters
    and then filter it with the best solution dumped in results file
    """
    result_set = exp.read_results_file(result_filepath)

    try:
        source_image = pickle.loads(result_set.parameters.source_image_dump)
        target_image = pickle.loads(result_set.parameters.target_image_dump)
        print "Using image dumps from result file"
    except AttributeError:
        print "Image dumps not found in result file - generating new images"
        noise_type = result_set.parameters.noise_type
        noise_param = result_set.parameters.noise_param
        source_image, target_image = exp.generate_images(
            noise_type, noise_param)

    render_image(source_image)
    # render_image(target_image)

    best_solution = pickle.loads(result_set.results.solution_dump)
    best_solution.source_image = source_image
    best_solution.target_image = target_image

    last_iteration_idx = result_set.results.iterations - 1
    # Best fitness from file
    best_fitness_file = result_set.iterations[
        last_iteration_idx].best_fitness

    print "File fitness: %f" % (best_fitness_file)
    print "Eval fitness: %f" % (best_solution._calculate_fitness())
    filtered_image = source_image.run_filters(best_solution)
    return filtered_image


if __name__ == "__main__":
    filtered_image = filter_image(sys.argv[1])
    render_image(filtered_image)
