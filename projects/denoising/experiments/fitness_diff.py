#!/usr/bin/env python
import sys
import pickle
import projects.denoising.experiments.experiment as exp
# from projects.denoising.imaging.utils import render_image

if __name__ == "__main__":
    # Directory with json result files
    result_dir = sys.argv[1]
    result_sets = exp.read_results(result_dir)

    # Sort by population size
    result_sets = sorted(
        result_sets,
        key=lambda s: s.parameters.population_size)

    for rs in result_sets:
        population_size = rs.parameters.population_size
        last_iteration_idx = rs.results.iterations - 1
        # Best fitness from file
        best_fitness_file = rs.iterations[last_iteration_idx].best_fitness

        # Evaluated fitness of the best solution
        best_solution = pickle.loads(rs.results.solution_dump)
        source_image = pickle.loads(rs.parameters.source_image_dump)
        target_image = pickle.loads(rs.parameters.target_image_dump)
        best_solution.source_image = source_image
        best_solution.target_image = target_image
        best_fitness_eval = best_solution._calculate_fitness()

        print "Size: %i, file: %f, real: %f, diff: %f" % (
            population_size,
            best_fitness_file, best_fitness_eval,
            abs(best_fitness_file - best_fitness_eval)
        )
        # render_image(source_image.run_filters(best_solution))
