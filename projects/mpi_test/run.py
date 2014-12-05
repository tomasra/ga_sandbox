#!/usr/bin/env python
from core.parallelizer import Parallelizer
import time
import random


def task():
    for i in xrange(100000):
        i * i
    # return random.randint()
    return 42


def write(value):
    print value

with Parallelizer() as p:
    if p.master_process:
        start = time.time()

        for iteration in xrange(10):
            for task_id in xrange(100):
                p.start_task(task_id, lambda: task())

            results = [
                task_result
                for task_id, task_result in p.finished_tasks()
            ]
            print "Iteration", iteration, "complete"

        end = time.time()
        print end - start

# start = time.time()

# for iteration in xrange(10):
#     for task_id in xrange(100):
#         result = task()
#     print "Iteration", iteration, "complete"

# end = time.time()
# print end - start
