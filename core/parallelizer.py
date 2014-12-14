try:
   import cPickle as pickle
except:
   import pickle

import dill     # For function pickling
from mpi4py import MPI
import cProfile

# For debugging
_PROFILING_ENABLED = True
_PROFILE_FILENAME = 'profile'
_PROFILE_EXTENSION = '.txt'

# Reserved process IDs
MASTER_PROC_ID = 0

# Messages
TAG_START_TASK = 1000
TAG_START_PREPARED_TASK = 1001
TAG_TASK_COMPLETE = 1100
TAG_BROADCAST_DATA = 1200
TAG_TERMINATE = 2000


PREPARED_TASKS = {}


def parallel_task(decorated_function):
    """
    Simple decorator to mark tasks to be run by workers.
    Decorated function should accept only **kwargs.
    That dictionary is merged from two parts:
    - Function call ('start_prepared_task') keyword arguments
    - Key/value pairs broadcasted from master to workers
    If the same keyword argument exists both in call arguments
    and broadcasted arguments, priority is given to broadcasted argument.

    @parallel_task
    def do_stuff(**kwargs):
        foo = kwargs['foo']
        bar = kwargs['bar']
        return foo + bar

    ### Assuming that 'bar' is already broadcasted to all workers
    parallelizer.start_prepared_task(
        task_id=0, 'do_stuff',
        foo="Here be dragons")
    """
    name = decorated_function.__name__
    if name in PREPARED_TASKS:
        raise ValueError("Parallel task with such name already exists")
    else:
        PREPARED_TASKS[name] = decorated_function
    return decorated_function


class Parallelizer(object):
    def __init__(self, prepared_tasks=None):
        self.comm = MPI.COMM_WORLD
        self.proc_count = self.comm.Get_size()
        self.proc_id = self.comm.Get_rank()

        self.prepared_tasks = {}
        if prepared_tasks is not None:
            for task in prepared_tasks:
                task_name = task.__name__
                self.prepared_tasks[task_name] = task

        # Workers
        self.received_data = {}

    def __enter__(self):
        """
        Start the workers
        """
        # For debugging
        if _PROFILING_ENABLED:
            self.profile = cProfile.Profile()
            self.profile.enable()

        # At least two processes must be available
        # (master and worker)
        if self.proc_count < 2:
            # Fallback
            return NullParallelizer(self.prepared_tasks.values())
        else:
            if self.master_process:
                # Prepare for incoming tasks
                self._available_workers = self._get_worker_ids()
                self._task_results = {}
                self._task_semaphore = 0
                return self
            else:
                # Enter the worker loop
                self._worker()
                return self

    def __exit__(self, type, value, traceback):
        """
        Breaks loops of all non-master processes
        """
        if self.master_process:
            # Stop workers
            for worker_id in self._get_worker_ids():
                self.comm.send(
                    None, dest=worker_id,
                    tag=TAG_TERMINATE)

        # For debugging
        if _PROFILING_ENABLED:
            self.profile.disable()
            self.profile.dump_stats(
                _PROFILE_FILENAME + str(self.proc_id) + _PROFILE_EXTENSION)

    def get_prepared_task(self, task_name):
        """
        Return function/parallel task with specified name
        """
        # Local prepared task dict has priority against global one
        if task_name in self.prepared_tasks:
            return self.prepared_tasks[task_name]
        elif task_name in PREPARED_TASKS:
            return PREPARED_TASKS[task_name]
        else:
            return None

    @property
    def master_process(self):
        return self.proc_id == MASTER_PROC_ID

    def _get_available_worker_id(self):
        """
        Return ID of the first available worker
        or wait until one becomes available.
        """
        if not self._available_workers:
            # All workers currently busy,
            # need to wait for notification from receiver process
            worker_id = self._wait_for_worker()
        else:
            worker_id = self._available_workers.pop()
        return worker_id

    def start_task(self, task_id, task):
        """
        Send a callable task (lambda) .
        """
        payload = (
            task_id,
            pickle.dumps(task, pickle.HIGHEST_PROTOCOL),
        )
        self.comm.send(
            payload,
            dest=self._get_available_worker_id(),
            tag=TAG_START_TASK)
        self._task_semaphore += 1

    def start_prepared_task(self, task_id, task_name, **kwargs):
        """
        Invoke a predefined task on receiver side with passed arguments.
        """
        if self.get_prepared_task(task_name) is not None:
            pickled_kwargs = pickle.dumps(kwargs, pickle.HIGHEST_PROTOCOL)
            payload = (
                task_id,
                task_name,
                pickled_kwargs,
            )
            self.comm.send(
                payload,
                dest=self._get_available_worker_id(),
                tag=TAG_START_PREPARED_TASK)
            self._task_semaphore += 1
        else:
            raise ValueError("Prepared task with such name was not found")

    def finished_tasks(self):
        """
        Wait for all workers to finish their tasks,
        return all task results collected so far
        and prepare for next batch of tasks.
        """
        if self.master_process:
            # Some tasks still running?
            running_tasks = self._task_semaphore
            for _ in xrange(running_tasks):
                self._wait_for_worker()

            # All tasks complete, return results
            for task_id, task_result in self._task_results.iteritems():
                yield task_id, task_result

            # Reset everything
            self._available_workers = self._get_worker_ids()
            self._task_results = {}

    def broadcast(self, **kwargs):
        """
        Distribute arbitrary key/value pairs to workers
        to be later used in calculations
        """
        if self.master_process:
            # Notify all workers about incoming data so they block
            # on 'bcast' call
            for worker_id in self._get_worker_ids():
                self.comm.send(
                    None,
                    dest=worker_id,
                    tag=TAG_BROADCAST_DATA)

            # Broadcast the data itself
            self.comm.bcast(kwargs, root=MASTER_PROC_ID)
        return self

    def _wait_for_worker(self):
        """
        Blocking receive from worker process to get notification
        about completed task
        """
        status = MPI.Status()
        message = self.comm.recv(
            source=MPI.ANY_SOURCE,
            tag=TAG_TASK_COMPLETE,
            status=status)
        task_id, task_result = message

        # Store result and decrease running task count
        self._task_results[task_id] = task_result
        self._task_semaphore -= 1

        # Worker ID
        return status.source

    def _get_worker_ids(self):
        """
        All available process IDs minus master process
        """
        all_ids = list(range(self.proc_count))
        try:
            all_ids.remove(MASTER_PROC_ID)
        except:
            pass
        return all_ids

    def _worker(self):
        """
        Accept task from master process, run it
        and send back the result. Rinse. Repeat.
        """
        while True:
            # Listen for any messages
            status = MPI.Status()
            message = self.comm.recv(
                source=MASTER_PROC_ID,
                tag=MPI.ANY_TAG,
                status=status)

            if status.tag == TAG_START_TASK:
                # Unpack task info
                task_id, task = message
                task = pickle.loads(task)
                # Run the task and send results to master
                result = task()
                payload = (task_id, result)
                self.comm.send(
                    payload,
                    dest=MASTER_PROC_ID, tag=TAG_TASK_COMPLETE)

            elif status.tag == TAG_START_PREPARED_TASK:
                # Unpack all task info and get the task itself
                task_id, task_name, kwargs = message
                kwargs = pickle.loads(kwargs)
                # Add any previously broadcasted/received keyword arguments
                # THIS MIGHT OVERRIDE EXISTING ARGUMENTS!
                kwargs = dict(kwargs.items() + self.received_data.items())

                # Run the task and send results back
                task = self.get_prepared_task(task_name)
                result = task(**kwargs)
                payload = (task_id, result)
                self.comm.send(
                    payload,
                    dest=MASTER_PROC_ID, tag=TAG_TASK_COMPLETE)

            elif status.tag == TAG_BROADCAST_DATA:
                # Receive dictionary from master process
                # and store its keys/values
                received_data = {}
                received_data = self.comm.bcast(
                    received_data, root=MASTER_PROC_ID)
                self.received_data.update(received_data)

            elif status.tag == TAG_TERMINATE:
                # Exit loop
                break
            else:
                raise RuntimeError(
                    "Worker: invalid command")


class NullParallelizer(Parallelizer):
    """
    Fake parallelizer for such cases when
    not enough processes are available and tasks have to be run
    in usual serial way.
    """
    def __init__(self, *args, **kwargs):
        super(NullParallelizer, self).__init__(*args, **kwargs)
        self._task_results = {}
        self._received_data = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    @property
    def master_process(self):
        return True

    def start_task(self, task_id, task):
        """
        Run the task and save results
        """
        self._task_results[task_id] = task()
        return self

    def start_prepared_task(self, task_id, task_name, *args, **kwargs):
        """
        Do the same
        """
        task = self.get_prepared_task(task_name)
        all_kwargs = dict(kwargs.items() + self._received_data.items())
        self._task_results[task_id] = task(**all_kwargs)
        return self

    def broadcast(self, **kwargs):
        self._received_data.update(kwargs)

    def finished_tasks(self):
        """
        Return all collected task results
        """
        for task_id, task_result in self._task_results.iteritems():
            yield task_id, task_result
        self._task_results = {}
