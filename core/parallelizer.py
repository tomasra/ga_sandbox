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
TAG_TERMINATE = 1200


PREPARED_TASKS = {}


def parallel_task(decorated_function):
    """
    Simple decorator to mark tasks to be run by workers
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
            return _NullParallelizer(self.proc_id)
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

    def start_prepared_task(self, task_id, task_name, *args, **kwargs):
        """
        Invoke a predefined task on receiver side with passed arguments.
        """
        if self.get_prepared_task(task_name) is not None:
            payload = (
                task_id,
                task_name,
                pickle.dumps(args, pickle.HIGHEST_PROTOCOL),
                pickle.dumps(kwargs, pickle.HIGHEST_PROTOCOL),
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
                task_id, task_name, args, kwargs = message
                args = pickle.loads(args)
                kwargs = pickle.loads(kwargs)

                # Run the task and send results back
                task = self.get_prepared_task(task_name)
                result = task(*args, **kwargs)
                payload = (task_id, result)
                self.comm.send(
                    payload,
                    dest=MASTER_PROC_ID, tag=TAG_TASK_COMPLETE)

            elif status.tag == TAG_TERMINATE:
                # Exit loop
                break
            else:
                raise RuntimeError(
                    "Worker: invalid command")


class _NullParallelizer(object):
    """
    Fake parallelizer for such cases when
    not enough processes are available and tasks have to be run
    in usual serial way.
    """
    def __init__(self, proc_id):
        self.proc_id = proc_id
        self._task_results = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    @property
    def master_process(self):
        return self.proc_id == MASTER_PROC_ID

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
        task = PREPARED_TASKS[task_name]
        self._task_results[task_id] = task(*args, **kwargs)
        return self

    def finished_tasks(self):
        """
        Return all collected task results
        """
        for task_id, task_result in self._task_results.iteritems():
            yield task_id, task_result
        self._task_results = {}
