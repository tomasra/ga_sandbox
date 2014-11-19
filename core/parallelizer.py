import pickle
import dill     # For function pickling
from mpi4py import MPI
MASTER_PROC_ID = 0


class Parallelizer(object):
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.proc_count = self.comm.Get_size()
        self.proc_id = self.comm.Get_rank()

        # For master process
        self._next_worker_id = MASTER_PROC_ID + 1

        # Created/completed tasks by worker ID
        self._task_count = {}

        # For worker process
        self._tasks = []
        self._task_results = []

    def __enter__(self):
        """
        Start the workers
        """
        if self.proc_id == MASTER_PROC_ID:
            # Reset task counts
            for worker_id in xrange(1, self.proc_count):
                self._task_count[worker_id] = 0
            return self
        else:
            self._worker()
            return self

    def __exit__(self, type, value, traceback):
        """
        Stop all workers
        """
        if self.proc_id == MASTER_PROC_ID:
            for worker_id in xrange(1, self.proc_count):
                self.comm.send('terminate', dest=worker_id)

    def create_task(self, task):
        """
        Give task to worker
        """
        if self.proc_id == MASTER_PROC_ID:
            # Notify the worker about incoming task
            worker_id = self.choose_worker()
            self.comm.send('create_task', dest=worker_id)
            self.comm.send(pickle.dumps(task), dest=worker_id)
            self._task_count[worker_id] += 1
        else:
            raise RuntimeError(
                "Attempted to create task in worker process")

    def start_tasks(self):
        """
        Notify workers to start their tasks
        """
        if self.proc_id == MASTER_PROC_ID:
            for worker_id in xrange(1, self.proc_count):
                self.comm.send('start_tasks', dest=worker_id)
            return self
        else:
            raise RuntimeError(
                "Attempted to start tasks in worker process")

    def finish_tasks(self):
        """
        Order the workers to return task results
        """
        if self.proc_id == MASTER_PROC_ID:
            results = []
            for worker_id in xrange(1, self.proc_count):
                # Ask worker to return all task results
                self.comm.send('finish_tasks', dest=worker_id)
                for _ in xrange(self._task_count[worker_id]):
                    # Collect the results
                    result = self.comm.recv(source=worker_id)
                    results.append(result)

                # Reset worker task counts
                self._task_count[worker_id] = 0

            return results
        else:
            raise RuntimeError(
                "Attempted to finish tasks in worker process")

    def choose_worker(self):
        """
        Simple round-robin to pick worker process ID
        """
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        if self._next_worker_id == self.proc_count:
            # Start over
            self._next_worker_id = MASTER_PROC_ID + 1
        return worker_id

    def _worker(self):
        """
        Worker 'thread'
        """
        while True:
            command = self.comm.recv(source=MASTER_PROC_ID)
            # print command, self.proc_id
            if command == 'create_task':
                # Add task to the 'TODO' list
                task = pickle.loads(self.comm.recv(source=MASTER_PROC_ID))
                self._tasks.append(task)
            elif command == 'start_tasks':
                # Do the tasks
                self._task_results = [
                    task()
                    for task in self._tasks
                ]
            elif command == 'finish_tasks':
                for result in self._task_results:
                    self.comm.send(result, dest=MASTER_PROC_ID)
                self._tasks = []
                self._task_results = []
            elif command == 'terminate':
                break
