from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:

    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN SOLUTION
    total_clock_cycles = num_batches + num_partitions - 1

    for clock in range(total_clock_cycles):
        current_step = [
            (microbatch_id, stage_id)
            for microbatch_id in range(num_batches)
            if 0 <= (stage_id := clock - microbatch_id) < num_partitions
        ]
        yield current_step
    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.

        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN SOLUTION
        # Step 1
        microbatches = list(torch.chunk(x, self.split_size, dim=0))
        num_microbatches = len(microbatches)
        num_stages = len(self.partitions)

        # Step 2
        pipeline_buffer = []
        for batch_idx in range(num_microbatches):
            stage_data = [None] * num_stages
            stage_data[0] = microbatches[batch_idx].to(self.devices[0])
            pipeline_buffer.append(stage_data)

        # Step 3
        execution_schedule = list(_clock_cycles(num_microbatches, num_stages))
        self.compute(pipeline_buffer, execution_schedule)

        # Step 4
        outputs = []
        for batch in pipeline_buffer:
            output = batch[-1].to(self.devices[-1])
            outputs.append(output)

        return torch.cat(outputs)
        # END SOLUTION

    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker.
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices
        in_queues = self.in_queues
        out_queues = self.out_queues

        # BEGIN SOLUTION
        for cycle in schedule:
            for microbatch_id, stage_id in cycle:
                device = devices[stage_id]
                partition = partitions[stage_id]
                in_queue = in_queues[stage_id]
                out_queue = out_queues[stage_id]

                if stage_id > 0:
                    input_tensor = batches[microbatch_id][stage_id - 1]
                else:
                    input_tensor = batches[microbatch_id][0]

                def compute_stage_output(input_tensor, partition_module, device):
                    def run():
                        if isinstance(input_tensor, tuple):
                            input_tensor_device = input_tensor[0].to(device)
                        else:
                            input_tensor_device = input_tensor.to(device)

                        return partition_module(input_tensor_device)

                    return run

                task_fn = compute_stage_output(input_tensor, partition, device)
                task = Task(task_fn)
                in_queue.put(task)

            for microbatch_id, stage_id in cycle:
                out_queue = out_queues[stage_id]
                success, result = out_queue.get()

                if not success:
                    raise result[1].with_traceback(result[2])

                task, batch = result
                batches[microbatch_id][stage_id] = batch
        # END SOLUTION

