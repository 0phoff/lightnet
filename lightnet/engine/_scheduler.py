#
#   Learning Rate Scheduling
#   Copyright EAVISE
#

import torch
import logging

__all__ = ['SchedulerCompositor']
log = logging.getLogger(__name__)


class SchedulerCompositor:
    """ This class can be used to schedule schedulers to run at different moments on the same parameters of a network. |br|
    This compositor has a notion of count values. These can be batch number, epoch number, etc. and dictate when each scheduler is being used.

    Args:
        *args (list of tuples): This class gets initialized with tuples of ``(count, scheduler)``, which determine when to start using which scheduler

    Example:
        >>> class DummyScheduler:
        ...     " Dummy scheduler that does nothing but print it's id value. "
        ...     def __init__(self, id_value):
        ...         self.id = id_value;
        ...     def step(self):
        ...         print(f'{self.last_epoch} - Dummy Scheduler: {self.id}')
        >>> s = ln.engine.SchedulerCompositor(
        ...     (1, DummyScheduler('start')),
        ...     (3, DummyScheduler('middle')),
        ...     (4, DummyScheduler('end')),
        ... )
        >>> # Note that then "count_value" argument is just for this example
        >>> for i in range(5):
        ...     s.step()
        1 - Dummy Scheduler: start
        2 - Dummy Scheduler: start
        3 - Dummy Scheduler: middle
        4 - Dummy Scheduler: end
        5 - Dummy Scheduler: end
    """
    def __init__(self, *args, last_epoch=0):
        if len(args) == 0:
            raise ValueError('Compositor requires at least one scheduler')

        self.epoch, self.sched = zip(*args)
        if not all(c1 < c2 for c1, c2 in zip(self.epoch, self.epoch[1:])):
            raise ValueError('Count values need to be strictly increasing')

        self.last_epoch = last_epoch

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ['
        clen = max(len(str(c)) for c in self.epoch)
        for i in range(len(self.epoch)):
            if hasattr(self.sched[i], '__name__'):
                name = self.sched[i].__name__
            else:
                name = self.sched[i].__class__.__name__

            format_string += f'\n  {self.epoch[i]:>{clen}}:  {name}'
        format_string += '\n]'
        return format_string

    def get_last_lr(self):
        sched = self._get_scheduler(self.last_epoch)
        if sched is not None:
            return sched.get_last_lr()

    def step(self, **kwargs):
        """ Stepping function that will select a scheduler and run it.

        Args:
            **kwargs (dict, optional): Extra arguments that will be passed on to the step function of the scheduler itself.
        """
        self.last_epoch += 1
        sched = self._get_scheduler(self.last_epoch)

        if sched is not None:
            sched.last_epoch = self.last_epoch
            return sched.step(**kwargs)

    def _get_scheduler(self, epoch):
        for i, e in enumerate(self.epoch):
            if epoch < e:
                i -= 1
                break

        if i < 0:
            log.error(f'No Scheduler defined for count value of {epoch}')
            return None
        else:
            return self.sched[i]

    def state_dict(self):
        state_dict = [s.state_dict() for s in self.sched]
        return state_dict

    def load_state_dict(self, state, strict=True):
        [self.sched[i].load_state_dict(s) for i, s in enumerate(state)]

    def to(self, device):
        """ Cast schedulers to a certain device.

        Args:
            device (torch.device): Device to cast the scheduler to.
        """
        for sched in self.sched:
            for param in sched.__dict__.values():
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(device)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(device)
