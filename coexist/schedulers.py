#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : schedulers.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 04.04.2021


import  os
import  sys
from    abc         import  ABC, abstractmethod


class Scheduler(ABC):
    '''An abstract class defining the interface that concrete schedulers need
    to implement.

    A scheduler, in general, is a program that defines a way to split parallel
    workloads in some computing environment. More specifically, it allows
    calling a Python script with some command-line arguments in parallel; in
    the simplest case:

        $> python3 some_script.py arg1 arg2 arg3

    Here, `python3` is the "scheduler", simply creating a new OS process in
    which a Python interpreter executes `some_script.py`. This is implemented
    in the `LocalScheduler` class.

    In a more complex, multi-cluster environment managed by SLURM:

        $> sbatch job_submission.sh some_script.py arg1 arg2 arg3

    Here, the `sbatch job_submission.sh` is the scheduling part, and the
    `job_submission.sh` SLURM script must be generated beforehand. This is
    implemented in the `SlurmScheduler` class.

    Subclassing
    -----------
    If you want to implement a concrete scheduler for another system, subclass
    `Scheduler` and implement the `generate` method, which should:

    - Generate any files needed to schedule a Python script's execution.
    - Return a list of the system commands to be prepended to the Python
      scripts (e.g. ["python3"] and ["sbatch", "job_submission.sh"] in the
      examples above).

    '''

    @abstractmethod
    def generate(self):
        pass



class LocalScheduler(Scheduler):
    '''Schedule parallel workloads on the local / shared-memory machine.
    '''

    def __init__(self, python_executable = [sys.executable]):
        self.python_executable = list(python_executable)


    def generate(self):
        return self.python_executable




class SlurmScheduler(Scheduler):

    def __init__(
        self,
        time,
        commands = ["module load Python"],
        interpreter = os.path.split(sys.executable)[1],
        qos = None,
        account = None,
        mail_type = "FAIL",
        ntasks = 1,
        mem = None,
        output = "logs/sim_slurm_%j.out",
        **kwargs,
    ):
        self.time = str(time)
        self.commands = list(commands)
        self.interpreter = str(interpreter)

        self.qos = str(qos) if qos is not None else None
        self.account = str(account) if account is not None else None
        self.mail_type = str(mail_type) if mail_type is not None else None

        self.ntasks = str(ntasks) if ntasks is not None else None
        self.mem = str(mem) if mem is not None else None

        self.output = str(output) if output is not None else None
        self.kwargs = kwargs


    def generate(self):
        filename = "access_single_submission.sh"

        with open(filename, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --time {self.time}\n")

            if self.qos is not None:
                f.write(f"#SBATCH --qos {self.qos}\n")

            if self.account is not None:
                f.write(f"#SBATCH --account {self.account}\n")

            if self.mail_type is not None:
                f.write(f"#SBATCH --mail-type {self.mail_type}\n")

            if self.ntasks is not None:
                f.write(f"#SBATCH --ntasks {self.ntasks}\n")

            if self.mem is not None:
                f.write(f"#SBATCH --mem {self.mem}\n")

            if self.output is not None:
                output_path = os.path.split(self.output)
                if output_path[0] and not os.path.isdir(output_path[0]):
                    os.mkdir(output_path[0])

                f.write(f"#SBATCH --output {self.output}\n")

            f.write("#SBATCH --wait\n")

            for key, val in self.kwargs.items():
                f.write(f"#SBATCH --{key} {val}\n")

            f.write("\n\n")
            for cmd in self.commands:
                if not cmd.endswith("\n"):
                    cmd += "\n"
                f.write(cmd)

            f.write((
                "\n\n# Run a single function evaluation with all command-line "
                "arguments redirected to Python\n"
            ))
            f.write(f"{self.interpreter} $*\n")

        return ["sbatch", filename]
