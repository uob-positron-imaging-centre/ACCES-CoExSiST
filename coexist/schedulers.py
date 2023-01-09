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

        $> ``python3 some_script.py arg1 arg2 arg3``

    Here, `python3` is the "scheduler", simply creating a new OS process in
    which a Python interpreter executes `some_script.py`. This is implemented
    in the ``LocalScheduler`` class.

    In a more complex, multi-cluster environment managed by SLURM:

        $> ``sbatch job_submission.sh some_script.py arg1 arg2 arg3``

    Here, the `sbatch job_submission.sh` is the scheduling part, and the
    `job_submission.sh` SLURM script must be generated beforehand. This is
    implemented in the ``SlurmScheduler`` class.

    **Subclassing:**

    If you want to implement a concrete scheduler for another system, subclass
    `Scheduler` and implement the `schedule(dirpath, index)` method (`dirpath`
    is the directory where the computation is run, e.g. "access_seed123" and
    `index` is the job number), which will be called when launching each
    simulation to:

    - Generate any files needed to schedule a Python script's execution.
    - Return a list of the system commands to be prepended to the Python
      scripts (e.g. ["python3"] and ["sbatch", "job_submission.sh"] in the
      examples above).
    '''

    @abstractmethod
    def schedule(self, dirpath, index):
        pass




class LocalScheduler(Scheduler):
    '''Schedule parallel workloads on the local / shared-memory machine.

    By default, it will use the ``sys.executable`` Python interpreter (i.e. the
    one used to execute the current code); you can set it to a different
    name, e.g. ``coexist.schedulers.LocalScheduler(["python3"])``.
    '''

    def __init__(self, python_executable = [sys.executable]):
        self.python_executable = list(python_executable)


    def schedule(self, dirpath, index):
        return self.python_executable


    def __repr__(self):
        return f"LocalScheduler(python_executable={self.python_executable})"




class SlurmScheduler(Scheduler):
    '''Launch simulations on a SLURM distributed cluster using ``sbatch``.

    First a bash script must be defined for launching each simulation job; this
    class generates this script, but some details must be defined by you; they
    are specified as class parameters, see examples below.

    Parameters
    ----------
    time : str
        The time allocated for *a single simulation*, given as a string, e.g.
        "1:0:0". Will be added as "#SBATCH --time 1:0:0".

    qos : str, optional
        The "#SBATCH --qos bbdefault" ``sbatch`` command.

    mail_type : str, default "FAIL"
        The "#SBATCH --mail-type FAIL" ``sbatch`` command.

    ntasks : str, default "1"
        The "#SBATCH --ntasks 1" ``sbatch`` command.

    mem : str, optional
        The "#SBATCH --mem 4G" ``sbatch`` command.

    commands : str or list[str], default "module load Python"
        Any other *non-SLURM* commands to run in the job submission script
        before executing the simulation; this is normally the setup work, e.g.
        loading necessary modules, environments, etc.

    interpreter : str, default os.path.split(sys.executable)[1]
        Name of the Python interpreter which will be used to execute the
        simulation script; this is normally set to the name of the executable
        used to run ACCES, e.g. "usr/bin/python3" -> "python3".

    **kwargs : other keyword arguments
        Other "#SBATCH" commands to include at the top of the job submission
        script; e.g. ``constraint = "cascadelake"`` is transformed into
        ``"#SBATCH --constraint cascadelake"``; ``mem_per_cpu = "4"`` is
        transformed into ``"#SBATCH --mem-per-cpu 4"``.

    Examples
    --------

    >>> from coexist.schedulers import SlurmScheduler
    >>> scheduler = SlurmScheduler(
    >>>     "10:0:0",          # Time allocated for a single simulation
    >>>     commands = """
    >>>         # Commands to add in the sbatch script after `#`
    >>>         set -e
    >>>         module purge; module load bluebear
    >>>         module load BEAR-Python-DataScience
    >>>     """,
    >>>     qos = "bbdefault",
    >>>     constraint = "cascadelake",   # Any other #SBATCH --<CMD> = "VAL"
    >>> )
    '''

    def __init__(
        self,
        time,
        qos = None,
        mail_type = "FAIL",
        ntasks = "1",
        mem = None,
        commands = "set -e\n",
        interpreter = os.path.split(sys.executable)[1],
        script = "access_slurm_submission.sh",
        **kwargs,
    ):
        self.time = str(time)
        self.commands = commands
        self.interpreter = str(interpreter)

        self.qos = str(qos) if qos is not None else None
        self.mail_type = str(mail_type) if mail_type is not None else None

        self.ntasks = str(ntasks) if ntasks is not None else None
        self.mem = str(mem) if mem is not None else None

        self.script = script
        self.kwargs = kwargs


    def generate(self, scriptpath):
        with open(scriptpath, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --time {self.time}\n")

            if self.qos is not None:
                f.write(f"#SBATCH --qos {self.qos}\n")

            if self.mail_type is not None:
                f.write(f"#SBATCH --mail-type {self.mail_type}\n")

            if self.ntasks is not None:
                f.write(f"#SBATCH --ntasks {self.ntasks}\n")

            if self.mem is not None:
                f.write(f"#SBATCH --mem {self.mem}\n")

            for key, val in self.kwargs.items():
                f.write(f"#SBATCH --{key.replace('_', '-')} {val}\n")

            f.write("#SBATCH --wait\n")

            f.write("\n\n")
            if isinstance(self.commands, str):
                f.write(self.commands)
            else:
                for cmd in self.commands:
                    # Small convenience, but if the strings in the list of
                    # commands don't end with a '\n', append it
                    if not cmd.endswith("\n"):
                        cmd += "\n"
                    f.write(cmd)

            f.write((
                "\n\n# Run a single function evaluation with all command-line "
                "arguments redirected to Python\n"
            ))
            f.write(f"{self.interpreter} $*\n")


    def schedule(self, dirpath, index):
        # Check directory exists
        if not os.path.isdir(dirpath):
            raise FileNotFoundError(
                f"The given `dirpath` = '{dirpath}' does not exist."
            )

        # Generate SLURM launch script if it doesn't exist
        scriptpath = os.path.join(dirpath, self.script)
        if not os.path.isfile(scriptpath):
            self.generate(scriptpath)

        # Check outputs directory exists and create it otherwise
        outputdir = os.path.join(dirpath, "outputs")
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)

        outputpath = os.path.join(outputdir, f"output.{index}.slurm-%j.out")
        return ["sbatch", f"--output={outputpath}", scriptpath]


    def __repr__(self):
        # Return pretty string representation of an arbitrary object
        docs = []
        for attr in dir(self):
            if not attr.startswith("_"):
                memb = getattr(self, attr)
                if not callable(memb):
                    docs.append(f"{attr} = {memb}")

        name = self.__class__.__name__
        underline = "-" * len(name)
        return f"{name}\n{underline}\n" + "\n".join(docs)
