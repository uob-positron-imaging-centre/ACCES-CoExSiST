#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : liggghts.py
# License: GNU v3.0
# Author : Dominik Werner
# Date   : 28.04.2023


import  re
import  os
import  pickle
import  pathlib
import  textwrap
import subprocess
import zmq
import  numpy       as      np
from    pyevtk.hl   import  pointsToVTK
import sys

# Local imports
from    ..base       import  Simulation
from    ..liggghts   import  LiggghtsSimulation



class LiggghtsMPI():
    """Class to manage the liggghts simulation with mpi

    Parameters
    ----------
    sim_name : str
        Name of the simulation file
    cores : int
        Number of cores to use
    verbose : bool
        If True, print the output of the liggghts simulation

    Methods
    -------

    load_file(sim_name)
        Load a simulation file

    command(args)
        Run a liggghts command

    close()
        Close the liggghts simulation


    """

    def __init__(
        self,
        sim_name,
        cores = 2,
        verbose = False,
    ):


        # Create class attributes
        self._verbose = bool(verbose)
        self._cmdargs = None
        self.filename = sim_name
        self.cores = cores
        self.start_process()
        # self.load_file(sim_name)


    def start_process(self):
        executing_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "mpi_liggghts_script.py"
        )
        executable = sys.executable
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PAIR)
        port_selected = self._socket.bind_to_random_port("tcp://*")
        cmds = ["mpiexec"]

        cmds += [
            "-n",
            str(self.cores),
            executable,
            executing_file,
            "--zmqport",
            str(port_selected),
            "--file",
            self.filename,
            "--verbose",
            str(self._verbose),
        ]
        if self._cmdargs is not None:
            cmds.extend(self._cmdargs)
        #print(" ", " ".join(cmds))

        self._process = subprocess.Popen(
            cmds,
            # stdout=subprocess.PIPE,
            stderr=None,
            # stdin=subprocess.PIPE,
            cwd=".",
            env=os.environ,
        )

        # receive the output, which emans the sim ran the input file
        self._receive()

    def _send(self, command, args, kwargs):
        # send a command to the liggghts instance running with mpi
        #print("Sending command ", command)
        self._socket.send(pickle.dumps({"c": command, "d": [args, kwargs]}))
        return self._receive()

    def _receive(self):
        # receive the output of the command
        output = pickle.loads(self._socket.recv())
        return output

    def load_file(self, filename):
        self._send("load_file", [filename])
        # print(self._receive())

    def __del__(self):
        self._send("close", None, None)
        self._process.terminate()

    def __getattribute__(self, __name: str):
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            if __name in LiggghtsSimulation.__dict__:
                def wrapper(*args, **kwargs):
                    return self._send(__name, args, kwargs)

                return wrapper
            else:
                raise AttributeError(f"Attribute {__name} not found")
