#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : view_sim.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import csv

import numpy as np
import pandas as pd

import plotly.graph_objs as go


positions = np.load("truth/positions_short.npy")
timesteps = np.load("truth/timesteps_short.npy")

#positions2 = np.load("truth/positions_short_opt.npy")
#timesteps2 = np.load("truth/timesteps_short_opt.npy")

#indices = np.arange(len(positions[0]))
#distances = np.linalg.norm(positions - positions2, axis = 2)
#distances[distances < 0.005 / 2] = 0    # thresholding

max_range = positions.max(axis = 0).max(axis = 0)
min_range = positions.min(axis = 0).min(axis = 0)


sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Timestep:",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 300, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
}

frames = []
for i, t in enumerate(timesteps):
    frames.append({
        "data": [
            go.Scatter3d(
                x = positions[i][:, 0],
                y = positions[i][:, 1],
                z = positions[i][:, 2],
                mode = "markers",
                marker = dict(
                    opacity = 0.8,
                    size = 3,
                    color = np.arange(len(positions[i])), # distances[i],
                )
            ),
            #go.Scatter3d(
            #    x = positions2[i][:, 0],
            #    y = positions2[i][:, 1],
            #    z = positions2[i][:, 2],
            #    mode = "markers",
            #    marker = dict(
            #        opacity = 0.8,
            #        size = 3,
            #        color = "red",
            #    )
            #),
        ],
        "name": str(t)
    })

    slider_step = {
        "args": [
            [str(t)],
            {
                "frame": {
                    "duration": 300,
                    "redraw": True
                },
                "mode": "immediate",
                "transition": {"duration": 300}
            }
        ],
        "label": str(t),
        "method": "animate"
    }
    sliders_dict["steps"].append(slider_step)


fig = go.Figure(
    data = frames[0]["data"],
    layout = dict(
        scene = dict(
            xaxis_title = "x",
            yaxis_title = "y",
            zaxis_title = "z",

            xaxis_range = [min_range[0], max_range[0]],
            yaxis_range = [min_range[1], max_range[1]],
            zaxis_range = [min_range[2], max_range[2]],
        ),
        updatemenus = [{
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": 100,
                                "redraw": True
                            },
                            "fromcurrent": True
                        }
                    ],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {
                                "duration": 0,
                                "redraw": True
                            },
                            "mode": "immediate",
                        }
                    ],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }],
        sliders = [sliders_dict]
    ),
    frames = frames
)

fig.show()


