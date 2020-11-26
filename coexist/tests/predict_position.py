#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : tests.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 05.11.2020


import numpy as np
import plotly.graph_objs as go

import coexist


t0 = 0.
t1 = 0.013215291119800373
t2 = 0.026430582239600747
t3 = 0.03964587335940112

# p0 = p(t0)
p0 = np.array([[0., 0., 0.],
               [0., 37.83956221494102, 29.972669412804233],
               [0., 26.276539892183347, 53.00230860509723]])
p1 = np.array([[0., 0.15246976542027932, 0.5681698553847081],
               [0., 37.9060626019841, 29.718817997536284],
               [0., 26.361703979753734, 52.962831961776864]])
p2 = np.array([[0., 0.30410576268765915, 1.131524137716995],
               [0., 37.97239953644785, 29.463879381130667],
               [0., 26.446793016329224, 52.921677607292615]])

p3, u0 = coexist.Coexist.predict_positions(t0, p0, t1, p1, t2, p2, t3)

p = np.stack((p0, p1, p2, p3), axis = 0)


fig = go.Figure()

for i in range(p.shape[1]):
    fig.add_trace(
        go.Scatter3d(
            x = p[:, i, 0],
            y = p[:, i, 1],
            z = p[:, i, 2],
            mode = "markers",
        )
    )
fig.show()


'''
re = np.logspace(-2, 6, 1000)
cd = 24 / re * (1 + 0.186 * re ** 0.6529) + re * 0.4373 / (re + 7185.35)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x = re,
        y = cd,
    )
)
fig.update_xaxes(type="log")
fig.update_yaxes(type="log")
fig.show()
'''
