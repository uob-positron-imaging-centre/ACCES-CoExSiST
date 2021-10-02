#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ballistics.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.01.2021


# SymPy is an optional dependency for the ballistics calculations
try:
    import  sympy
except ImportError:
    pass


def ballistic():
    '''Derive the analytical expression of a single particle's trajectory
    when only gravity, buoyancy and drag act on it.

    For the z-dimension, its solution is implicit and computationally expensive
    to find - you can copy and run this code with `uzt = ...` commented out.
    '''

    from sympy import Symbol, Function, Eq

    # Nice printing to terminal
    sympy.init_printing()

    # Define symbols and functions used in integration
    t = Symbol("t")
    t0 = Symbol("t0")

    ux = Function('ux')
    ux0 = Symbol("ux0")
    dux_dt = ux(t).diff(t)

    uy = Function('uy')
    uy0 = Symbol("uy0")
    duy_dt = uy(t).diff(t)

    uz = Function('uz')
    uz0 = Symbol("uz0")
    duz_dt = uz(t).diff(t)

    cd = Symbol("cd")           # Drag coefficient
    d = Symbol("d")             # Particle diameter
    g = Symbol("g")             # Gravitational acceleration
    rho_p = Symbol("rho_p")     # Particle density
    rho_f = Symbol("rho_f")     # Fluid density

    # Define expressions to integrate for velocities in each dimension
    # Eq: d(ux) / dt = expr...
    ux_eq = Eq(dux_dt, -3 / (4 * d) * cd * ux(t) ** 2)
    uy_eq = Eq(duy_dt, -3 / (4 * d) * cd * uy(t) ** 2)
    uz_eq = Eq(duz_dt, -3 / (4 * d) * cd * uz(t) ** 2 - g + rho_f * g / rho_p)

    uxt = sympy.dsolve(
        ux_eq,
        ics = {ux(t0): ux0},
    )

    uyt = sympy.dsolve(
        uy_eq,
        ics = {uy(t0): uy0},
    )

    # This takes a few minutes and returns a nasty implicit solution!
    uzt = sympy.dsolve(
        uz_eq,
        ics = {uz(t0): uz0},
    )

    return uxt, uyt, uzt




def ballistic_approx():
    '''Derive the analytical expression of a single particle's trajectory
    when only gravity, buoyancy and *constant drag* act on it.

    For small changes in velocity, the drag is effectively constant,
    simplifying the solution tremendously. Given three points, it would be
    possible to infer the value of drag.
    '''

    from sympy import Symbol, Function, Eq

    # Nice printing to terminal
    sympy.init_printing()

    # Define symbols and functions used in integration
    t = Symbol("t")
    t0 = Symbol("t_0")

    ux = Function('u_x')
    ux0 = Symbol("u_x0")
    dux_dt = ux(t).diff(t)

    uy = Function('u_y')
    uy0 = Symbol("u_y0")
    duy_dt = uy(t).diff(t)

    uz = Function('u_z')
    uz0 = Symbol("u_z0")
    duz_dt = uz(t).diff(t)

    adx = Symbol("a_dx")        # Acceleration due to drag in the x-direction
    ady = Symbol("a_dy")        # Acceleration due to drag in the y-direction
    adz = Symbol("a_dz")        # Acceleration due to drag in the z-direction

    g = Symbol("g")             # Gravitational acceleration
    rho_p = Symbol("rho_p")     # Particle density
    rho_f = Symbol("rho_f")     # Fluid density

    # Define expressions to integrate for velocities in each dimension
    # Eq: d(ux) / dt = expr...
    ux_eq = Eq(dux_dt, -adx)        # Some constant acceleration due to drag
    uy_eq = Eq(duy_dt, -ady)
    uz_eq = Eq(duz_dt, -adz - g + rho_f * g / rho_p)

    # Solve equations of motion to find analytical expressions for velocities
    # in each dimension as functions of time
    uxt = sympy.dsolve(
        ux_eq,
        ics = {ux(t0): ux0},
    )

    uyt = sympy.dsolve(
        uy_eq,
        ics = {uy(t0): uy0},
    )

    uzt = sympy.dsolve(
        uz_eq,
        ics = {uz(t0): uz0},
    )

    # Use expressions for velocities to derive positions as functions of time
    x = Function("x")
    x0 = Symbol("x_0")
    dx_dt = x(t).diff(t)

    y = Function("y")
    y0 = Symbol("y_0")
    dy_dt = y(t).diff(t)

    z = Function("z")
    z0 = Symbol("z_0")
    dz_dt = z(t).diff(t)

    # Define expressions to integrate for positions in each dimension
    # Eq: d(x) / dt = expr...
    x_eq = Eq(dx_dt, uxt.rhs)
    y_eq = Eq(dy_dt, uyt.rhs)
    z_eq = Eq(dz_dt, uzt.rhs)

    # Solve to find particle positions wrt. time
    xt = sympy.dsolve(
        x_eq,
        ics = {x(t0): x0},
    )

    yt = sympy.dsolve(
        y_eq,
        ics = {y(t0): y0},
    )

    zt = sympy.dsolve(
        z_eq,
        ics = {z(t0): z0},
    )

    return xt, yt, zt
