#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by C.L.Wang

import matplotlib.pyplot as plt
import numpy as np

from wilson_score.wilson_score_model import wilson_score


def show_wilson():
    value = np.linspace(0.01, 100, 1000)
    u, v = np.meshgrid(value, value)

    fig, ax = plt.subplots(1, 2)
    levels = np.linspace(0, 1, 10)

    cs = ax[0].contourf(u, v, wilson_score(u, u + v), levels=levels)
    cb1 = fig.colorbar(cs, ax=ax[0], format="%.2f")

    cs = ax[1].contourf(u, v, wilson_score(u, u + v, 6.), levels=levels)
    cb2 = fig.colorbar(cs, ax=ax[1], format="%.2f")

    ax[0].set_xlabel(u'pos')
    ax[0].set_ylabel(u'neg')
    cb1.set_label(u'wilson(z=2)')

    ax[1].set_xlabel(u'pos')
    ax[1].set_ylabel(u'neg')
    cb2.set_label(u'wilson(z=6)')

    plt.show()


if __name__ == '__main__':
    show_wilson()
