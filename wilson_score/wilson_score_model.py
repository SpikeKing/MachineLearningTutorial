#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import numpy as np


def wilson_score(pos, total, p_z=2.):
    """
    威尔逊得分计算函数
    参考：https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    :param pos: 正例数
    :param total: 总数
    :param p_z: 正太分布的分位数
    :return: 威尔逊得分
    """
    pos_rat = pos * 1. / total * 1.  # 正例比率
    score = (pos_rat + (np.square(p_z) / (2. * total))
             - ((p_z / (2. * total)) * np.sqrt(4. * total * (1. - pos_rat) * pos_rat + np.square(p_z)))) / \
            (1. + np.square(p_z) / total)
    return score


def wilson_score_norm(mean, var, total, p_z=2.):
    """
    威尔逊得分计算函数 正态分布版 支持如5星评价，或百分制评价
    :param mean: 均值
    :param var: 方差
    :param total: 总数
    :param p_z: 正太分布的分位数
    :return: 
    """
    # 归一化，符合正太分布的分位数
    score = (mean + (np.square(p_z) / (2. * total))
             - ((p_z / (2. * total)) * np.sqrt(4. * total * var + np.square(p_z)))) / \
            (1 + np.square(p_z) / total)
    return score


def test_of_values():
    """
    五星评价的归一化实例，百分制类似
    :return: 总数，均值，方差
    """
    max = 5.  # 五星评价的最大值
    min = 1.  # 五星评价的最小值
    values = np.array([1., 2., 3., 4., 5.])  # 示例

    norm_values = (values - min) / (max - min)  # 归一化
    total = norm_values.size  # 总数
    mean = np.mean(norm_values)  # 归一化后的均值
    var = np.var(norm_values)  # 归一化后的方差
    return total, mean, var


total, mean, var = test_of_values()
print "total: %s, mean: %s, var: %s" % (total, mean, var)

print 'score: %s' % wilson_score_norm(mean=mean, var=var, total=total)
print 'score: %s' % wilson_score(90, 90 + 10, p_z=2.)
print 'score: %s' % wilson_score(90, 90 + 10, p_z=6.)
print 'score: %s' % wilson_score(900, 900 + 100, p_z=6.)
