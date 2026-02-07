import string

import highspy
import pyomo.environ as pyo
import pyomo.opt as opt


assert highspy is not None


def create_model(inbounds, outbounds):

    # 
