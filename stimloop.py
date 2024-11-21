# import stuff
from simulate import *
from analyze import *
from objects import *
from loop import *

name_all='/Users/mli853/Documents/placerg-main/variables/loop_stimvaryN512e-6.0et2.0.pkl'
name_sum='/Users/mli853/Documents/placerg-main/variables/sum_stimvaryN512e-6.0et2.0.pkl'
labelname= 'stim'

arra='/Users/mli853/Documents/placerg-main/data/a_stim*N512e-6.0et2.0.pkl'
arrenv='/Users/mli853/Documents/placerg-main/data/env_stim*N512e-6.0et2.0.pkl'


globfunc(arra, arrenv, name_all, name_sum, labelname)
