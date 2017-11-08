# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 21:10:50 2017

@author: Hansika
"""

import matlab.engine
a = matlab.double([1,4,9,16,25])
b = eng.sqrt(a)
print(b)