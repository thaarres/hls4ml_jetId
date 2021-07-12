import os

Nmax = 200000
i_job = 20

for i in range(i_job):
    first = i*10000
    last = (i+1)*10000
    os.system("python makeh5.py %i %i" %(first, last))