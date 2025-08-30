#This a hoomd script create by DoMD
from hoomd_script import *
from sys import argv
import math

s = init.read_xml(argv[1])
#{'A-B': (0.94, 1111.0)} {'A': (0.40556914330358024, 26.93974218357792), 'B': (0.531658254896884, 25.416895154211282)}

for p in s.particles:
    if p.type == "A":
        p.diameter = 0.8
    if p.type == "B":
        p.diameter = 1.0


lj = pair.slj(r_cut=3.0,d_max=1.0)
lj.pair_coeff.set('A', 'A', epsilon=27, sigma=1)
lj.pair_coeff.set('A', 'B', epsilon=26, sigma=1)
lj.pair_coeff.set('B', 'B', epsilon=25, sigma=1)
lj.set_params(mode='shift')
nlist.reset_exclusions(exclusions = ['bond','angle'])
harmonic = bond.harmonic(name='mybond')
harmonic.bond_coeff.set('A-B',k=1111.0, r0=0.94)
ang = angle.harmonic()
ang.set_coeff('B-A-B',k=round(11.0,1),t0=round(math.pi*180/180.0,1))
ang.set_coeff('A-B-A',k=round(11.0,1),t0=round(math.pi*180/180.0,1))
g = group.all()


integrate.mode_standard(dt=0.001)
nve = integrate.nve(group=g,limit=0.01)
run(1e6)
nve.disable()

integrate.mode_standard(dt=0.00001)
nvt = integrate.nvt(group=g,T=1.0,tau=0.65)
xml_ = dump.xml(filename='nvt',period=5e5)
xml_.set_params(position=True, body=True,type=True, image=True, bond=True, angle=True)
pre_ = analyze.log(filename='nvt',quantities=['lx','volume','pressure'],header_prefix='#',period=1000,overwrite=True)
run(1e6)
nvt.disable()
xml_.disable()

integrate.mode_standard(dt=0.0001)
npt = integrate.npt(group=g,T = 1.0, tau=0.65 , P = 0.0, tauP = 50.0)
xml_ = dump.xml(filename='pre',period=5e5)
xml_.set_params(position=True, body=True,type=True, image=True, bond=True, angle=True)
pre = analyze.log(filename='pre.log',quantities=['lx','volume','pressure'],header_prefix='#',period=1000,overwrite=True)
run(1e6 + 1 )
npt.disable()
xml_.disable()
dump.xml(filename='final.xml',position=True, body=True,type=True, image=True, bond=True, angle=True)
## GOOD LUCK!
