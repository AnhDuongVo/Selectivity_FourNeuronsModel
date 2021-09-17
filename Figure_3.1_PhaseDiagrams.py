### Figure 3.1 Phase diagrams

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rc('font',**{'family':'sans-serif','sans-serif':['CMU sans serif']})
params = {
   'axes.labelsize': 18,
   'axes.spines.top'  : False ,
   'axes.spines.right'  : False ,
   'axes.linewidth' : 1.3,
   'font.size': 18,
   'legend.fontsize': 18,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'xtick.major.top'      : False,
   'ytick.major.right'      : False,
   'figure.figsize': [6, 5],
   'lines.linewidth' : 2,
   #'errorbar.capsize' : 10,
'mathtext.fontset' : 'cm',
"figure.subplot.left"    : 0.2 , # the left side of the subplots of the figure
"figure.subplot.right"   : 0.9   , # the right side of the subplots of the figure
"figure.subplot.bottom"   : 0.17  ,  # the bottom of the subplots of the figure
"figure.subplot.top"     : 0.88 ,
    'axes.unicode_minus': False
}
mpl.rcParams.update(params)

fs = 25
colors = ["C6", "darkred", "darkblue", "indianred", "cornflowerblue"]

#################### model 1 ####################

Ks = np.linspace(-5,0,500, endpoint=False)
Ks_small = np.linspace(-1,0,100, endpoint=False)
Ks_2 = np.linspace(0,1,50, endpoint=True)
Ks_3 = np.linspace(-5,-1,100, endpoint=True)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

upper_limit = (-1/Ks_small).tolist()
upper_limit.append(10)
lower_limit =(-1*Ks_small).tolist()
lower_limit.append(0)
Ks_small=Ks_small.tolist()
Ks_small.append(0)
plt.fill_between(Ks_small,lower_limit, upper_limit, facecolor=colors[0], alpha=0.4,)
plt.fill_between(Ks_2,0,10, facecolor=colors[0], alpha=0.4,)
plt.fill_between(Ks_3,-1/Ks_3,-1*Ks_3, facecolor=colors[0], alpha=0.4,)
plt.fill_between(Ks, -1*Ks, hatch='|',facecolor="none", edgecolor=colors[1],)
plt.fill_between(Ks, -1/Ks, 10, hatch='-',facecolor="none", edgecolor=colors[2],)

plt.ylim(ymin=0, ymax=5)
plt.xlim(xmin=-5,xmax=1)
plt.xlabel(r'$K=(c - ab)$',fontsize=fs)
plt.ylabel(r"$M$ (Input modulation)",fontsize=fs)
plt.tight_layout()
plt.show()
fig.savefig('phase1.pdf', dpi=400)

#################### model 2 ####################

d = np.linspace(0,3,500, endpoint=True)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
M = 1.05

# LS
plt.fill_between(d,1, 3, facecolor=colors[0], alpha=0.4,label = "LS")
plt.fill_between(d,0, 1/M, facecolor=colors[0], alpha=0.4,)

# HS left
lower_limit =(M+d).tolist()
plt.fill_between(d,lower_limit, 3, hatch='|',facecolor="none", edgecolor=colors[1],label="HS left")

# HS right
lower_limit =(1/M+d).tolist()
plt.fill_between(d,lower_limit, 3, hatch='-',facecolor="none", edgecolor=colors[2],label="HS right")

# SHS left
upper_limit = (1/M+d).tolist()
plt.fill_between(d,0, upper_limit, hatch='/',facecolor="none", edgecolor=colors[3],label="SHS left")

# SHS right
lower_limit =(M-2*d).tolist()
upper_limit = (M+d).tolist()
plt.fill_between(d,lower_limit, upper_limit, hatch='\\',facecolor="none", edgecolor=colors[4],label="SHS right")

plt.ylim(ymin=0, ymax=3)
plt.xlim(xmin=0,xmax=3) 
plt.xlabel(r'$d$',fontsize=fs)
plt.ylabel(r'$\gamma=ab+d$',fontsize=fs)
plt.tight_layout()
plt.show()
fig.savefig('phase2.pdf', dpi=400)

#################### model 3 ####################

xi = np.linspace(0,3,500, endpoint=True)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
M = 1.05

# LS
plt.fill_between(xi,(M*xi).tolist(), 4, facecolor=colors[0], alpha=0.4,label = "low")
plt.fill_between(xi,0, (xi/M).tolist(), facecolor=colors[0], alpha=0.4,)

# HS left
plt.fill_between(xi,0, (xi/M).tolist(), hatch='|',facecolor="none", edgecolor=colors[1],label="HS right")

# HS right
lower_limit =(1/M+xi).tolist()
plt.fill_between(xi,0, (xi*M).tolist(), hatch='-',facecolor="none", edgecolor=colors[2],label="HS left")
plt.ylim(ymin=1, ymax=3)
plt.xlim(xmin=0,xmax=3) 
plt.xlabel(r'$\xi=ab$',fontsize=fs)
plt.ylabel(r'$\kappa=be+1$',fontsize=fs)
plt.tight_layout()
plt.show()
fig.savefig('phase3.pdf', dpi=400)

#################### legend ####################

d = np.linspace(0,3,500, endpoint=True)

fig, ax = plt.subplots(figsize=(5,5))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
M = 1.05

# LS
plt.fill_between(d,1, 1, facecolor=colors[0], alpha=0.4,label = "LS")
plt.fill_between(d,0, 0, facecolor=colors[0], alpha=0.4,)

# HS left
lower_limit =(M+d).tolist()
plt.fill_between(d,3, 3, hatch='|',facecolor="none", edgecolor=colors[1],label="HS Left")

# HS right
lower_limit =(1/M+d).tolist()
plt.fill_between(d,3, 3, hatch='-',facecolor="none", edgecolor=colors[2],label="HS Right")

# SHS left
upper_limit = (1/M+d).tolist()
plt.fill_between(d,0, 0, hatch='/',facecolor="none", edgecolor=colors[3],label="SHS Left")

# SHS right
lower_limit =(M-2*d).tolist()
upper_limit = (M+d).tolist()
plt.fill_between(d,0, 0, hatch='\\',facecolor="none", edgecolor=colors[4],label="SHS Right")

plt.ylim(ymin=0, ymax=3)
plt.xlim(xmin=0,xmax=3) 
plt.legend()
fig.savefig('labels.pdf', dpi=400)
plt.show()
