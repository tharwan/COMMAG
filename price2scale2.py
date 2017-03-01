import numpy as np
import matplotlib.pyplot as plt
from os import path
from p2s_cython import updatePhysics
from utilities import uniqueFilename
pi = np.pi



def update_manipulation(n,p_i_man):

	p_i_man = np.zeros(p_i_man.shape)
	# mask = np.random.rand(p_i_man.shape[0])>0.5
	end = int(np.round(p_i_man.shape[0]*1/4.0))
	if n % 150 != 0:
		p_i_man[:end] = +0.1
	if n % 150 == 0:
		p_i_man[:end] = -0.1
	return p_i_man

# def updatePhysics(r_i_sum,U,Ri):
# 	# C = np.sum(r_i)
# 	C = r_i_sum
# 	R = Ri + 1/C
# 	I = U / R
# 	U_s = U - Ri * I
# 	return U_s,I,U_s*I


# @profile
def run(size=10E3,turns=60000,base_price=1,price_spread=1,price_min=100,R=200,
		Ri=2,U=10,f_prob=0.001,
		U_diff_max=4,U_ok=1,pricewalk=None,t_max=200,t_min=15,base_k=None,manipulate=False):

	R = float(R)
	R_scale = R * size
	Ri = float(Ri)
	U = float(U)
	if base_k is None:
		base_k = np.round(R / Ri) * (1 - (np.random.rand(size)-0.5))
		# base_k = np.round(R / Ri) * np.ones(size)
	print np.mean(base_k)
	U_s_history = np.empty(turns)
	P_history = np.empty(turns)
	p_i_mean = np.empty(turns)
	p_i_max = np.empty(turns)
	p_i_min = np.empty(turns)
	P_man = np.empty(turns)
	
	p_i_man = np.zeros(size)

	if pricewalk is None:
		#define price
		# pricewalk = np.cumsum(np.random.normal(size=turns) * price_spread)
		# pricewalk += -np.min(pricewalk) + base_price
		chi = np.random.normal(scale=1,size=turns)
		P_ = 1
		# v0 = 0.2
		# rh = 0.1
		v0 = 0.004
		rh = 0.02
		pricewalk = np.zeros(turns)
		pricewalk[0] = P_
		for idx in xrange(1,len(pricewalk)):
			pricewalk[idx]= -v0*(pricewalk[idx-1]-P_)+rh*chi[idx-1]+pricewalk[idx-1]
		
		t = np.arange(turns)
		# pricewalk += (np.sin(2*np.pi*t/100))/5
		# pricewalk += P_ + (np.sin(2*np.pi*t/2000))/5

	p_max = np.max(pricewalk)

	Ri_t = (pricewalk-np.min(pricewalk))
	Ri_t /= np.max(Ri_t)
	Ri_t -= 0.5
	Ri_t = Ri * (1 + Ri_t/8)
	# Ri_t = Ri * np.ones(turns) * (1+1/8.0)

	# plt.plot(Ri_t)
	# plt.show()



	# initalize "agent" state
	p_i = pricewalk[0]*0.5 + 0.1*np.random.normal(size=size)
	k_i = base_k.copy()
	r_i = k_i / R_scale
	r_i_sum = np.sum(r_i)
	t_i = np.random.randint(0,t_max,size)
	t_i_max = np.random.normal(t_max,t_max/4,size)
	# t_i_max = t_max

	U_s,I,Power = updatePhysics(r_i_sum,U,Ri_t[0])
	U_ref = U/2
	# p_histo = np.histogram(p_i,10)


	for n in xrange(turns):
		# we can reset all k
		# the user is asumed to be done after every round
		k_i = base_k.copy()
		r_i = k_i / R_scale
		r_i_sum = np.sum(r_i)


		if manipulate:
			if callable(manipulate):
				p_i_man = manipulate(n,p_i_man)
			else:
				p_i_man = update_manipulation(n,p_i_man)
		P = pricewalk[n]
		P += p_i_man
		
		idx_list = np.concatenate([np.argwhere(np.logical_and(p_i > P,t_i > t_min)),
										np.argwhere(t_i > t_i_max)])
		U_s,I,Power = updatePhysics(r_i_sum,U,Ri_t[n])
		

		if len(idx_list) > 0:
			rdn_idx_list = np.random.permutation(idx_list)

			for idx in rdn_idx_list:
				# jump to smaller timescale
				U_s,I,Power = updatePhysics(r_i_sum,U,Ri_t[n])
				

				# voltage difference in % with sign
				U_diff = (((U_s - U_ref) / U_ref) * 100 + U_ok) / U_diff_max
				# if U_diff < -U_diff_max:  # if voltage is below 10% of nominal value
				# 	# we should not do anything for the good of the system
				# 	#print "break", U_diff, n
				# 	break  # we can leave because no more devices should be switched on
				# elif U_diff < 0:
				# 	# will only be executed if case above not true
				if U_diff < U_ok:
					if -U_diff < np.random.rand():
						#print -U_diff/U_diff_max
						r_i_sum -= base_k[idx] / R_scale

						k_i[idx] = base_k[idx] * 2

						# if t_i[idx] < t_i_max[idx]:
						# p_i[idx] = np.random.rand()*p_i[idx]
						p_i[idx] = np.random.rand() * P[idx]

						r_i[idx] = k_i[idx] / R_scale
						r_i_sum += k_i[idx] / R_scale
						t_i[idx] = 0
				else:
					# system still good
					r_i_sum -= base_k[idx] / R_scale

					k_i[idx] = base_k[idx] * 2
					# if t_i[idx] < t_i_max[idx]:
					# p_i[idx] = np.random.rand()*p_i[idx]
					p_i[idx] = np.random.rand() * P[idx]
					r_i[idx] = k_i[idx] / R_scale
					r_i_sum += k_i[idx] / R_scale
					t_i[idx] = 0
				
		# keep track
		U_s_history[n] = U_s/U_ref
		P_history[n] = Power 
		# h = np.histogram(p_i,10)
		# p_i_mean[n] = h[1][np.argmax(h[0])]
		p_i_mean[n] = np.median(p_i) 
		p_i_max[n] = np.max(p_i) 
		p_i_min[n] = np.min(p_i) 
		P_man[n] = np.mean(P)
		# idx_mask = p_i < P
		f_mask = np.random.rand(size)<f_prob
		# mask = np.logical_and(idx_mask,f_mask)
		mask = f_mask
		p_i[mask] += np.random.rand(np.sum(mask)) * (1-p_i[mask])
		t_i += 1
		
	stat = {"price": pricewalk, "U": U_s_history, "mean": p_i_mean,
			"max": p_i_max, "min": p_i_min, "power": P_history, "Ri_t": Ri_t,
			"base_k": base_k,"P_man" : P_man}
		
	return stat, base_k

if __name__ == "__main__":
	import seaborn as sns
	sns.set_context("poster")

	# f = np.load("p2s_I_80.npz")
	# pricewalk = f["price"]
	pricewalk = None
	stat,  base_k  = run(pricewalk=pricewalk)
	pricewalk = stat['price']
	statI, base_k = run(pricewalk=pricewalk,U_diff_max=100000000,base_k=base_k)
	statII, base_k = run(pricewalk=pricewalk,U_diff_max=100000000,base_k=base_k,
							manipulate=True)

	name = uniqueFilename('p2s2_I.npz')
	print name
	np.savez(name,**stat)
	np.savez(uniqueFilename('p2s2_II.npz'),**statI)
	
	plt.figure(figsize=(12,8))
	plt.subplot(211)
	plt.plot(statII["U"]*100,label='manipulated')
	plt.plot(statI["U"]*100,label='w/o local control')
	plt.plot(stat["U"]*100,label='with local control')
	
	plt.legend(loc=0)
	plt.ylabel('Voltage %')
	plt.ylim([80,110])
	
	plt.subplot(212)
	plt.plot(stat["price"],label='price')
	plt.plot(statII["P_man"],label='price man')
	plt.plot(statII["mean"],label='median $p_i$ man.')
	plt.plot(stat["mean"],label='median $p_i$')
	# plt.plot(stat["max"],'b--')
	# plt.plot(stat["min"],'b--')

	plt.plot(statI["mean"],'k',label='median $p_i$ w/o')
	
	# plt.plot(statI["max"],'k--')
	# plt.plot(statI["min"],'k--')
	plt.ylim([0,2])
	
	plt.ylabel('Price')
	plt.legend(loc=0)

	plt.figure(figsize=(12,8))
	plt.plot(statII["power"]/12.7*100,label='manipulate')
	plt.plot(statI["power"]/12.7*100,label='w/o local control')
	plt.plot(stat["power"]/12.7*100,label='with local control')
	
	plt.legend(loc=0)
	plt.ylabel('Power')
	




	plt.show()
