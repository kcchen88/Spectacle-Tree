import os,sys,re,gzip,gc
import numpy as np 
import scipy as sp
import scipy.io
from scipy.sparse import *
from scipy.sparse.linalg import *
import pickle
import math

def readData(param_file,AllComb):
	"""
	Read in tree, chromatin marks and chromosomes from the param_file.
	If AllComb is true, consider all possible combinations of marks.
        If AllComb is false, consider all combinations that exist in the genome.
	"""
	exec('from '+param_file[0:-3]+' import *')
	chrs = chromosomes
	Samples = tree.keys()
	seq = {}
	comb = {}
	for S in Samples:
                print S
		i = Samples.index(S)
		seq[i] = {}
		nsegment = 0
		lenChr = {}
		for chr in chrs:
			lenChr[chr] = 0
			filename = inputdir+'/'+S+'_'+chr+'_binary.txt'	
			if os.path.isfile(filename):
				f = open(filename)
			else:
				f = gzip.open(filename+'.gz')
			line = f.readline()
			line = f.readline()
			marks0 = re.split("[\t\n]",line.rstrip())
			for line in f:
				nsegment += 1
				lenChr[chr] += 1
				sig = line.rstrip().replace('\t','')
				sig2 = ''
				for a in range(len(marks0)):
					if (marks0[a] in marks):	
                                                sig2 += sig[a]
				b = int(sig2,2)
				comb[b] = 1
				if (b>0):	
                                        seq[i][nsegment] = b
			f.close()
	(transObs,transObs2) = ({},{})
	if AllComb:
		n = 2**len(marks)
		for i in range(n):
			transObs[i] = i
			transObs2[i] = i
		seq2 = seq
	else:	
		n = len(comb)
		for i in range(n):
			transObs[i] = comb.keys()[i]
			transObs2[comb.keys()[i]] = i
		seq2 = {}
		for i in seq.keys():
			seq2[i] = {}
			for s in seq[i].keys():
				seq2[i][s] = transObs2[seq[i][s]]
	return (seq2,tree,Samples,marks,chrs,n,nsegment,lenChr,transObs)

def get_range(seq,m,n,sample_size):
	""" 
	Compute singular vectors for three views
	U[i,j] : singular vectors for view j in Sample i
	U[i,1] : U from Pairs12
	U[i,2] : U from Pairs21
	U[i,3] : U from Pairs31
	"""
	U = {}
	D = seq.__len__()	
	for i in range(0,D):
                print "Node " + str(i)
		seq_i = {}
		for s in range(0,sample_size):
			triple = (seq[i].get(s,0),seq[i].get(s+1,0),seq[i].get(s+2,0))
			seq_i[s] = triple
		[iw_i,unique_pos_i] = iw_seq(seq_i,sample_size)
		unique_size_i = unique_pos_i.__len__()

		for views in range(1,4):
			cooccur = dok_matrix((n,n))
			for N in unique_pos_i.keys():
				pos = unique_pos_i[N]
				if (views==1):	(pos1,pos2) = (pos,pos+2)
				elif (views==2):	(pos1,pos2) = (pos+1,pos)
				elif (views==3):	(pos1,pos2) = (pos+2,pos)
				cooccur[seq[i].get(pos1,0),seq[i].get(pos2,0)] += iw_i[N]
			cooccur /= float(sample_size)
			(U2,t1,t2) = svds(cooccur,m)
			U[i,views] = np.matrix(sp.zeros((n,m)))
			U2 = np.matrix(U2)
			for k in range(0,m):
				U[i,views][:,k] = np.sign(U2[0,m-k-1])*U2[:,m-k-1]
	return U


def iw_seq(seq_i,sample_size):
	"""
	Count number of occurrences in the genome for each observation and store
        the genomic position where it occurs for the first time.
	This is used to efficiently go through the whole genome data.
	"""
	iw_i = {}
	unique_pos_i = {}		
	for s in range(0,sample_size):
		if ((seq_i[s]) not in unique_pos_i):
			unique_pos_i[seq_i[s]] = s
		iw_i[seq_i[s]] = iw_i.get(seq_i[s],0) + 1
	return (iw_i,unique_pos_i)

def symmetrize(depth_i,path_i,seq,U,m,n,sample_size,Prod_Proj):
	"""
	Symmetrize the tensor
	"""
	# Count all combinations of triples from all samples; eg six-tuples for 2 samples; 
        # x^u_1,x^u_2,x^u_3, ..., x^r_1,x^r_2,x^r_3 where the tree path is r=root, ..., u.
	seqext = {}
	for s in range(0,sample_size):
		comb = ()
		for j in range(0,depth_i):
			comb += (seq[path_i[j]].get(s,0),seq[path_i[j]].get(s+1,0),seq[path_i[j]].get(s+2,0))
		seqext[s] = comb		
	(iw_i,unique_pos_i) = iw_seq(seqext,sample_size)
	unique_size_i = unique_pos_i.__len__()
	
        # P^{u,Hu}_{2,3}_proj = U[u,2]^T * P^{u,Hu}_{2,3} * U[Hu,3]
	P_2i_3 = symmetrize_2i_3(depth_i,path_i,seq,U,m,n,sample_size,Prod_Proj)
	
        # P^{u,Hu}_{2,1}_proj = U[u,2]^T * P^{u,Hu}_{2,1} * U[Hu,1]
	P_2i_1 = symmetrize_2i_1(depth_i,path_i,seq,U,m,n,sample_size,Prod_Proj)
	
        # P^{Hu,Hu}_{1,3}_proj = U[Hu,1]^T * P^{Hu,Hu}_{1,3} * U[Hu,3]
	P_1_3 = symmetrize_1_3(depth_i,path_i,seq,U,m,n,sample_size,Prod_Proj)
	
        # S1 = P^{u,Hu}_{2,3}_proj * (P^{Hu,Hu}_{1,3}_proj)^{-1}
	# S3 = P^{u,Hu}_{2,1}_proj * (P^{Hu,Hu}_{3,1}_proj)^{-1}
	# M^u_2 = P^{Hu,u}_{1,2}(U[Hu,1]*S1^T,U[u,2]) = S1 * U[Hu,1]^T * P^{Hu,u}_{1,2} * U[u,2] = P^{u,Hu}_{2,3}_proj * (P^{Hu,Hu}_{1,3}_proj)^{-1} * P^{u,Hu}_{1,2}_proj
	M_2i = P_2i_3 * np.linalg.inv(P_1_3) * P_2i_1.transpose()	
	
        # Whitening step
	S2 = sp.sparse.identity(m).todense()
	if (Prod_Proj):	
		(U1,s1,t2) = np.linalg.svd(np.array((M_2i+M_2i.transpose())*.5))
		U2 = np.matrix(U1)
		for k in range(0,m):
			U2[:,k] = np.sign(U2[0,k])*U2[:,k]
			S2[k,k] = 1/np.sqrt(s1[k])
	else:	
		(U1,s1,t2) = svds(lil_matrix((M_2i+M_2i.transpose())*.5),m)
		U1 = np.matrix(U1)
		U2 = np.matrix(sp.zeros((n,m)))
		for k in range(0,m):
			U2[:,k] = np.sign(U1[0,m-k-1])*U1[:,m-k-1]
			S2[k,k] = 1/np.sqrt(s1[m-k-1])
	whiten = U2 * S2
	S1 = P_2i_3 * np.linalg.inv(P_1_3)
	S3 = P_2i_1 * np.linalg.inv(P_1_3.transpose())
	A_1 = S1.transpose() * whiten
	A_3 = S3.transpose() * whiten
	return (A_1,A_3,whiten,iw_i,unique_pos_i)


def symmetrize_2i_3(depth_i,path_i,seq,U,m,n,sample_size,Prod_Proj):
	"""
	Project view 3
	"""
	# Count all combinations of (x^u_2, x^u_3, ..., x^r_3) where the tree path for u, Hu is (r=root, ..., u).
	seq_2i_3 = {}
	for s in range(0,sample_size):
		comb_2i_3 = (seq[path_i[0]].get(s+1,0),)
		for j in range(depth_i):	
			comb_2i_3 += (seq[path_i[j]].get(s+2,0),)
		seq_2i_3[s] = comb_2i_3
	(iw_2i_3,unique_pos_2i_3) = iw_seq(seq_2i_3,sample_size)
	#print unique_pos_2i_3.__len__()
	# P^{u,Hu}_{2,3}_proj = U[u,2]^T * P^{u,Hu}_{2,3} * U[Hu,3]
	if (Prod_Proj):	P_2i_3 = np.matrix(sp.zeros((m,m**depth_i)))	
	else:	P_2i_3 = np.matrix(sp.zeros((n,m**depth_i)))	
	for N in unique_pos_2i_3.keys():
		pos = unique_pos_2i_3[N]
		if (Prod_Proj):	y_2i = U[path_i[0],2][seq[path_i[0]].get(pos+1,0),:].transpose()
		else:	y_2i = seq[path_i[0]].get(pos+1,0)
		y_3 = np.matrix([[1]])
		for j in range(depth_i):
			y_3 = np.kron(y_3,U[path_i[j],3][seq[path_i[j]].get(pos+2,0),:].transpose())
		if (Prod_Proj):	P_2i_3 = P_2i_3 + y_2i * y_3.transpose() * iw_2i_3[N]
		else:	P_2i_3[y_2i,:] = P_2i_3[y_2i,:] + y_3.transpose() * iw_2i_3[N]
	P_2i_3 /= float(sample_size)
	return P_2i_3


def symmetrize_2i_1(depth_i,path_i,seq,U,m,n,sample_size,Prod_Proj):
	"""
	Project view 1
	"""
	# Count all combinations of (x^u_2, x^u_1, ..., x^r_1) where the tree path for u, Hu is (r=root, ..., u).
	seq_2i_1 = {}
	for s in range(0,sample_size):
		comb_2i_1 = (seq[path_i[0]].get(s+1,0),)
		for j in range(depth_i):	
			comb_2i_1 += (seq[path_i[j]].get(s,0),)
		seq_2i_1[s] = comb_2i_1
	(iw_2i_1,unique_pos_2i_1) = iw_seq(seq_2i_1,sample_size)
	#print unique_pos_2i_1.__len__()
	# P^{u,Hu}_{2,1}_proj = U[u,2]^T * P^{u,Hu}_{2,1} * U[Hu,1]
	if (Prod_Proj):	
                P_2i_1 = np.matrix(sp.zeros((m,m**depth_i)))	
	else:	
                P_2i_1 = np.matrix(sp.zeros((n,m**depth_i)))	
	for N in unique_pos_2i_1.keys():
		pos = unique_pos_2i_1[N]
		if (Prod_Proj):	y_2i = U[path_i[0],2][seq[path_i[0]].get(pos+1,0),:].transpose()
		else:	y_2i = seq[path_i[0]].get(pos+1,0)
		y_1 = np.matrix([[1]])
		for j in range(depth_i):
			y_1 = np.kron(y_1,U[path_i[j],1][seq[path_i[j]].get(pos,0),:].transpose())
		if (Prod_Proj):	P_2i_1 = P_2i_1 + y_2i * y_1.transpose() * iw_2i_1[N]
		else:	P_2i_1[y_2i,:] = P_2i_1[y_2i,:] + y_1.transpose() * iw_2i_1[N]
	P_2i_1 /= float(sample_size)
	return P_2i_1


def symmetrize_1_3(depth_i,path_i,seq,U,m,n,sample_size,Prod_Proj):
	"""
	Project views 1 and 3
	"""
	# Count all combinations of (x^u_1, x^u_3, ..., x^r_1, x^r_3) where the tree path for u, Hu is (r=root, ..., u).
	seq_1_3 = {}
	for s in range(0,sample_size):
		comb_1_3 = ()
		for j in range(depth_i):	
			comb_1_3 += (seq[path_i[j]].get(s,0),seq[path_i[j]].get(s+2,0))
		seq_1_3[s] = comb_1_3	
	(iw_1_3,unique_pos_1_3) = iw_seq(seq_1_3,sample_size)

	# P^{Hu,Hu}_{1,3}_proj = U[Hu,1]^T * P^{Hu,Hu}_{1,3} * U[Hu,3]

	P_1_3s = {}
	for j in range(depth_i):
		P_1_3s[j] = np.matrix(sp.zeros((m,m)))
	for N in unique_pos_1_3.keys():
		pos = unique_pos_1_3[N]

		for j in range(depth_i):
			y_1 = U[path_i[j],1][seq[path_i[j]].get(pos,0),:].transpose()
			y_3 = U[path_i[j],3][seq[path_i[j]].get(pos+2,0),:].transpose()
			P_1_3s[j] = P_1_3s[j] + y_1 * y_3.transpose() * iw_1_3[N]
	for j in range(depth_i):
		P_1_3s[j] /= float(sample_size)

	P_1_3 = np.matrix([[1]])
	for j in range(depth_i):
		P_1_3 = np.kron(P_1_3,P_1_3s[j])
	return P_1_3


def precomputeTensorSlices(seq,U,A_1,A_3,i,path_i,m,sample_size,iw_i,unique_pos_i):
	"""	
	Precompution step to efficiently perform the computation at the next iteration.
	M^u_3 = P^{Hu,u,Hu}_{1,2,3}(U[Hu,1]*S1^T,U[u,2],U[Hu,3]*S3^T) 
	A_1 = S1^T * W
	A_3 = S3^T * W
	i-th slice of M_3: [M_3]_i = A_1^T * U[Hu,1]^T * [P^{Hu,u,Hu}_{1,2,3}]_i * U[Hu,3] * A_3 
	= W^T * (U[Hu,1]*S1^T)^T * [P^{Hu,u,Hu}_{1,2,3}]_i * (U[Hu,3]*S3^T) * W
	"""
	M_3_i = {}
	for N in unique_pos_i.keys():
		pos = unique_pos_i[N]
		s_2i = seq[i].get(pos+1,0)
		M_3_i[s_2i] = M_3_i.get(s_2i,sp.zeros((m,m)))
		y_1 = np.matrix([[1]])
		y_3 = np.matrix([[1]])
		for j in range(path_i.__len__()):
			y_1 = kron(y_1,U[path_i[j],1][seq[path_i[j]].get(pos,0),:].transpose())
			y_3 = kron(y_3,U[path_i[j],3][seq[path_i[j]].get(pos+2,0),:].transpose())
		M_3_i[s_2i] = M_3_i[s_2i] + (A_1.transpose()*y_1) * (A_3.transpose()*y_3).transpose() * iw_i[N]
	for s_2i in M_3_i.keys():
		M_3_i[s_2i] /= float(sample_size)
	return M_3_i

def simult_power_iter(seq,U,M_3_i,whiten,i,path_i,m,n,sample_size,iw_i,unique_pos_i,Prod_Proj,cutoff):
        """
        Perform the simultaneous tensor power method to decompose the tensor.
        """
	m2 = m
	nIter = 20 #max number of iterations
	[V,r1] = np.linalg.qr(np.random.randn(m,m))
	V_n = sp.zeros((m,m))
	Lambda = sp.zeros((m,1))
	for t in range(nIter):
		for j in range(0,m2):
			# compute: V_n_j = M^u_3(W*V_j,W,W*V_j) / ||M^u_3(W*V_j,W,W*V_j)||
			V_n[:,j] = single_power_iter(seq,U,M_3_i,whiten,i,path_i,m,n,V[:,j],sample_size,iw_i,unique_pos_i,Prod_Proj)

			# compute: Lambda_j = M^u_3(W*V_j,W*V_j,W*V_j) 
			Lambda[j,0] = np.dot(V[:,j].T,V_n[:,j])
			V_n[:,j] /= np.linalg.norm(V_n[:,j])

		[V_n,r1] = np.linalg.qr(V_n) #orthogonalization step	
		similarity = min(abs(np.diagonal(np.dot(V_n[:,0:m2].T,V[:,0:m2]))))
		print similarity
		if (similarity > cutoff):	
                        break
		V = np.array(V_n)
	return (Lambda,V)

def single_power_iter(seq,U,M_3_i,whiten,i,path_i,m,n,v,sample_size,iw_i,unique_pos_i,Prod_Proj):
	"""
        Perform the power iteration for a single vector using the precomputed tensor slices M_3_i
        """
        # i-th slice of M_3: [M_3]_i = W^T * (U[Hu,1]*S1^T)^T * [P^{Hu,u,Hu}_{1,2,3}]_i * (U[Hu,3]*S3^T) * W
	# M^u_3 = P^{Hu,u,Hu}_{1,2,3}(U[Hu,1]*S1^T,U[u,2],U[Hu,3]*S3^T)
	# v_n = M^u_3(Wv,W,Wv) = sum_i{W^T*(U[u,2]_i)^T * (Wv)^T * (U[Hu,1]*S1^T)^T * [P^{Hu,u,Hu}_{1,2,3}]_i * (U[Hu,3]*S3^T) * Wv} 
	v = np.matrix(v).transpose()
	v_n = sp.zeros((m,1))
	I = identity(n).todense()
	for s_2i in M_3_i.keys():
		if (Prod_Proj):	
                        y_2i = U[path_i[0],2][s_2i,:].transpose()
		else:	
                        y_2i = I[s_2i,:].transpose()
		v_n += np.array((whiten.transpose()*y_2i) * (v.transpose() * M_3_i[s_2i] * v))
	return v_n[:,0]

def recover_W(O_d,path,seq,m,sample_size):
	"""	
	Recover initial state distributions
	If u is root r: W^r = (O^r)^+ * P^r_1
	else:	W^u = (O^u)^+ * P^{u,pi(u)}_{1,1} * ((O^u)^+)^T
	"""
	D = seq.__len__()
	W = {}
	for i in range(D):
                if (path[i].__len__()==1):	
                        dims = (m,)
                else:	
                        dims = (m,m)
                seq_i = {}
                for s in range(0,sample_size):
                        comb = () 
                        comb += (seq[path[i][0]].get(s,0),)		
                        if (path[i].__len__()>=2):	
                                comb += (seq[path[i][1]].get(s,0),)		
                        seq_i[s] = comb
                [iw_i,unique_pos_i] = iw_seq(seq_i,sample_size)
                unique_size_i = unique_pos_i.__len__()
                P_1i = sp.zeros(dims)
                for N in unique_pos_i.keys():
                        pos = unique_pos_i[N]
                        y_1_i = np.array(O_d[path[i][0]][:,seq[path[i][0]].get(pos,0)])[:,0]
                        if (path[i].__len__()>=2):	# u is not root
                                y_2_i = np.array(O_d[path[i][1]][:,seq[path[i][1]].get(pos,0)])[:,0]
                                y_1_i = np.outer(y_1_i,y_2_i)	
                        P_1i = P_1i + y_1_i * iw_i[N]
                P_1i /= float(sample_size)
                W[i] = {}
                W[i][0] = P_1i
                W[i][1] = sp.zeros(dims)
                if (path[i].__len__()==1):
                        W[i][1] = abs(W[i][0])/sum(abs(W[i][0]))
                else:
                        for k in range(m):
                                W[i][1][:,k] = abs(W[i][0][:,k])/sum(abs(W[i][0][:,k]))
                print unique_size_i
	return W	


def recover_T(O_d,path,seq,m,sample_size):
	"""
	Recover Transition Matrices
	If u is root: Q^r = (O^u)^+ * P^{r,r}_{2,1} * ((O^r)^+)^T
	else:	Q^u = P^{u,pi(u),u}_{2,2,1}(((O^r)^+)^T,((O^r)^+)^T,((O^r)^+)^T)
	Normalize over the z^u_2 coordinate to get T^u.
	"""
	D = seq.__len__()
	T = {}
	for i in range(D):
		T[i] = {}
		if (path[i].__len__()==1):	# u is root
			P_2i_1i = sp.zeros((m,m))
			seq_i = {}
			for s in range(0,sample_size):
				comb = (seq[path[i][0]].get(s,0),seq[path[i][0]].get(s+1,0))
				seq_i[s] = comb
			[iw_i,unique_pos_i] = iw_seq(seq_i,sample_size)
			unique_size_i = unique_pos_i.__len__()
			for N in unique_pos_i.keys():
				pos = unique_pos_i[N]
				y_1_i = np.array(O_d[path[i][0]][:,seq[path[i][0]].get(pos,0)])[:,0]
				y_2_i = np.array(O_d[path[i][0]][:,seq[path[i][0]].get(pos+1,0)])[:,0]
				P_2i_1i = P_2i_1i + np.outer(y_2_i,y_1_i) * iw_i[N]

			P_2i_1i /= float(sample_size)	
			T[i][0] = abs(P_2i_1i)
			T[i][1] = np.zeros((m,m))
			for k in range(m):
				T[i][1][:,k] = T[i][0][:,k]/sum(T[i][0][:,k])
		else:	# u is not root
			P_2i_1i_2pi = sp.zeros((m,m,m))
			seq_i = {}
			for s in range(0,sample_size):
				comb = (seq[path[i][0]].get(s,0),seq[path[i][0]].get(s+1,0),seq[path[i][1]].get(s+1,0))
				seq_i[s] = comb
			[iw_i,unique_pos_i] = iw_seq(seq_i,sample_size)
			unique_size_i = unique_pos_i.__len__()
			for N in unique_pos_i.keys():
				pos = unique_pos_i[N]
				y_1_i = np.array(O_d[path[i][0]][:,seq[path[i][0]].get(pos,0)])[:,0]
				y_2_i = np.array(O_d[path[i][0]][:,seq[path[i][0]].get(pos+1,0)])[:,0]
				y_2_pi = np.array(O_d[path[i][1]][:,seq[path[i][1]].get(pos+1,0)])[:,0]

				P_2i_1i_2pi = P_2i_1i_2pi + np.reshape(np.kron(y_2_i,np.kron(y_1_i,y_2_pi)),(m,m,m)) * iw_i[N]
			P_2i_1i_2pi /= float(sample_size)	
			T[i][0] = abs(P_2i_1i_2pi)
			T[i][1] = np.zeros((m,m,m))
			for j in range(m):
				for k in range(m):
					T[i][1][:,j,k] = T[i][0][:,j,k]/sum(T[i][0][:,j,k])

	return T

def flatten(P):
	"""
	Flatten parameters for one sample by averaging out other samples.
	"""
	dims = sp.shape(P)
	if (dims.__len__()==1):	# vector
		P_new = P
	elif (dims.__len__()==2):	# matrix
		P_new = sp.zeros((dims[0],))
		for i in range(dims[0]):
			P_new[i] = sum(P[i,:])/float(dims[1])
	elif (dims.__len__()==3):	# tensor
		P_new = sp.zeros((dims[1],dims[2]))
		for i in range(dims[0]):
			for j in range(dims[1]):
				P_new[i,j] = sum(P[i,j,:])/float(dims[1])
	return P_new

def writeModel(O,T,Pi,marks,transObs,nobs,model_file,TreeStructured):
	""" Write model parameters for a single HMM. """ 
	(M2,K) = np.shape(O)
	with open(model_file,'w') as f:
		f.write(str(K)+'\t'+str(nobs)+'\tU\t0\t0\n')
		if TreeStructured:
			for j in range(0,K):
				for i in range(0,K):
					f.write('probinit\t'+str(j+1)+'\t'+str(i+1)+'\t'+str(Pi[i,j])+'\n')
			for k in range(0,K):
				for j in range(0,K):
					for i in range(0,K):
						f.write('transitionprobs\t'+str(k+1)+'\t'+str(j+1)+'\t'+str(i+1)+'\t'+str(T[i,j,k])+'\n')
		else:
			for i in range(0,K):
				f.write('probinit\t'+str(i+1)+'\t'+str(Pi[i])+'\n')
			for j in range(0,K):
				for i in range(0,K):
					f.write('transitionprobs\t'+str(j+1)+'\t'+str(i+1)+'\t'+str(T[i,j])+'\n')
		for j in range(0,K):
			for i in sorted(transObs.keys()):
				f.write('emissionprobs\t'+str(j+1)+'\t'+str(transObs[i])+'\t'+str(O[i,j])+'\n')
	

def writeEmissionMatrix(O,marks,transObs,filename):
	""" Write emission matrix to the output file. """
	(M2,K) = np.shape(O)
	with open(filename,'w') as f:
		PRINT = [str(K),str(len(transObs)),'(Emission order)']
		for mark in marks:  
			PRINT.append(mark)
		f.write('\t'.join(PRINT)+'\n')
		for j in range(0,K):
			for d in sorted(transObs.keys()):
				f.write(str(j+1)+'\t'+str(transObs[d])+'\t'+str(O[d,j])+'\n')

def getEmission(O2,transObs,marks):
	n = len(marks)
	Emission = {}
	for (i,j) in O2.iterkeys():
		Emission[j+1] = Emission.get(j+1,{})
		v = O2[i,j]
		i2 = transObs[i] % (2**n)
		for a in range(0,n):
			mark = marks[n-1-a]
			if (int(i2/2)*2!=i2): Emission[j+1][mark] = Emission[j+1].get(mark,0) + v
			i2 /= 2
			if (i2==0):  break
	return Emission

def printEmission(Emission,marks0):
        """
        Print the emission matrix to stdout
        """
	PRINT = ''
	for mark in marks0:
		PRINT += '\t'+mark
	print PRINT
	for i in sorted(Emission.keys()):
		PRINT = str(i)
		for mark in marks0:
			PRINT += '\t'+str(round(Emission[i].get(mark,0),4))
		print PRINT

# main function of Spectacle-Tree.py
if __name__=="__main__":
	"""
 	[EXAMPLE] python Spectacle-Tree.py 6 Param_Spectacle_Tree.py Output imp_tree
	"""
	try:
		m = int(sys.argv[1]) # Number of chromatin states
		param_file = sys.argv[2]
		outdir = sys.argv[3] # Directory must already exist
		outfile = sys.argv[4]
	except (ValueError, IndexError):
 		print 'usage: python Spectacle-Tree.py 6 Param_Spectacle_Tree.py Output imp_tree'
		quit(0)
	(seq,tree,Samples,marks,chrs,n,nsegment,lenChr,transObs) = readData(param_file,False)
	sample_size = nsegment-2
	D = seq.__len__() # Number of samples
	O = {}
	O_d = {}
	Prod_Proj = True # Boolean: use product projections technique or not
        cutoff = 1.-1e-6 # Cutoff for simultaneous tensor power method 

# Step 1: Compute all the range matrices U's via SVD on the cooccurence matrices
	print 'Computing the range matrices'
	U = get_range(seq,m,n,sample_size)

# Step 2: For each tree node, compute the unique path to the root in the tree
	print 'Computing the path to the root for each tree node'
	depth = {}
	path = {}
	for i in range(0,D):
		path[i] = []
		temp = Samples[i]
		while (temp!='root'):
			path[i].append(Samples.index(temp))
			depth[i] = depth.get(i,0) + 1
			temp = tree[temp]
	
	for i in range(0,D): #iterate over all tree nodes

# 	Step 3: Symmetrization of the matrices
		print 'Performing symmetrization for node %d' % i
		print "Sample: " + tree.keys()[i]
		[A_1,A_3,whiten,iw_i,unique_pos_i] = symmetrize(depth[i],path[i],seq,U,m,n,sample_size,Prod_Proj)

#	 	Step 4: Run simultaneous tensor power method
		print 'Performing tensor decomposition'
		# T^u = M^u_3(W^u,W^u,W^u)  
		# M^u_3 = P^{Hu,u,Hu}_{1,2,3}(U[Hu,1]*S1^T,U[u,2],U[Hu,3]*S3^T) 
		# i-th slice of M_3: [M_3]_i = W^T * (U[Hu,1]*S1^T)^T * [P^{Hu,u,Hu}_{1,2,3}]_i * (U[Hu,3]*S3^T) * W

	 	M_3_i = precomputeTensorSlices(seq,U,A_1,A_3,i,path[i],m,sample_size,iw_i,unique_pos_i)
		[Lambda,V] = simult_power_iter(seq,U,M_3_i,whiten,i,path[i],m,n,sample_size,iw_i,unique_pos_i,Prod_Proj,cutoff) 

# 	Step 5:	Recover the observation matrix of the HMM
		LambdaD = identity(m).todense() 
		for k in range(m):	
                        LambdaD[k,k] = Lambda[k,0]
		# Recover Emission matrix: O^u = U[u,2]*(W^T)^{-1} * V * Diag(Lambda)

 		if (Prod_Proj):	

			O_i = U[i,2] * (np.linalg.inv(whiten.transpose()) * np.matrix(V) * np.matrix(LambdaD))
 		else:	
			O_i = (np.linalg.pinv(whiten.transpose()) * np.matrix(V) * np.matrix(LambdaD))

		B_i = np.matrix(sp.zeros((n,m)))
		for k in range(m):
			B_i[:,k] = O_i[:,k] / sum(O_i[:,k])
		O[i] = {}
		O[i][0] = B_i
		O[i][1] = np.matrix(sp.zeros((n,m))) 
		for k in range(m):
			O[i][1][:,k] = abs(O_i[:,k]) / sum(abs(O_i[:,k]))
		for k in range(m):
			O[i][1][:,k] /= sum(O[i][1][:,k])
		Emission = getEmission(dok_matrix(O[i][1]),transObs,marks)
		printEmission(Emission,marks)

# Step 6: Recover the initial state distribution and transition probabilities of the HMM
	print 'Recovering initial state distribution and transition probabilities of the HMM'
	for i in range(D):
		O_d[i] = np.linalg.pinv(O[i][0])
	W = recover_W(O_d,path,seq,m,sample_size)
	T = recover_T(O_d,path,seq,m,sample_size)

# Step 7: Output the HMM parameters to the output file
	for i in range(D): 
		Sample = tree.keys()[i]
		if len(path[i])==1:
			writeModel(O[i][1],T[i][1],W[i][1],marks,transObs,len(marks),outdir+'/model_comb_'+str(m)+'_'+Sample+'_'+outfile+'.txt',False)
			writeEmissionMatrix(O[i][1],marks,transObs,outdir+'/emissions_comb_'+str(m)+'_'+Sample+'_'+outfile+'.txt')
		else:
		 	W2 = flatten(W[i][1])
			T2 = flatten(T[i][1])
			writeModel(O[i][1],T2,W2,marks,transObs,len(marks),outdir+'/model_comb_'+str(m)+'_'+Sample+'_'+outfile+'_flat.txt',False)





