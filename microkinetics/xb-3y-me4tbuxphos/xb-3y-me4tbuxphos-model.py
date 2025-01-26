import sys
import numpy as np
import scipy.integrate

OFile = 'microkinetics/xb-3p-me4tbuxphos/xb-3p-me4tbuxphos-results.csv'

# Parameters
species = 16
trep = 1.0 # s
dt = 0.1 # maximum timestep for solving the system
tfin = 1000.0 # Final Time
xini = np.zeros(species)
xini[11] = 0.02
xini[0] = 0.002
xini[12] = 0.07
# Model at T=333.15 K
def model(t,x):
    dxdt = np.zeros(16)
    
    #Constants
    k00 = 1.4984492035e+09
    k01 = 3.6932830108e+10
    k02 = 3.1227283414e+03
    k03 = 3.6932830108e+10
    k04 = 3.6932830108e+10
    k05 = 9.4049722163e-12
    k06 = 1.6191183989e+04
    k07 = 1.3421165982e+11
    k08 = 2.9170346476e-07
    k09 = 1.4681284845e+05
    k10 = 3.6932830108e+10
    k11 = 4.9837425697e-12
    k12 = 1.6864727486e+04
    k13 = 2.4124918871e+05
    k14 = 2.3710521920e+06
    k15 = 2.1714809578e-02
    k16 = 8.7991498302e-28
    k17 = 2.6278089263e-28
    k18 = 5.5198696880e-21
    k19 = 2.1292465469e-21
    k20 = 3.6932830108e+10
    k21 = 4.1491715155e+00
    k22 = 1.4045306801e-14
    k23 = 3.1944241581e-28
    k24 = 9.4245469871e-26
    k25 = 1.2375199341e-12
    
    #Ratelaws
    r00 = k00*x[0]*x[11]
    r01 = k01*x[1]
    r02 = k02*x[1]
    r03 = k03*x[2]
    r04 = k04*x[2]
    r05 = k05*x[3]
    r06 = k06*x[3]*x[12]
    r07 = k07*x[4]*x[13]
    r08 = k08*x[4]
    r09 = k09*x[5]
    r10 = k10*x[5]
    r11 = k11*x[0]*x[14]
    r12 = k12*x[1]
    r13 = k13*x[6]
    r14 = k14*x[6]
    r15 = k15*x[7]
    r16 = k16*x[7]*x[12]
    r17 = k17*x[8]*x[13]
    r18 = k18*x[8]
    r19 = k19*x[9]
    r20 = k20*x[9]
    r21 = k21*x[0]*x[15]
    r22 = k22*x[7]
    r23 = k23*x[10]*x[13]
    r24 = k24*x[10]*x[12]
    r25 = k25*x[8]
    
    #MassBalances
    dxdt[0] = -r00+r01+r10-r11+r20-r21
    dxdt[1] = +r00-r01-r02+r03-r12+r13
    dxdt[2] = +r02-r03-r04+r05
    dxdt[3] = +r04-r05-r06+r07
    dxdt[4] = +r06-r07-r08+r09
    dxdt[5] = +r08-r09-r10+r11
    dxdt[6] = +r12-r13-r14+r15
    dxdt[7] = +r14-r15-r16+r17-r22+r23
    dxdt[8] = +r16-r17-r18+r19+r24-r25
    dxdt[9] = +r18-r19-r20+r21
    dxdt[10] = +r22-r23-r24+r25
    dxdt[11] = -r00+r01
    dxdt[12] = -r06+r07-r16+r17-r24+r25
    dxdt[13] = +r06-r07+r16-r17+r22-r23
    dxdt[14] = +r10-r11
    dxdt[15] = +r20-r21
    
    return dxdt
    
def jacobian(t,x):
    Jac = np.zeros(shape=(16,16))
    
    #Constants
    k00 = 1.4984492035e+09
    k01 = 3.6932830108e+10
    k02 = 3.1227283414e+03
    k03 = 3.6932830108e+10
    k04 = 3.6932830108e+10
    k05 = 9.4049722163e-12
    k06 = 1.6191183989e+04
    k07 = 1.3421165982e+11
    k08 = 2.9170346476e-07
    k09 = 1.4681284845e+05
    k10 = 3.6932830108e+10
    k11 = 4.9837425697e-12
    k12 = 1.6864727486e+04
    k13 = 2.4124918871e+05
    k14 = 2.3710521920e+06
    k15 = 2.1714809578e-02
    k16 = 8.7991498302e-28
    k17 = 2.6278089263e-28
    k18 = 5.5198696880e-21
    k19 = 2.1292465469e-21
    k20 = 3.6932830108e+10
    k21 = 4.1491715155e+00
    k22 = 1.4045306801e-14
    k23 = 3.1944241581e-28
    k24 = 9.4245469871e-26
    k25 = 1.2375199341e-12
    
    #Non-zero Elements
    Jac[0,0] = -k00*x[11]-k11*x[14]-k21*x[15]
    Jac[0,1] = +k01
    Jac[0,5] = +k10
    Jac[0,9] = +k20
    Jac[0,11] = -k00*x[0]
    Jac[0,14] = -k11*x[0]
    Jac[0,15] = -k21*x[0]
    Jac[1,0] = +k00*x[11]
    Jac[1,1] = -k01-k02-k12
    Jac[1,2] = +k03
    Jac[1,6] = +k13
    Jac[1,11] = +k00*x[0]
    Jac[2,1] = +k02
    Jac[2,2] = -k03-k04
    Jac[2,3] = +k05
    Jac[3,2] = +k04
    Jac[3,3] = -k05-k06*x[12]
    Jac[3,4] = +k07*x[13]
    Jac[3,12] = -k06*x[3]
    Jac[3,13] = +k07*x[4]
    Jac[4,3] = +k06*x[12]
    Jac[4,4] = -k07*x[13]-k08
    Jac[4,5] = +k09
    Jac[4,12] = +k06*x[3]
    Jac[4,13] = -k07*x[4]
    Jac[5,0] = +k11*x[14]
    Jac[5,4] = +k08
    Jac[5,5] = -k09-k10
    Jac[5,14] = +k11*x[0]
    Jac[6,1] = +k12
    Jac[6,6] = -k13-k14
    Jac[6,7] = +k15
    Jac[7,6] = +k14
    Jac[7,7] = -k15-k16*x[12]-k22
    Jac[7,8] = +k17*x[13]
    Jac[7,10] = +k23*x[13]
    Jac[7,12] = -k16*x[7]
    Jac[7,13] = +k17*x[8]+k23*x[10]
    Jac[8,7] = +k16*x[12]
    Jac[8,8] = -k17*x[13]-k18-k25
    Jac[8,9] = +k19
    Jac[8,10] = +k24*x[12]
    Jac[8,12] = +k16*x[7]+k24*x[10]
    Jac[8,13] = -k17*x[8]
    Jac[9,0] = +k21*x[15]
    Jac[9,8] = +k18
    Jac[9,9] = -k19-k20
    Jac[9,15] = +k21*x[0]
    Jac[10,7] = +k22
    Jac[10,8] = +k25
    Jac[10,10] = -k23*x[13]-k24*x[12]
    Jac[10,12] = -k24*x[10]
    Jac[10,13] = -k23*x[10]
    Jac[11,0] = -k00*x[11]
    Jac[11,1] = +k01
    Jac[11,11] = -k00*x[0]
    Jac[12,3] = -k06*x[12]
    Jac[12,4] = +k07*x[13]
    Jac[12,7] = -k16*x[12]
    Jac[12,8] = +k17*x[13]+k25
    Jac[12,10] = -k24*x[12]
    Jac[12,12] = -k06*x[3]-k16*x[7]-k24*x[10]
    Jac[12,13] = +k07*x[4]+k17*x[8]
    Jac[13,3] = +k06*x[12]
    Jac[13,4] = -k07*x[13]
    Jac[13,7] = +k16*x[12]+k22
    Jac[13,8] = -k17*x[13]
    Jac[13,10] = -k23*x[13]
    Jac[13,12] = +k06*x[3]+k16*x[7]
    Jac[13,13] = -k07*x[4]-k17*x[8]-k23*x[10]
    Jac[14,0] = -k11*x[14]
    Jac[14,5] = +k10
    Jac[14,14] = -k11*x[0]
    Jac[15,0] = -k21*x[15]
    Jac[15,9] = +k20
    Jac[15,15] = -k21*x[0]
    
    return Jac
    
t = np.arange(0,tfin,trep)
# Time indexes and Out predefinition
solution = scipy.integrate.solve_ivp(fun=model, jac=jacobian, y0=xini,
                                     t_span=(0,tfin), t_eval=t,
                                     method='LSODA', max_step=min(dt,trep),
                                     rtol=1e-07,atol=1e-12)

if not solution.success:
    print(solution.message)
else:
    print(f"""
          nfev={solution.nfev}
          njev={solution.njev}
          nlu={solution.nlu}
          status={solution.status}
          success={solution.success}
          """)
    x = np.zeros(shape=(len(solution.t),species+1))
    x[:,0] = solution.t
    x[:,1:] = solution.y[:,:].transpose()
    np.savetxt(OFile,x,delimiter='\t')
