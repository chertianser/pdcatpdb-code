import sys
import numpy as np
import scipy.integrate

OFile = 'microkinetics/main-3e-pcy3/main-3e-pcy3-results.csv'

# Parameters
species = 16
trep = 0.01 # s
dt = 0.001 # maximum timestep for solving the system
tfin = 10.0 # Final Time
xini = np.zeros(species)
xini[11] = 0.02
xini[10] = 0.001
xini[12] = 0.07
# Model at T=333.15 K
def model(t,x):
    dxdt = np.zeros(16)
    
    #Constants
    k00 = 3.5342421152e+10
    k01 = 3.5342421152e+10
    k02 = 1.1751149682e+09
    k03 = 3.5342421160e+10
    k04 = 4.1826357698e+03
    k05 = 1.0463856627e+06
    k06 = 6.1360220311e+09
    k07 = 2.6992036847e-11
    k08 = 2.6718270331e+06
    k09 = 5.0449611257e+11
    k10 = 3.8315994545e-06
    k11 = 1.1963240136e+05
    k12 = 3.5342421160e+10
    k13 = 7.5690727586e-06
    k14 = 1.0590781710e+07
    k15 = 3.0406985602e+08
    k16 = 1.3660892423e+05
    k17 = 2.1138097600e+03
    k18 = 9.6524102665e+02
    k19 = 1.1611985364e+03
    k20 = 1.7848913902e+01
    k21 = 6.0260273913e-05
    k22 = 3.5342421160e+10
    k23 = 2.7216305439e-02
    
    #Ratelaws
    r00 = k00*x[10]
    r01 = k01*x[0]*x[0]
    r02 = k02*x[0]*x[11]
    r03 = k03*x[1]
    r04 = k04*x[1]
    r05 = k05*x[2]
    r06 = k06*x[2]
    r07 = k07*x[3]
    r08 = k08*x[3]*x[12]
    r09 = k09*x[4]*x[13]
    r10 = k10*x[4]
    r11 = k11*x[5]
    r12 = k12*x[5]
    r13 = k13*x[0]*x[14]
    r14 = k14*x[1]
    r15 = k15*x[6]
    r16 = k16*x[6]
    r17 = k17*x[7]
    r18 = k18*x[7]*x[12]
    r19 = k19*x[8]*x[13]
    r20 = k20*x[8]
    r21 = k21*x[9]
    r22 = k22*x[9]
    r23 = k23*x[0]*x[15]
    
    #MassBalances
    dxdt[0] = +2.0*r00-2.0*r01-r02+r03+r12-r13+r22-r23
    dxdt[1] = +r02-r03-r04+r05-r14+r15
    dxdt[2] = +r04-r05-r06+r07
    dxdt[3] = +r06-r07-r08+r09
    dxdt[4] = +r08-r09-r10+r11
    dxdt[5] = +r10-r11-r12+r13
    dxdt[6] = +r14-r15-r16+r17
    dxdt[7] = +r16-r17-r18+r19
    dxdt[8] = +r18-r19-r20+r21
    dxdt[9] = +r20-r21-r22+r23
    dxdt[10] = -r00+r01
    dxdt[11] = -r02+r03
    dxdt[12] = -r08+r09-r18+r19
    dxdt[13] = +r08-r09+r18-r19
    dxdt[14] = +r12-r13
    dxdt[15] = +r22-r23
    
    return dxdt
    
def jacobian(t,x):
    Jac = np.zeros(shape=(16,16))
    
    #Constants
    k00 = 3.5342421152e+10
    k01 = 3.5342421152e+10
    k02 = 1.1751149682e+09
    k03 = 3.5342421160e+10
    k04 = 4.1826357698e+03
    k05 = 1.0463856627e+06
    k06 = 6.1360220311e+09
    k07 = 2.6992036847e-11
    k08 = 2.6718270331e+06
    k09 = 5.0449611257e+11
    k10 = 3.8315994545e-06
    k11 = 1.1963240136e+05
    k12 = 3.5342421160e+10
    k13 = 7.5690727586e-06
    k14 = 1.0590781710e+07
    k15 = 3.0406985602e+08
    k16 = 1.3660892423e+05
    k17 = 2.1138097600e+03
    k18 = 9.6524102665e+02
    k19 = 1.1611985364e+03
    k20 = 1.7848913902e+01
    k21 = 6.0260273913e-05
    k22 = 3.5342421160e+10
    k23 = 2.7216305439e-02
    
    #Non-zero Elements
    Jac[0,0] = -2.0*2.0*k01*x[0]-k02*x[11]-k13*x[14]-k23*x[15]
    Jac[0,1] = +k03
    Jac[0,5] = +k12
    Jac[0,9] = +k22
    Jac[0,10] = +2.0*k00
    Jac[0,11] = -k02*x[0]
    Jac[0,14] = -k13*x[0]
    Jac[0,15] = -k23*x[0]
    Jac[1,0] = +k02*x[11]
    Jac[1,1] = -k03-k04-k14
    Jac[1,2] = +k05
    Jac[1,6] = +k15
    Jac[1,11] = +k02*x[0]
    Jac[2,1] = +k04
    Jac[2,2] = -k05-k06
    Jac[2,3] = +k07
    Jac[3,2] = +k06
    Jac[3,3] = -k07-k08*x[12]
    Jac[3,4] = +k09*x[13]
    Jac[3,12] = -k08*x[3]
    Jac[3,13] = +k09*x[4]
    Jac[4,3] = +k08*x[12]
    Jac[4,4] = -k09*x[13]-k10
    Jac[4,5] = +k11
    Jac[4,12] = +k08*x[3]
    Jac[4,13] = -k09*x[4]
    Jac[5,0] = +k13*x[14]
    Jac[5,4] = +k10
    Jac[5,5] = -k11-k12
    Jac[5,14] = +k13*x[0]
    Jac[6,1] = +k14
    Jac[6,6] = -k15-k16
    Jac[6,7] = +k17
    Jac[7,6] = +k16
    Jac[7,7] = -k17-k18*x[12]
    Jac[7,8] = +k19*x[13]
    Jac[7,12] = -k18*x[7]
    Jac[7,13] = +k19*x[8]
    Jac[8,7] = +k18*x[12]
    Jac[8,8] = -k19*x[13]-k20
    Jac[8,9] = +k21
    Jac[8,12] = +k18*x[7]
    Jac[8,13] = -k19*x[8]
    Jac[9,0] = +k23*x[15]
    Jac[9,8] = +k20
    Jac[9,9] = -k21-k22
    Jac[9,15] = +k23*x[0]
    Jac[10,0] = +2.0*k01*x[0]
    Jac[10,10] = -k00
    Jac[11,0] = -k02*x[11]
    Jac[11,1] = +k03
    Jac[11,11] = -k02*x[0]
    Jac[12,3] = -k08*x[12]
    Jac[12,4] = +k09*x[13]
    Jac[12,7] = -k18*x[12]
    Jac[12,8] = +k19*x[13]
    Jac[12,12] = -k08*x[3]-k18*x[7]
    Jac[12,13] = +k09*x[4]+k19*x[8]
    Jac[13,3] = +k08*x[12]
    Jac[13,4] = -k09*x[13]
    Jac[13,7] = +k18*x[12]
    Jac[13,8] = -k19*x[13]
    Jac[13,12] = +k08*x[3]+k18*x[7]
    Jac[13,13] = -k09*x[4]-k19*x[8]
    Jac[14,0] = -k13*x[14]
    Jac[14,5] = +k12
    Jac[14,14] = -k13*x[0]
    Jac[15,0] = -k23*x[15]
    Jac[15,9] = +k22
    Jac[15,15] = -k23*x[0]
    
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
