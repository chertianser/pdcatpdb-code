import sys
import numpy as np
import scipy.integrate

OFile = 'microkinetics/ester-3t-tbujohnphos/ester-3t-tbujohnphos-results.csv'

# Parameters
species = 17
trep = 100.0 # s
dt = 10.0 # maximum timestep for solving the system
tfin = 100000.0 # Final Time
xini = np.zeros(species)
xini[12] = 0.02
xini[11] = 0.001
xini[13] = 0.07
# Model at T=333.15 K
def model(t,x):
    dxdt = np.zeros(17)
    
    #Constants
    k00 = 3.5342421152e+10
    k01 = 3.5342421152e+10
    k02 = 2.8784451009e+03
    k03 = 3.5342421152e+10
    k04 = 9.3140108724e+07
    k05 = 3.5342421152e+10
    k06 = 8.5915138197e+05
    k07 = 2.6327121734e+08
    k08 = 1.7878752820e+09
    k09 = 2.5296618177e-13
    k10 = 7.2783322906e+05
    k11 = 6.7159630225e+08
    k12 = 2.8536949609e-10
    k13 = 4.2253529700e+04
    k14 = 3.5342421160e+10
    k15 = 1.1624139580e-09
    k16 = 2.0559524209e+04
    k17 = 1.5848644991e+06
    k18 = 2.2014949157e+07
    k19 = 8.9815394015e+02
    k20 = 6.0811185749e+00
    k21 = 4.0351202569e-01
    k22 = 3.9508010084e+00
    k23 = 1.0717185815e-02
    k24 = 3.5342421160e+10
    k25 = 3.9359504049e-01
    
    #Ratelaws
    r00 = k00*x[11]
    r01 = k01*x[0]*x[0]
    r02 = k02*x[0]*x[12]
    r03 = k03*x[1]
    r04 = k04*x[0]*x[12]
    r05 = k05*x[6]
    r06 = k06*x[1]
    r07 = k07*x[2]
    r08 = k08*x[2]
    r09 = k09*x[3]
    r10 = k10*x[3]*x[13]
    r11 = k11*x[4]*x[14]
    r12 = k12*x[4]
    r13 = k13*x[5]
    r14 = k14*x[5]
    r15 = k15*x[0]*x[15]
    r16 = k16*x[6]
    r17 = k17*x[7]
    r18 = k18*x[7]
    r19 = k19*x[8]
    r20 = k20*x[8]*x[13]
    r21 = k21*x[9]*x[14]
    r22 = k22*x[9]
    r23 = k23*x[10]
    r24 = k24*x[10]
    r25 = k25*x[0]*x[16]
    
    #MassBalances
    dxdt[0] = +2.0*r00-2.0*r01-r02+r03-r04+r05+r14-r15+r24-r25
    dxdt[1] = +r02-r03-r06+r07
    dxdt[2] = +r06-r07-r08+r09
    dxdt[3] = +r08-r09-r10+r11
    dxdt[4] = +r10-r11-r12+r13
    dxdt[5] = +r12-r13-r14+r15
    dxdt[6] = +r04-r05-r16+r17
    dxdt[7] = +r16-r17-r18+r19
    dxdt[8] = +r18-r19-r20+r21
    dxdt[9] = +r20-r21-r22+r23
    dxdt[10] = +r22-r23-r24+r25
    dxdt[11] = -r00+r01
    dxdt[12] = -r02+r03-r04+r05
    dxdt[13] = -r10+r11-r20+r21
    dxdt[14] = +r10-r11+r20-r21
    dxdt[15] = +r14-r15
    dxdt[16] = +r24-r25
    
    return dxdt
    
def jacobian(t,x):
    Jac = np.zeros(shape=(17,17))
    
    #Constants
    k00 = 3.5342421152e+10
    k01 = 3.5342421152e+10
    k02 = 2.8784451009e+03
    k03 = 3.5342421152e+10
    k04 = 9.3140108724e+07
    k05 = 3.5342421152e+10
    k06 = 8.5915138197e+05
    k07 = 2.6327121734e+08
    k08 = 1.7878752820e+09
    k09 = 2.5296618177e-13
    k10 = 7.2783322906e+05
    k11 = 6.7159630225e+08
    k12 = 2.8536949609e-10
    k13 = 4.2253529700e+04
    k14 = 3.5342421160e+10
    k15 = 1.1624139580e-09
    k16 = 2.0559524209e+04
    k17 = 1.5848644991e+06
    k18 = 2.2014949157e+07
    k19 = 8.9815394015e+02
    k20 = 6.0811185749e+00
    k21 = 4.0351202569e-01
    k22 = 3.9508010084e+00
    k23 = 1.0717185815e-02
    k24 = 3.5342421160e+10
    k25 = 3.9359504049e-01
    
    #Non-zero Elements
    Jac[0,0] = -2.0*2.0*k01*x[0]-k02*x[12]-k04*x[12]-k15*x[15]-k25*x[16]
    Jac[0,1] = +k03
    Jac[0,5] = +k14
    Jac[0,6] = +k05
    Jac[0,10] = +k24
    Jac[0,11] = +2.0*k00
    Jac[0,12] = -k02*x[0]-k04*x[0]
    Jac[0,15] = -k15*x[0]
    Jac[0,16] = -k25*x[0]
    Jac[1,0] = +k02*x[12]
    Jac[1,1] = -k03-k06
    Jac[1,2] = +k07
    Jac[1,12] = +k02*x[0]
    Jac[2,1] = +k06
    Jac[2,2] = -k07-k08
    Jac[2,3] = +k09
    Jac[3,2] = +k08
    Jac[3,3] = -k09-k10*x[13]
    Jac[3,4] = +k11*x[14]
    Jac[3,13] = -k10*x[3]
    Jac[3,14] = +k11*x[4]
    Jac[4,3] = +k10*x[13]
    Jac[4,4] = -k11*x[14]-k12
    Jac[4,5] = +k13
    Jac[4,13] = +k10*x[3]
    Jac[4,14] = -k11*x[4]
    Jac[5,0] = +k15*x[15]
    Jac[5,4] = +k12
    Jac[5,5] = -k13-k14
    Jac[5,15] = +k15*x[0]
    Jac[6,0] = +k04*x[12]
    Jac[6,6] = -k05-k16
    Jac[6,7] = +k17
    Jac[6,12] = +k04*x[0]
    Jac[7,6] = +k16
    Jac[7,7] = -k17-k18
    Jac[7,8] = +k19
    Jac[8,7] = +k18
    Jac[8,8] = -k19-k20*x[13]
    Jac[8,9] = +k21*x[14]
    Jac[8,13] = -k20*x[8]
    Jac[8,14] = +k21*x[9]
    Jac[9,8] = +k20*x[13]
    Jac[9,9] = -k21*x[14]-k22
    Jac[9,10] = +k23
    Jac[9,13] = +k20*x[8]
    Jac[9,14] = -k21*x[9]
    Jac[10,0] = +k25*x[16]
    Jac[10,9] = +k22
    Jac[10,10] = -k23-k24
    Jac[10,16] = +k25*x[0]
    Jac[11,0] = +2.0*k01*x[0]
    Jac[11,11] = -k00
    Jac[12,0] = -k02*x[12]-k04*x[12]
    Jac[12,1] = +k03
    Jac[12,6] = +k05
    Jac[12,12] = -k02*x[0]-k04*x[0]
    Jac[13,3] = -k10*x[13]
    Jac[13,4] = +k11*x[14]
    Jac[13,8] = -k20*x[13]
    Jac[13,9] = +k21*x[14]
    Jac[13,13] = -k10*x[3]-k20*x[8]
    Jac[13,14] = +k11*x[4]+k21*x[9]
    Jac[14,3] = +k10*x[13]
    Jac[14,4] = -k11*x[14]
    Jac[14,8] = +k20*x[13]
    Jac[14,9] = -k21*x[14]
    Jac[14,13] = +k10*x[3]+k20*x[8]
    Jac[14,14] = -k11*x[4]-k21*x[9]
    Jac[15,0] = -k15*x[15]
    Jac[15,5] = +k14
    Jac[15,15] = -k15*x[0]
    Jac[16,0] = -k25*x[16]
    Jac[16,10] = +k24
    Jac[16,16] = -k25*x[0]
    
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
