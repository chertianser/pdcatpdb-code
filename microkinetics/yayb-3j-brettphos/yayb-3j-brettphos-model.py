import sys
import numpy as np
import scipy.integrate

OFile = 'microkinetics/yayb-3j-brettphos/yayb-3j-brettphos-results.csv'

# Parameters
species = 18
trep = 100.0 # s
dt = 10.0 # maximum timestep for solving the system
tfin = 100000.0 # Final Time
xini = np.zeros(species)
xini[13] = 0.02
xini[12] = 0.001
xini[14] = 0.1
# Model at T=333.15 K
def model(t,x):
    dxdt = np.zeros(18)
    
    #Constants
    k00 = 3.6932830108e+10
    k01 = 3.6932830108e+10
    k02 = 5.2660239711e+09
    k03 = 3.6932830108e+10
    k04 = 6.5027206561e+01
    k05 = 3.2112003466e+06
    k06 = 1.5855264586e+08
    k07 = 3.0140504252e-16
    k08 = 8.3175116365e+01
    k09 = 3.6932830108e+10
    k10 = 1.2943358429e-06
    k11 = 4.9432736144e+02
    k12 = 3.6932830108e+10
    k13 = 1.3823488942e-05
    k14 = 6.9979863711e+03
    k15 = 4.6028429492e+03
    k16 = 1.3281431174e+04
    k17 = 3.5433351579e+03
    k18 = 1.5427115491e+04
    k19 = 6.6324414267e-02
    k20 = 5.1816068460e-04
    k21 = 3.1682605113e-04
    k22 = 3.6932830108e+10
    k23 = 4.7707679779e-01
    k24 = 4.8519261408e+08
    k25 = 2.0842866226e+06
    k26 = 6.7604021089e+01
    k27 = 7.7714413280e+08
    k28 = 1.1798810515e+00
    k29 = 3.2840854523e+00
    k30 = 3.1657415178e-01
    k31 = 5.6165746244e+03
    
    #Ratelaws
    r00 = k00*x[12]
    r01 = k01*x[0]*x[0]
    r02 = k02*x[0]*x[13]
    r03 = k03*x[1]
    r04 = k04*x[1]
    r05 = k05*x[2]
    r06 = k06*x[2]
    r07 = k07*x[3]
    r08 = k08*x[3]*x[14]
    r09 = k09*x[4]*x[15]
    r10 = k10*x[4]
    r11 = k11*x[5]
    r12 = k12*x[5]
    r13 = k13*x[0]*x[16]
    r14 = k14*x[1]
    r15 = k15*x[6]
    r16 = k16*x[6]
    r17 = k17*x[7]
    r18 = k18*x[7]*x[14]
    r19 = k19*x[8]*x[15]
    r20 = k20*x[8]
    r21 = k21*x[9]
    r22 = k22*x[9]
    r23 = k23*x[0]*x[17]
    r24 = k24*x[1]
    r25 = k25*x[10]
    r26 = k26*x[10]
    r27 = k27*x[2]
    r28 = k28*x[1]*x[14]
    r29 = k29*x[11]
    r30 = k30*x[11]
    r31 = k31*x[2]*x[14]
    
    #MassBalances
    dxdt[0] = +2.0*r00-2.0*r01-r02+r03+r12-r13+r22-r23
    dxdt[1] = +r02-r03-r04+r05-r14+r15-r24+r25-r28+r29
    dxdt[2] = +r04-r05-r06+r07+r26-r27+r30-r31
    dxdt[3] = +r06-r07-r08+r09
    dxdt[4] = +r08-r09-r10+r11
    dxdt[5] = +r10-r11-r12+r13
    dxdt[6] = +r14-r15-r16+r17
    dxdt[7] = +r16-r17-r18+r19
    dxdt[8] = +r18-r19-r20+r21
    dxdt[9] = +r20-r21-r22+r23
    dxdt[10] = +r24-r25-r26+r27
    dxdt[11] = +r28-r29-r30+r31
    dxdt[12] = -r00+r01
    dxdt[13] = -r02+r03
    dxdt[14] = -r08+r09-r18+r19-r28+r29+r30-r31
    dxdt[15] = +r08-r09+r18-r19
    dxdt[16] = +r12-r13
    dxdt[17] = +r22-r23
    
    return dxdt
    
def jacobian(t,x):
    Jac = np.zeros(shape=(18,18))
    
    #Constants
    k00 = 3.6932830108e+10
    k01 = 3.6932830108e+10
    k02 = 5.2660239711e+09
    k03 = 3.6932830108e+10
    k04 = 6.5027206561e+01
    k05 = 3.2112003466e+06
    k06 = 1.5855264586e+08
    k07 = 3.0140504252e-16
    k08 = 8.3175116365e+01
    k09 = 3.6932830108e+10
    k10 = 1.2943358429e-06
    k11 = 4.9432736144e+02
    k12 = 3.6932830108e+10
    k13 = 1.3823488942e-05
    k14 = 6.9979863711e+03
    k15 = 4.6028429492e+03
    k16 = 1.3281431174e+04
    k17 = 3.5433351579e+03
    k18 = 1.5427115491e+04
    k19 = 6.6324414267e-02
    k20 = 5.1816068460e-04
    k21 = 3.1682605113e-04
    k22 = 3.6932830108e+10
    k23 = 4.7707679779e-01
    k24 = 4.8519261408e+08
    k25 = 2.0842866226e+06
    k26 = 6.7604021089e+01
    k27 = 7.7714413280e+08
    k28 = 1.1798810515e+00
    k29 = 3.2840854523e+00
    k30 = 3.1657415178e-01
    k31 = 5.6165746244e+03
    
    #Non-zero Elements
    Jac[0,0] = -2.0*2.0*k01*x[0]-k02*x[13]-k13*x[16]-k23*x[17]
    Jac[0,1] = +k03
    Jac[0,5] = +k12
    Jac[0,9] = +k22
    Jac[0,12] = +2.0*k00
    Jac[0,13] = -k02*x[0]
    Jac[0,16] = -k13*x[0]
    Jac[0,17] = -k23*x[0]
    Jac[1,0] = +k02*x[13]
    Jac[1,1] = -k03-k04-k14-k24-k28*x[14]
    Jac[1,2] = +k05
    Jac[1,6] = +k15
    Jac[1,10] = +k25
    Jac[1,11] = +k29
    Jac[1,13] = +k02*x[0]
    Jac[1,14] = -k28*x[1]
    Jac[2,1] = +k04
    Jac[2,2] = -k05-k06-k27-k31*x[14]
    Jac[2,3] = +k07
    Jac[2,10] = +k26
    Jac[2,11] = +k30
    Jac[2,14] = -k31*x[2]
    Jac[3,2] = +k06
    Jac[3,3] = -k07-k08*x[14]
    Jac[3,4] = +k09*x[15]
    Jac[3,14] = -k08*x[3]
    Jac[3,15] = +k09*x[4]
    Jac[4,3] = +k08*x[14]
    Jac[4,4] = -k09*x[15]-k10
    Jac[4,5] = +k11
    Jac[4,14] = +k08*x[3]
    Jac[4,15] = -k09*x[4]
    Jac[5,0] = +k13*x[16]
    Jac[5,4] = +k10
    Jac[5,5] = -k11-k12
    Jac[5,16] = +k13*x[0]
    Jac[6,1] = +k14
    Jac[6,6] = -k15-k16
    Jac[6,7] = +k17
    Jac[7,6] = +k16
    Jac[7,7] = -k17-k18*x[14]
    Jac[7,8] = +k19*x[15]
    Jac[7,14] = -k18*x[7]
    Jac[7,15] = +k19*x[8]
    Jac[8,7] = +k18*x[14]
    Jac[8,8] = -k19*x[15]-k20
    Jac[8,9] = +k21
    Jac[8,14] = +k18*x[7]
    Jac[8,15] = -k19*x[8]
    Jac[9,0] = +k23*x[17]
    Jac[9,8] = +k20
    Jac[9,9] = -k21-k22
    Jac[9,17] = +k23*x[0]
    Jac[10,1] = +k24
    Jac[10,2] = +k27
    Jac[10,10] = -k25-k26
    Jac[11,1] = +k28*x[14]
    Jac[11,2] = +k31*x[14]
    Jac[11,11] = -k29-k30
    Jac[11,14] = +k28*x[1]+k31*x[2]
    Jac[12,0] = +2.0*k01*x[0]
    Jac[12,12] = -k00
    Jac[13,0] = -k02*x[13]
    Jac[13,1] = +k03
    Jac[13,13] = -k02*x[0]
    Jac[14,1] = -k28*x[14]
    Jac[14,2] = -k31*x[14]
    Jac[14,3] = -k08*x[14]
    Jac[14,4] = +k09*x[15]
    Jac[14,7] = -k18*x[14]
    Jac[14,8] = +k19*x[15]
    Jac[14,11] = +k29+k30
    Jac[14,14] = -k08*x[3]-k18*x[7]-k28*x[1]-k31*x[2]
    Jac[14,15] = +k09*x[4]+k19*x[8]
    Jac[15,3] = +k08*x[14]
    Jac[15,4] = -k09*x[15]
    Jac[15,7] = +k18*x[14]
    Jac[15,8] = -k19*x[15]
    Jac[15,14] = +k08*x[3]+k18*x[7]
    Jac[15,15] = -k09*x[4]-k19*x[8]
    Jac[16,0] = -k13*x[16]
    Jac[16,5] = +k12
    Jac[16,16] = -k13*x[0]
    Jac[17,0] = -k23*x[17]
    Jac[17,9] = +k22
    Jac[17,17] = -k23*x[0]
    
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
