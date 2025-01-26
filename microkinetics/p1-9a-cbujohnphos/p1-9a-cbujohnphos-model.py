import sys
import numpy as np
import scipy.integrate

OFile = 'microkinetics/p1-9a-cbujohnphos/p1-9a-cbujohnphos-results.csv'

# Parameters
species = 20
trep = 1000.0 # s
dt = 100.0 # maximum timestep for solving the system
tfin = 1000000.0 # Final Time
xini = np.zeros(species)
xini[12] = 0.02
xini[17] = 0.001
xini[19] = 0.002
xini[13] = 0.07
# Model at T=333.15 K
def model(t,x):
    dxdt = np.zeros(20)
    
    #Constants
    k00 = 2.4621886739e+10
    k01 = 2.4621886739e+10
    k02 = 8.8800651339e+07
    k03 = 2.4621886739e+10
    k04 = 3.1842820743e+02
    k05 = 1.5607725228e+03
    k06 = 4.7415416026e+09
    k07 = 4.5918421547e-13
    k08 = 7.9074893114e+03
    k09 = 2.4621886739e+10
    k10 = 2.3434283163e-07
    k11 = 2.7071053219e+03
    k12 = 2.4621886739e+10
    k13 = 2.1734792440e-04
    k14 = 2.9993627054e+06
    k15 = 3.2518756235e+05
    k16 = 1.0096350708e+05
    k17 = 8.2515027590e+00
    k18 = 1.2580992939e+01
    k19 = 1.4448243014e+01
    k20 = 5.2952120080e+00
    k21 = 2.5549214276e-04
    k22 = 2.4621886739e+10
    k23 = 7.5582156774e+00
    k24 = 5.2474494385e+02
    k25 = 3.3698412210e-15
    k26 = 3.1534063754e+03
    k27 = 1.7633600518e-14
    k28 = 2.4621886739e+10
    k29 = 2.4621886739e+10
    k30 = 2.4621886739e+10
    k31 = 2.4621886739e+10
    
    #Ratelaws
    r00 = k00*x[11]
    r01 = k01*x[0]*x[0]
    r02 = k02*x[0]*x[12]
    r03 = k03*x[1]
    r04 = k04*x[1]
    r05 = k05*x[2]
    r06 = k06*x[2]
    r07 = k07*x[3]
    r08 = k08*x[3]*x[13]
    r09 = k09*x[4]*x[14]
    r10 = k10*x[4]
    r11 = k11*x[5]
    r12 = k12*x[5]
    r13 = k13*x[0]*x[15]
    r14 = k14*x[1]
    r15 = k15*x[6]
    r16 = k16*x[6]
    r17 = k17*x[7]
    r18 = k18*x[7]*x[13]
    r19 = k19*x[8]*x[14]
    r20 = k20*x[8]
    r21 = k21*x[9]
    r22 = k22*x[9]
    r23 = k23*x[0]*x[16]
    r24 = k24*x[7]*x[19]
    r25 = k25*x[10]*x[14]
    r26 = k26*x[8]*x[19]
    r27 = k27*x[10]*x[13]
    r28 = k28*x[17]
    r29 = k29*x[18]*x[18]
    r30 = k30*x[18]*x[19]
    r31 = k31*x[0]
    
    #MassBalances
    dxdt[0] = +2.0*r00-2.0*r01-r02+r03+r12-r13+r22-r23+r30-r31
    dxdt[1] = +r02-r03-r04+r05-r14+r15
    dxdt[2] = +r04-r05-r06+r07
    dxdt[3] = +r06-r07-r08+r09
    dxdt[4] = +r08-r09-r10+r11
    dxdt[5] = +r10-r11-r12+r13
    dxdt[6] = +r14-r15-r16+r17
    dxdt[7] = +r16-r17-r18+r19-r24+r25
    dxdt[8] = +r18-r19-r20+r21-r26+r27
    dxdt[9] = +r20-r21-r22+r23
    dxdt[10] = +r24-r25+r26-r27
    dxdt[11] = -r00+r01
    dxdt[12] = -r02+r03
    dxdt[13] = -r08+r09-r18+r19+r26-r27
    dxdt[14] = +r08-r09+r18-r19+r24-r25
    dxdt[15] = +r12-r13
    dxdt[16] = +r22-r23
    dxdt[17] = -r28+r29
    dxdt[18] = +2.0*r28-2.0*r29-r30+r31
    dxdt[19] = -r24+r25-r26+r27-r30+r31
    
    return dxdt
    
def jacobian(t,x):
    Jac = np.zeros(shape=(20,20))
    
    #Constants
    k00 = 2.4621886739e+10
    k01 = 2.4621886739e+10
    k02 = 8.8800651339e+07
    k03 = 2.4621886739e+10
    k04 = 3.1842820743e+02
    k05 = 1.5607725228e+03
    k06 = 4.7415416026e+09
    k07 = 4.5918421547e-13
    k08 = 7.9074893114e+03
    k09 = 2.4621886739e+10
    k10 = 2.3434283163e-07
    k11 = 2.7071053219e+03
    k12 = 2.4621886739e+10
    k13 = 2.1734792440e-04
    k14 = 2.9993627054e+06
    k15 = 3.2518756235e+05
    k16 = 1.0096350708e+05
    k17 = 8.2515027590e+00
    k18 = 1.2580992939e+01
    k19 = 1.4448243014e+01
    k20 = 5.2952120080e+00
    k21 = 2.5549214276e-04
    k22 = 2.4621886739e+10
    k23 = 7.5582156774e+00
    k24 = 5.2474494385e+02
    k25 = 3.3698412210e-15
    k26 = 3.1534063754e+03
    k27 = 1.7633600518e-14
    k28 = 2.4621886739e+10
    k29 = 2.4621886739e+10
    k30 = 2.4621886739e+10
    k31 = 2.4621886739e+10
    
    #Non-zero Elements
    Jac[0,0] = -2.0*2.0*k01*x[0]-k02*x[12]-k13*x[15]-k23*x[16]-k31
    Jac[0,1] = +k03
    Jac[0,5] = +k12
    Jac[0,9] = +k22
    Jac[0,11] = +2.0*k00
    Jac[0,12] = -k02*x[0]
    Jac[0,15] = -k13*x[0]
    Jac[0,16] = -k23*x[0]
    Jac[0,18] = +k30*x[19]
    Jac[0,19] = +k30*x[18]
    Jac[1,0] = +k02*x[12]
    Jac[1,1] = -k03-k04-k14
    Jac[1,2] = +k05
    Jac[1,6] = +k15
    Jac[1,12] = +k02*x[0]
    Jac[2,1] = +k04
    Jac[2,2] = -k05-k06
    Jac[2,3] = +k07
    Jac[3,2] = +k06
    Jac[3,3] = -k07-k08*x[13]
    Jac[3,4] = +k09*x[14]
    Jac[3,13] = -k08*x[3]
    Jac[3,14] = +k09*x[4]
    Jac[4,3] = +k08*x[13]
    Jac[4,4] = -k09*x[14]-k10
    Jac[4,5] = +k11
    Jac[4,13] = +k08*x[3]
    Jac[4,14] = -k09*x[4]
    Jac[5,0] = +k13*x[15]
    Jac[5,4] = +k10
    Jac[5,5] = -k11-k12
    Jac[5,15] = +k13*x[0]
    Jac[6,1] = +k14
    Jac[6,6] = -k15-k16
    Jac[6,7] = +k17
    Jac[7,6] = +k16
    Jac[7,7] = -k17-k18*x[13]-k24*x[19]
    Jac[7,8] = +k19*x[14]
    Jac[7,10] = +k25*x[14]
    Jac[7,13] = -k18*x[7]
    Jac[7,14] = +k19*x[8]+k25*x[10]
    Jac[7,19] = -k24*x[7]
    Jac[8,7] = +k18*x[13]
    Jac[8,8] = -k19*x[14]-k20-k26*x[19]
    Jac[8,9] = +k21
    Jac[8,10] = +k27*x[13]
    Jac[8,13] = +k18*x[7]+k27*x[10]
    Jac[8,14] = -k19*x[8]
    Jac[8,19] = -k26*x[8]
    Jac[9,0] = +k23*x[16]
    Jac[9,8] = +k20
    Jac[9,9] = -k21-k22
    Jac[9,16] = +k23*x[0]
    Jac[10,7] = +k24*x[19]
    Jac[10,8] = +k26*x[19]
    Jac[10,10] = -k25*x[14]-k27*x[13]
    Jac[10,13] = -k27*x[10]
    Jac[10,14] = -k25*x[10]
    Jac[10,19] = +k24*x[7]+k26*x[8]
    Jac[11,0] = +2.0*k01*x[0]
    Jac[11,11] = -k00
    Jac[12,0] = -k02*x[12]
    Jac[12,1] = +k03
    Jac[12,12] = -k02*x[0]
    Jac[13,3] = -k08*x[13]
    Jac[13,4] = +k09*x[14]
    Jac[13,7] = -k18*x[13]
    Jac[13,8] = +k19*x[14]+k26*x[19]
    Jac[13,10] = -k27*x[13]
    Jac[13,13] = -k08*x[3]-k18*x[7]-k27*x[10]
    Jac[13,14] = +k09*x[4]+k19*x[8]
    Jac[13,19] = +k26*x[8]
    Jac[14,3] = +k08*x[13]
    Jac[14,4] = -k09*x[14]
    Jac[14,7] = +k18*x[13]+k24*x[19]
    Jac[14,8] = -k19*x[14]
    Jac[14,10] = -k25*x[14]
    Jac[14,13] = +k08*x[3]+k18*x[7]
    Jac[14,14] = -k09*x[4]-k19*x[8]-k25*x[10]
    Jac[14,19] = +k24*x[7]
    Jac[15,0] = -k13*x[15]
    Jac[15,5] = +k12
    Jac[15,15] = -k13*x[0]
    Jac[16,0] = -k23*x[16]
    Jac[16,9] = +k22
    Jac[16,16] = -k23*x[0]
    Jac[17,17] = -k28
    Jac[17,18] = +2.0*k29*x[18]
    Jac[18,0] = +k31
    Jac[18,17] = +2.0*k28
    Jac[18,18] = -2.0*2.0*k29*x[18]-k30*x[19]
    Jac[18,19] = -k30*x[18]
    Jac[19,0] = +k31
    Jac[19,7] = -k24*x[19]
    Jac[19,8] = -k26*x[19]
    Jac[19,10] = +k25*x[14]+k27*x[13]
    Jac[19,13] = +k27*x[10]
    Jac[19,14] = +k25*x[10]
    Jac[19,18] = -k30*x[19]
    Jac[19,19] = -k24*x[7]-k26*x[8]-k30*x[18]
    
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
