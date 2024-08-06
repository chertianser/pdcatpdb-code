import sys
import numpy as np
import scipy.integrate

OFile = 'microkinetics/ob-3o-tbubrettphos/ob-3o-tbubrettphos-results.csv'

# Parameters
species = 23
trep = 1000.0 # s
dt = 100.0 # maximum timestep for solving the system
tfin = 1000000.0 # Final Time
xini = np.zeros(species)
xini[18] = 0.02
xini[17] = 0.001
xini[19] = 0.1
# Model at T=333.15 K
def model(t,x):
    dxdt = np.zeros(23)
    
    #Constants
    k00 = 3.6932830108e+10
    k01 = 3.6932830108e+10
    k02 = 4.6643627286e+09
    k03 = 3.6932830108e+10
    k04 = 8.2458712369e-01
    k05 = 1.0438502579e+08
    k06 = 1.6047586668e+09
    k07 = 6.3713966576e-15
    k08 = 4.1764579613e+05
    k09 = 2.8964387389e+09
    k10 = 8.4923754201e-12
    k11 = 8.2976284686e+04
    k12 = 3.6932830108e+10
    k13 = 5.7234213614e-12
    k14 = 9.3084663287e+03
    k15 = 5.3515262043e+03
    k16 = 3.1990484916e+06
    k17 = 5.7996427089e+00
    k18 = 4.4109669052e-08
    k19 = 6.2375549571e-10
    k20 = 9.5714977852e-02
    k21 = 6.8007614385e+00
    k22 = 3.6932830108e+10
    k23 = 1.8613376310e-01
    k24 = 2.0763166310e-01
    k25 = 1.2284502988e-02
    k26 = 2.0305456022e+08
    k27 = 8.0302261972e+06
    k28 = 2.4326922462e-05
    k29 = 8.8567415907e-06
    k30 = 6.0409043507e+00
    k31 = 8.1106793724e-03
    k32 = 3.6932830108e+10
    k33 = 5.2810194341e+09
    k34 = 3.6932830108e+10
    k35 = 1.0944761179e+07
    k36 = 2.6634473467e+06
    k37 = 3.6932830108e+10
    k38 = 3.6932830108e+10
    k39 = 7.2223019984e+08
    k40 = 3.6932830108e+10
    k41 = 5.1810604614e+06
    
    #Ratelaws
    r00 = k00*x[17]
    r01 = k01*x[0]*x[0]
    r02 = k02*x[0]*x[18]
    r03 = k03*x[1]
    r04 = k04*x[1]
    r05 = k05*x[2]
    r06 = k06*x[2]
    r07 = k07*x[3]
    r08 = k08*x[3]*x[19]
    r09 = k09*x[4]*x[20]
    r10 = k10*x[4]
    r11 = k11*x[5]
    r12 = k12*x[5]
    r13 = k13*x[0]*x[21]
    r14 = k14*x[1]
    r15 = k15*x[6]
    r16 = k16*x[6]
    r17 = k17*x[7]
    r18 = k18*x[7]*x[19]
    r19 = k19*x[8]*x[20]
    r20 = k20*x[8]
    r21 = k21*x[9]
    r22 = k22*x[9]
    r23 = k23*x[0]*x[22]
    r24 = k24*x[7]
    r25 = k25*x[11]
    r26 = k26*x[11]
    r27 = k27*x[12]
    r28 = k28*x[13]
    r29 = k29*x[14]
    r30 = k30*x[15]
    r31 = k31*x[16]
    r32 = k32*x[12]
    r33 = k33*x[10]*x[22]
    r34 = k34*x[10]*x[19]
    r35 = k35*x[13]
    r36 = k36*x[13]*x[19]
    r37 = k37*x[15]
    r38 = k38*x[16]
    r39 = k39*x[14]*x[19]
    r40 = k40*x[14]
    r41 = k41*x[0]*x[20]
    
    #MassBalances
    dxdt[0] = +2.0*r00-2.0*r01-r02+r03+r12-r13+r22-r23+r40-r41
    dxdt[1] = +r02-r03-r04+r05-r14+r15
    dxdt[2] = +r04-r05-r06+r07
    dxdt[3] = +r06-r07-r08+r09
    dxdt[4] = +r08-r09-r10+r11
    dxdt[5] = +r10-r11-r12+r13
    dxdt[6] = +r14-r15-r16+r17
    dxdt[7] = +r16-r17-r18+r19-r24+r25
    dxdt[8] = +r18-r19-r20+r21
    dxdt[9] = +r20-r21-r22+r23
    dxdt[10] = +r32-r33-r34+r35
    dxdt[11] = +r24-r25-r26+r27
    dxdt[12] = +r26-r27-r32+r33
    dxdt[13] = -r28+r29+r34-r35-r36+r37
    dxdt[14] = +r28-r29+r38-r39-r40+r41
    dxdt[15] = -r30+r31+r36-r37
    dxdt[16] = +r30-r31-r38+r39
    dxdt[17] = -r00+r01
    dxdt[18] = -r02+r03
    dxdt[19] = -r08+r09-r18+r19-r34+r35-r36+r37+r38-r39
    dxdt[20] = +r08-r09+r18-r19+r40-r41
    dxdt[21] = +r12-r13
    dxdt[22] = +r22-r23+r32-r33
    
    return dxdt
    
def jacobian(t,x):
    Jac = np.zeros(shape=(23,23))
    
    #Constants
    k00 = 3.6932830108e+10
    k01 = 3.6932830108e+10
    k02 = 4.6643627286e+09
    k03 = 3.6932830108e+10
    k04 = 8.2458712369e-01
    k05 = 1.0438502579e+08
    k06 = 1.6047586668e+09
    k07 = 6.3713966576e-15
    k08 = 4.1764579613e+05
    k09 = 2.8964387389e+09
    k10 = 8.4923754201e-12
    k11 = 8.2976284686e+04
    k12 = 3.6932830108e+10
    k13 = 5.7234213614e-12
    k14 = 9.3084663287e+03
    k15 = 5.3515262043e+03
    k16 = 3.1990484916e+06
    k17 = 5.7996427089e+00
    k18 = 4.4109669052e-08
    k19 = 6.2375549571e-10
    k20 = 9.5714977852e-02
    k21 = 6.8007614385e+00
    k22 = 3.6932830108e+10
    k23 = 1.8613376310e-01
    k24 = 2.0763166310e-01
    k25 = 1.2284502988e-02
    k26 = 2.0305456022e+08
    k27 = 8.0302261972e+06
    k28 = 2.4326922462e-05
    k29 = 8.8567415907e-06
    k30 = 6.0409043507e+00
    k31 = 8.1106793724e-03
    k32 = 3.6932830108e+10
    k33 = 5.2810194341e+09
    k34 = 3.6932830108e+10
    k35 = 1.0944761179e+07
    k36 = 2.6634473467e+06
    k37 = 3.6932830108e+10
    k38 = 3.6932830108e+10
    k39 = 7.2223019984e+08
    k40 = 3.6932830108e+10
    k41 = 5.1810604614e+06
    
    #Non-zero Elements
    Jac[0,0] = -2.0*2.0*k01*x[0]-k02*x[18]-k13*x[21]-k23*x[22]-k41*x[20]
    Jac[0,1] = +k03
    Jac[0,5] = +k12
    Jac[0,9] = +k22
    Jac[0,14] = +k40
    Jac[0,17] = +2.0*k00
    Jac[0,18] = -k02*x[0]
    Jac[0,20] = -k41*x[0]
    Jac[0,21] = -k13*x[0]
    Jac[0,22] = -k23*x[0]
    Jac[1,0] = +k02*x[18]
    Jac[1,1] = -k03-k04-k14
    Jac[1,2] = +k05
    Jac[1,6] = +k15
    Jac[1,18] = +k02*x[0]
    Jac[2,1] = +k04
    Jac[2,2] = -k05-k06
    Jac[2,3] = +k07
    Jac[3,2] = +k06
    Jac[3,3] = -k07-k08*x[19]
    Jac[3,4] = +k09*x[20]
    Jac[3,19] = -k08*x[3]
    Jac[3,20] = +k09*x[4]
    Jac[4,3] = +k08*x[19]
    Jac[4,4] = -k09*x[20]-k10
    Jac[4,5] = +k11
    Jac[4,19] = +k08*x[3]
    Jac[4,20] = -k09*x[4]
    Jac[5,0] = +k13*x[21]
    Jac[5,4] = +k10
    Jac[5,5] = -k11-k12
    Jac[5,21] = +k13*x[0]
    Jac[6,1] = +k14
    Jac[6,6] = -k15-k16
    Jac[6,7] = +k17
    Jac[7,6] = +k16
    Jac[7,7] = -k17-k18*x[19]-k24
    Jac[7,8] = +k19*x[20]
    Jac[7,11] = +k25
    Jac[7,19] = -k18*x[7]
    Jac[7,20] = +k19*x[8]
    Jac[8,7] = +k18*x[19]
    Jac[8,8] = -k19*x[20]-k20
    Jac[8,9] = +k21
    Jac[8,19] = +k18*x[7]
    Jac[8,20] = -k19*x[8]
    Jac[9,0] = +k23*x[22]
    Jac[9,8] = +k20
    Jac[9,9] = -k21-k22
    Jac[9,22] = +k23*x[0]
    Jac[10,10] = -k33*x[22]-k34*x[19]
    Jac[10,12] = +k32
    Jac[10,13] = +k35
    Jac[10,19] = -k34*x[10]
    Jac[10,22] = -k33*x[10]
    Jac[11,7] = +k24
    Jac[11,11] = -k25-k26
    Jac[11,12] = +k27
    Jac[12,10] = +k33*x[22]
    Jac[12,11] = +k26
    Jac[12,12] = -k27-k32
    Jac[12,22] = +k33*x[10]
    Jac[13,10] = +k34*x[19]
    Jac[13,13] = -k28-k35-k36*x[19]
    Jac[13,14] = +k29
    Jac[13,15] = +k37
    Jac[13,19] = +k34*x[10]-k36*x[13]
    Jac[14,0] = +k41*x[20]
    Jac[14,13] = +k28
    Jac[14,14] = -k29-k39*x[19]-k40
    Jac[14,16] = +k38
    Jac[14,19] = -k39*x[14]
    Jac[14,20] = +k41*x[0]
    Jac[15,13] = +k36*x[19]
    Jac[15,15] = -k30-k37
    Jac[15,16] = +k31
    Jac[15,19] = +k36*x[13]
    Jac[16,14] = +k39*x[19]
    Jac[16,15] = +k30
    Jac[16,16] = -k31-k38
    Jac[16,19] = +k39*x[14]
    Jac[17,0] = +2.0*k01*x[0]
    Jac[17,17] = -k00
    Jac[18,0] = -k02*x[18]
    Jac[18,1] = +k03
    Jac[18,18] = -k02*x[0]
    Jac[19,3] = -k08*x[19]
    Jac[19,4] = +k09*x[20]
    Jac[19,7] = -k18*x[19]
    Jac[19,8] = +k19*x[20]
    Jac[19,10] = -k34*x[19]
    Jac[19,13] = +k35-k36*x[19]
    Jac[19,14] = -k39*x[19]
    Jac[19,15] = +k37
    Jac[19,16] = +k38
    Jac[19,19] = -k08*x[3]-k18*x[7]-k34*x[10]-k36*x[13]-k39*x[14]
    Jac[19,20] = +k09*x[4]+k19*x[8]
    Jac[20,0] = -k41*x[20]
    Jac[20,3] = +k08*x[19]
    Jac[20,4] = -k09*x[20]
    Jac[20,7] = +k18*x[19]
    Jac[20,8] = -k19*x[20]
    Jac[20,14] = +k40
    Jac[20,19] = +k08*x[3]+k18*x[7]
    Jac[20,20] = -k09*x[4]-k19*x[8]-k41*x[0]
    Jac[21,0] = -k13*x[21]
    Jac[21,5] = +k12
    Jac[21,21] = -k13*x[0]
    Jac[22,0] = -k23*x[22]
    Jac[22,9] = +k22
    Jac[22,10] = -k33*x[22]
    Jac[22,12] = +k32
    Jac[22,22] = -k23*x[0]-k33*x[10]
    
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
