import sys
import numpy as np
import scipy.integrate

OFile = 'microkinetics/simplified/simplified-results.csv'

# Parameters
species = 10
trep = 100000.0 # s
dt = 10000.0 # maximum timestep for solving the system
tfin = 1000000.0 # Final Time
xini = np.zeros(species)
xini[5] = 0.02
xini[4] = 0.001
xini[6] = 0.1
# Model at T=333.15 K
def model(t,x):
    dxdt = np.zeros(10)
    
    #Constants
    k00 = 2.4621886739e+10
    k01 = 2.4621886739e+10
    k02 = 3.7295662371e+10
    k03 = 3.7295662371e+10
    k04 = 4.5223793267e-01
    k05 = 1.8007114031e-17
    k06 = 1.0034903572e+03
    k07 = 1.0034903574e+03
    k08 = 3.7295662379e+10
    k09 = 1.5585808273e-06
    
    #Ratelaws
    r00 = k00*x[4]
    r01 = k01*x[0]*x[0]
    r02 = k02*x[0]*x[5]
    r03 = k03*x[1]
    r04 = k04*x[1]
    r05 = k05*x[2]
    r06 = k06*x[1]*x[6]
    r07 = k07*x[3]*x[7]
    r08 = k08*x[3]
    r09 = k09*x[0]*x[9]
    
    #MassBalances
    dxdt[0] = +2.0*r00-2.0*r01-r02+r03+r08-r09
    dxdt[1] = +r02-r03-r04+r05-r06+r07
    dxdt[2] = +r04-r05
    dxdt[3] = +r06-r07-r08+r09
    dxdt[4] = -r00+r01
    dxdt[5] = -r02+r03
    dxdt[6] = -r06+r07
    dxdt[7] = +r06-r07
    dxdt[9] = +r08-r09
    
    return dxdt
    
def jacobian(t,x):
    Jac = np.zeros(shape=(10,10))
    
    #Constants
    k00 = 2.4621886739e+10
    k01 = 2.4621886739e+10
    k02 = 3.7295662371e+10
    k03 = 3.7295662371e+10
    k04 = 4.5223793267e-01
    k05 = 1.8007114031e-17
    k06 = 1.0034903572e+03
    k07 = 1.0034903574e+03
    k08 = 3.7295662379e+10
    k09 = 1.5585808273e-06
    
    #Non-zero Elements
    Jac[0,0] = -2.0*2.0*k01*x[0]-k02*x[5]-k09*x[9]
    Jac[0,1] = +k03
    Jac[0,3] = +k08
    Jac[0,4] = +2.0*k00
    Jac[0,5] = -k02*x[0]
    Jac[0,9] = -k09*x[0]
    Jac[1,0] = +k02*x[5]
    Jac[1,1] = -k03-k04-k06*x[6]
    Jac[1,2] = +k05
    Jac[1,3] = +k07*x[7]
    Jac[1,5] = +k02*x[0]
    Jac[1,6] = -k06*x[1]
    Jac[1,7] = +k07*x[3]
    Jac[2,1] = +k04
    Jac[2,2] = -k05
    Jac[3,0] = +k09*x[9]
    Jac[3,1] = +k06*x[6]
    Jac[3,3] = -k07*x[7]-k08
    Jac[3,6] = +k06*x[1]
    Jac[3,7] = -k07*x[3]
    Jac[3,9] = +k09*x[0]
    Jac[4,0] = +2.0*k01*x[0]
    Jac[4,4] = -k00
    Jac[5,0] = -k02*x[5]
    Jac[5,1] = +k03
    Jac[5,5] = -k02*x[0]
    Jac[6,1] = -k06*x[6]
    Jac[6,3] = +k07*x[7]
    Jac[6,6] = -k06*x[1]
    Jac[6,7] = +k07*x[3]
    Jac[7,1] = +k06*x[6]
    Jac[7,3] = -k07*x[7]
    Jac[7,6] = +k06*x[1]
    Jac[7,7] = -k07*x[3]
    Jac[9,0] = -k09*x[9]
    Jac[9,3] = +k08
    Jac[9,9] = -k09*x[0]
    
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
