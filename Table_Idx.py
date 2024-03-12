class T_Idx:
    # bus indices
    # define bus types
    PQ      = 1
    PV      = 2
    REF     = 3
    NONE    = 4

    # define the indices
    BUS_I       = 0    #  bus number (1 to 29997)
    BUS_TYPE    = 1    #  bus type (1 - PQ bus, 2 - PV bus, 3 - reference bus, 4 - isolated bus)
    PD          = 2    #  Pd, real power demand (MW)
    QD          = 3    #  Qd, reactive power demand (MVAr)
    GS          = 4    #  Gs, shunt conductance (MW at V = 1.0 p.u.)
    BS          = 5    #  Bs, shunt susceptance (MVAr at V = 1.0 p.u.)
    BUS_AREA    = 6    #  area number, 1-100
    VM          = 7    #  Vm, voltage magnitude (p.u.)
    VA          = 8    #  Va, voltage angle (degrees)
    BASE_KV     = 9    #  baseKV, base voltage (kV)
    ZONE        = 10   #  zone, loss zone (1-999)
    VMAX        = 11   #  maxVm, maximum voltage magnitude (p.u.)      (not in PTI format)
    VMIN        = 12   #  minVm, minimum voltage magnitude (p.u.)      (not in PTI format)

    # included in opf solution, not necessarily in input
    # assume objective function has units, u
    LAM_P       = 13   #  Lagrange multiplier on real power mismatch (u/MW)
    LAM_Q       = 14   #  Lagrange multiplier on reactive power mismatch (u/MVAr)
    MU_VMAX     = 15   #  Kuhn-Tucker multiplier on upper voltage limit (u/p.u.)
    MU_VMIN     = 16   #  Kuhn-Tucker multiplier on lower voltage limit (u/p.u.)

    # gen indices
    #  define the indices
    GEN_BUS     = 0    #  bus number
    PG          = 1    #  Pg, real power output (MW)
    QG          = 2    #  Qg, reactive power output (MVAr)
    QMAX        = 3    #  Qmax, maximum reactive power output at Pmin (MVAr)
    QMIN        = 4    #  Qmin, minimum reactive power output at Pmin (MVAr)
    VG          = 5    #  Vg, voltage magnitude setpoint (p.u.)
    MBASE       = 6    #  mBase, total MVA base of this machine, defaults to baseMVA
    GEN_STATUS  = 7    #  status, 1 - machine in service, 0 - machine out of service
    PMAX        = 8    #  Pmax, maximum real power output (MW)
    PMIN        = 9    #  Pmin, minimum real power output (MW)
    PC1         = 10   #  Pc1, lower real power output of PQ capability curve (MW)
    PC2         = 11   #  Pc2, upper real power output of PQ capability curve (MW)
    QC1MIN      = 12   #  Qc1min, minimum reactive power output at Pc1 (MVAr)
    QC1MAX      = 13   #  Qc1max, maximum reactive power output at Pc1 (MVAr)
    QC2MIN      = 14   #  Qc2min, minimum reactive power output at Pc2 (MVAr)
    QC2MAX      = 15   #  Qc2max, maximum reactive power output at Pc2 (MVAr)
    RAMP_AGC    = 16   #  ramp rate for load following/AGC (MW/min)
    RAMP_10     = 17   #  ramp rate for 10 minute reserves (MW)
    RAMP_30     = 18   #  ramp rate for 30 minute reserves (MW)
    RAMP_Q      = 19   #  ramp rate for reactive power (2 sec timescale) (MVAr/min)
    APF         = 20   #  area participation factor
 
    #  included in opf solution, not necessarily in input
    #  assume objective function has units, u
    MU_PMAX     = 21   #  Kuhn-Tucker multiplier on upper Pg limit (u/MW)
    MU_PMIN     = 22   #  Kuhn-Tucker multiplier on lower Pg limit (u/MW)
    MU_QMAX     = 23   #  Kuhn-Tucker multiplier on upper Qg limit (u/MVAr)
    MU_QMIN     = 24   #  Kuhn-Tucker multiplier on lower Qg limit (u/MVAr)

    # branch indices
    #  define the indices
    F_BUS       = 0    #  f, from bus number
    T_BUS       = 1    #  t, to bus number
    BR_R        = 2    #  r, resistance (p.u.)
    BR_X        = 3    #  x, reactance (p.u.)
    BR_B        = 4    #  b, total line charging susceptance (p.u.)
    RATE_A      = 5    #  rateA, MVA rating A (long term rating)
    RATE_B      = 6    #  rateB, MVA rating B (short term rating)
    RATE_C      = 7    #  rateC, MVA rating C (emergency rating)
    TAP         = 8    #  ratio, transformer off nominal turns ratio
    SHIFT       = 9    #  angle, transformer phase shift angle (degrees)
    BR_STATUS   = 10   #  initial branch status, 1 - in service, 0 - out of service
    ANGMIN      = 11   #  minimum angle difference, angle(Vf) - angle(Vt) (degrees)
    ANGMAX      = 12   #  maximum angle difference, angle(Vf) - angle(Vt) (degrees)
 
    #  included in power flow solution, not necessarily in input
    PF          = 13   #  real power injected at "from" bus end (MW)       (not in PTI format)
    QF          = 14   #  reactive power injected at "from" bus end (MVAr) (not in PTI format)
    PT          = 15   #  real power injected at "to" bus end (MW)         (not in PTI format)
    QT          = 16   #  reactive power injected at "to" bus end (MVAr)   (not in PTI format)
 
    #  included in opf solution, not necessarily in input assume objective function has units, u
    MU_SF       = 17   #  Kuhn-Tucker multiplier on MVA limit at "from" bus (u/MVA)
    MU_ST       = 18   #  Kuhn-Tucker multiplier on MVA limit at "to" bus (u/MVA)
    MU_ANGMIN   = 19   #  Kuhn-Tucker multiplier lower angle difference limit (u/degree)
    MU_ANGMAX   = 20   #  Kuhn-Tucker multiplier upper angle difference limit (u/degree)

    # gencost indices
    # define cost models
    PW_LINEAR   = 1
    POLYNOMIAL  = 2
 
    # define the indices
    MODEL       = 0    # cost model, 1 = piecewise linear, 2 = polynomial
    STARTUP     = 1    # startup cost in US dollars
    SHUTDOWN    = 2    # shutdown cost in US dollars
    NCOST       = 3    # number breakpoints in piecewise linear cost function,
                       # or number of coefficients in polynomial cost function
    COST        = 4    # parameters defining total cost function begin in this col
                    # (MODEL = 1) : p0, f0, p1, f1, ..., pn, fn
                   #      where p0 < p1 < ... < pn and the cost f(p) is defined
                   #      by the coordinates (p0,f0), (p1,f1), ..., (pn,fn) of
                   #      the end/break-points of the piecewise linear cost
                   # (MODEL = 2) : cn, ..., c1, c0
                   #      n+1 coefficients of an n-th order polynomial cost fcn,
                   #      starting with highest order, where cost is
                   #      f(p) = cn*p^n + ... + c1*p + c0