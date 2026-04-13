"""
Modified Bessel Functions
Based on Abramowitz and Stegun, sections 9.8, 9.6.2.6
"""

import numpy as np


def modbess_i0(x):
    """
    Modified Bessel function of the first kind of order 0
    Test values for y = exp(-x)*I0(x):
    x = 0.1, y = 0.9071009258
    x = 1.0, y = 0.4657596077
    x = 3.0, y = 0.2430003542
    x = 5.0, y = 0.1835408126
    """
    x = float(x)
    t = x / 3.75
    
    if x <= 3.75:
        # Abramowitz and Stegun, section 9.8.1: I0(x) error < 1.6e-7
        a = np.array([3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813])
        t2 = t**2
        i0 = 1.0 + a[0]*t2*(1.0 + a[1]/a[0]*t2*(1.0 + a[2]/a[1]*t2*(1.0 + 
                    a[3]/a[2]*t2*(1.0 + a[4]/a[3]*t2*(1.0 + a[5]/a[4]*t2)))))
    else:
        # Abramowitz and Stegun, section 9.8.2: sqrt(x)*exp(-x)*I0(x) error < 1.9e-7
        a = np.array([0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281,
                     -0.02057706, 0.02635537, -0.01647633, 0.00392377])
        i0 = a[0] + a[1]/t*(1.0 + a[2]/a[1]/t*(1.0 + a[3]/a[2]/t*(1.0 +
             a[4]/a[3]/t*(1.0 + a[5]/a[4]/t*(1.0 + a[6]/a[5]/t*(1.0 +
             a[7]/a[6]/t*(1.0 + a[8]/a[7]/t)))))))
        i0 *= np.exp(x) / np.sqrt(x)
    
    return i0


def modbess_i1(x):
    """
    Modified Bessel function of the first kind of order 1
    Test values for y = exp(-x)*I1(x):
    x = 0.1, y = 0.0452984468
    x = 1.0, y = 0.2079104154
    x = 3.0, y = 0.1968267133
    x = 5.0, y = 0.1639722669
    """
    x = float(x)
    t = x / 3.75
    
    if x <= 3.75:
        # Abramowitz and Stegun, section 9.8.3: I1(x)/x error < 8e-9
        a = np.array([0.87890594, 0.51498869, 0.15084934, 0.02658733, 0.00301532, 0.00032411])
        t2 = t**2
        i1 = 0.5 + a[0]*t2*(1.0 + a[1]/a[0]*t2*(1.0 + a[2]/a[1]*t2*(1.0 +
                    a[3]/a[2]*t2*(1.0 + a[4]/a[3]*t2*(1.0 + a[5]/a[4]*t2)))))
        i1 *= x
    else:
        # Abramowitz and Stegun, section 9.8.4: sqrt(x)*exp(-x)*I1(x) error < 2.2e-7
        a = np.array([0.39894228, -0.03988024, -0.00362018, 0.00163801, -0.01031555,
                     0.02282967, -0.02895312, 0.01787654, -0.00420059])
        i1 = a[0] + a[1]/t*(1.0 + a[2]/a[1]/t*(1.0 + a[3]/a[2]/t*(1.0 +
             a[4]/a[3]/t*(1.0 + a[5]/a[4]/t*(1.0 + a[6]/a[5]/t*(1.0 +
             a[7]/a[6]/t*(1.0 + a[8]/a[7]/t)))))))
        i1 *= np.exp(x) / np.sqrt(x)
    
    return i1


def modbess_k0(x):
    """
    Modified Bessel function of the second kind of order 0
    Test values for y = exp(+x)*K0(x):
    x = 0.1, y = 2.6823261023
    x = 1.0, y = 1.1444630797
    x = 3.0, y = 0.6977615980
    x = 5.0, y = 0.5478075643
    x = 10.0, y = 0.3916319344
    """
    x = float(x)
    
    if x <= 2:
        # Abramowitz and Stegun, section 9.8.5: K0(x) error < 1e-8
        a = np.array([0.42278420, 0.23069756, 0.03488590, 0.00262698, 0.00010750, 0.00000740])
        xo2 = x**2 / 4.0
        k0 = -0.57721566 + a[0]*xo2*(1.0 + a[1]/a[0]*xo2*(1.0 + a[2]/a[1]*xo2*(1.0 +
                           a[3]/a[2]*xo2*(1.0 + a[4]/a[3]*xo2*(1.0 + a[5]/a[4]*xo2)))))
        k0 += -np.log(x/2.0)*modbess_i0(x)
    else:
        # Abramowitz and Stegun, section 9.8.6: sqrt(x)*exp(x)*K0(x) error < 1.9e-7
        a = np.array([1.25331414, -0.07832358, 0.02189568, -0.01062446,
                     0.00587872, -0.00251540, 0.00053208])
        xu2 = 2.0 / x
        k0 = a[0] + a[1]*xu2*(1.0 + a[2]/a[1]*xu2*(1.0 + a[3]/a[2]*xu2*(1.0 +
             a[4]/a[3]*xu2*(1.0 + a[5]/a[4]*xu2*(1.0 + a[6]/a[5]*xu2)))))
        k0 *= np.exp(-x) / np.sqrt(x)
    
    return k0


def modbess_k1(x):
    """
    Modified Bessel function of the second kind of order 1
    Test values for y = exp(+x)*K1(x):
    x = 0.1, y = 10.890182683
    x = 1.0, y = 1.6361534863
    x = 3.0, y = 0.8065634800
    x = 5.0, y = 0.6002738587
    x = 10.0, y = 0.4107665704
    """
    x = float(x)
    
    if x <= 2:
        # Abramowitz and Stegun, section 9.8.7: x*K1(x) error < 8e-9
        a = np.array([0.15443144, -0.67278579, -0.18156897, -0.01919402, -0.00110404, -0.00004686])
        xo2 = x**2 / 4.0
        k1 = a[0]*xo2*(1.0 + a[1]/a[0]*xo2*(1.0 + a[2]/a[1]*xo2*(1.0 +
             a[3]/a[2]*xo2*(1.0 + a[4]/a[3]*xo2*(1.0 + a[5]/a[4]*xo2)))))
        k1 += 1.0 + x*np.log(x/2.0)*modbess_i1(x)
        k1 /= x
    else:
        # Abramowitz and Stegun, section 9.8.8: sqrt(x)*exp(x)*K1(x) error < 2.2e-7
        a = np.array([1.25331414, 0.23498619, -0.03655620, 0.01504268,
                     -0.00780353, 0.00325614, -0.00068245])
        xu2 = 2.0 / x
        k1 = a[0] + a[1]*xu2*(1.0 + a[2]/a[1]*xu2*(1.0 + a[3]/a[2]*xu2*(1.0 +
             a[4]/a[3]*xu2*(1.0 + a[5]/a[4]*xu2*(1.0 + a[6]/a[5]*xu2)))))
        k1 *= np.exp(-x) / np.sqrt(x)
    
    return k1


def modbess_k2(x):
    """
    Modified Bessel function of the second kind of order 2
    Test values for y = x^2*K2(x):
    x = 0.1, y = 1.995039646
    x = 1.0, y = 1.624838899
    x = 3.0, y = 0.553594126
    x = 5.0, y = 0.132723593
    Test values for y = exp(x)*K2(x):
    x = 5.0, y = 0.7879171
    x = 10.0, y = 0.47378525
    x = 20.0, y = 0.30708743
    """
    k0 = modbess_k0(x)
    k1 = modbess_k1(x)
    k2 = k0 + 2.0 * k1 / x
    
    return k2
