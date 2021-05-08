function s = hypersphereCapArea(n,r,h)
%https://en.wikipedia.org/wiki/Spherical_cap#Hyperspherical_cap

    s = 0.5*unitHypersphereSurfArea(n)*(r^(n-1))...
        *betainc((2*r*h-h^2)/(r^2),(n-1)/2,0.5);
end