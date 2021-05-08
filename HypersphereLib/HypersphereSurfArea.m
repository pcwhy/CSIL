function s = HypersphereSurfArea(n,r)
%https://www.phys.uconn.edu/~rozman/Courses/P2400_17S/downloads/nsphere.pdf
    s = unitHypersphereSurfArea(n)*r^(n-1);
end