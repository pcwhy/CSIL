function a = unitHypersphereSurfArea(n)
%https://en.wikipedia.org/wiki/Unit_sphere#:~:text=The%20surface%20area%20of%20an,dimensional%20ball%20of%20radius%20r.
%https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
    a = (2*pi^(n/2))/(gamma(n/2));
end

