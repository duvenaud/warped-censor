% Bincoef.m

N = 10;
Nc = 0:1:200;

o = [];
for v=Nc
    o = [o; nchoosek(v + N, v) * 0.3^v];
end

plot(o);