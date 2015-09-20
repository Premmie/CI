function Y = Targetfunction(x)
    targets = dlmread('targets.txt');
    Y = zeros(length(targets),x);
    for i=1:length(targets)
        Y(i,targets(i)) = 1;
    end
end       