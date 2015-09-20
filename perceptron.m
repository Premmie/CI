function Y = perceptron(x,w)
X = x*w';
e = 2.71828;
Y = 1/(1+e.^X);
end