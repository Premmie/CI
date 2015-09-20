function Output = Opdracht1(nh,out)
    ni = 10; % Amount of input neurons
    no = 7; % Amount of output neurons
    alpha = 0.1; % Constant of learning rate
    trainingset = 60; % Percentage of data used for training set
    validationset = 20; % Percentage of data used for validation set
    epochs = 20; % Number of epochs to train the network
    Features = dlmread('features.txt'); % Features(sample,feature)
    trs = round(length(Features)*trainingset/100); % Amount of training samples
    vas = round(length(Features)*validationset/100); % Amount of validation samples
    tes = length(Features) - trs - vas; % Amount of test samples
    Wh = rand(nh,ni)*2-1; % Weights of the hidden neurons Wh(neuron,input)
    Wo = rand(no,nh)*2-1; % Weights of the output neurons Wo(neuron,input)
    yh = zeros(1,nh); % Outputs of the hidden neurons
    yo = zeros(1,no); % Outputs of the output neurons
    eo = zeros(1,no); % Error of the output neurons
    dh = zeros(1,nh); % Variable of the hidden neurons
    do = zeros(1,no); % Variable of the output neurons
    MSEtr = zeros(1,trs); % Mean Squared error of the training set per sample
    MSEva = zeros(1,vas); % Mean Squared error of the validation set per sample
    MSEte = zeros(1,tes); % Mean Squared error of the test set per sample
    MeanSquaredErrortr = zeros(1,epochs); % The mean of the MSE of the training set per epoch
    MeanSquaredErrorva = zeros(1,epochs); % The mean of the MSE of the validation set per epoch
    MeanSquaredErrorte = zeros(1,epochs); % The mean of the MSE of the test set
    Targets = Targetfunction(no); % Targets(sample,outputneuron)
    PredictedClass = zeros(1,tes);
    for k = 1:epochs % Calculates new W's every new epoch
        for j = 1:trs % Calculates new W's for every sample in the training set
            for i = 1:nh % Calculates the outputs of the hidden neurons
                yh(1,i) = perceptron(Features(j,:),Wh(i,:));
            end
            for i = 1:no % Calculates the outputs of the output neurons
               yo(1,i) = perceptron(yh,Wo(i,:));
            end
            for i = 1:no % Calculates the error and new W's for the output neurons
                eo(1,i) = Targets(j,i) - yo(1,i);
                do(1,i) = eo(1,i) .* yo(1,i) .* (1 - yo(1,i));
                dW = alpha .* yh .* do(1,i);
                Wo(i,:) = Wo(i,:) + dW;
            end
            for i = 1:nh % Calculates the new W's for the hidden neurons
                dh(1,i) = Wo(:,i)' * do(1,:)' .* (1 - yh(1,i));
                dW = alpha .* Features(j,:) .* dh(1,i);
                Wh(i,:) = Wh(i,:) + dW;
            end
            MSEtr(1,j) = eo(1,:) * eo(1,:)' / no; % Calculates the MSE per sample
        end
        MeanSquaredErrortr(1,k) = mean(MSEtr); % Calculates the mean of the MSE per epoch
        for j = 1:vas % Calculates the error for every sample in de validation set
            for i = 1:nh % Calculates the outputs of the hidden neurons
                yh(1,i) = perceptron(Features(trs+j,:),Wh(i,:));
            end
            for i = 1:no % Calculates the outputs of the output neurons
               yo(1,i) = perceptron(yh,Wo(i,:));
            end
            for i = 1:no % Calculats the error for the output neurons
                eo(1,i) = Targets(trs+j,i) - yo(1,i);
            end
            MSEva(1,j) = eo(1,:) * eo(1,:)' / no; % Calculates the MSE per sample
        end
        MeanSquaredErrorva(1,k) = mean(MSEva); % Calculates the mean of the MSE per epoch
        for j = 1:tes
            for i = 1:nh % Calculates the outputs of the hidden neurons
                yh(1,i) = perceptron(Features(vas+j,:),Wh(i,:));
            end
            for i = 1:no % Calculates the outputs of the output neurons
               yo(1,i) = perceptron(yh,Wo(i,:));
            end
            for i = 1:no % Calculats the error for the output neurons
                eo(1,i) = Targets(vas+j,i) - yo(1,i);
            end
            MSEte(1,j) = eo(1,:) * eo(1,:)' / no; % Calculates the MSE per sample
            [~, b] =  max(yo(1,:)); % Sets b to the output class
            PredictedClass(1,j) = b;
        end
        MeanSquaredErrorte(1,k) = mean(MSEte);
        if mod(k,10) == 0 % After each 10 epochs, halfs alpha
             alpha = alpha / 2;
        end
    end
    if out == 2
        Output = PredictedClass(1,:);
    else
    Output = [MeanSquaredErrortr(1,:);MeanSquaredErrorva(1,:);MeanSquaredErrorte(1,:)];
    end
end