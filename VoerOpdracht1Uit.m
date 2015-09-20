function VoerOpdracht1Uit
    % Opdracht 1 - 10
%     er = zeros(1,10); % Error of each training
%     Error = zeros(1,4); % Average error per amount of hidden neurons
%     for j = 1:4 % Computes the error per amount of hidden neurons
%         for i = 1:10 % Computes the error of 10 trainings
%             y = Opdracht1(j*7);
%             a = y(1,:);
%             b = y(2,:);
%             er = min(b);
%         end
%         Error(1,j) = mean(er)
%     end
%     x = (1:4) * 7
%     plot(x,Error)
    
    % Opdracht 1 - 11
    y = Opdracht1(30);
    a = y(1,:);
    b = y(2,:);
    x = 1:length(a);
    plot(x,a,'b')
    hold on
    plot(x,b,'r')
end