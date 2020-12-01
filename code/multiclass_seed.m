% -- Multiclass with perceptron
x = seeds(:,1:end-1);  

% defines the output for each class
t_class_1 = [ones(70,1); -1*ones(140,1)];                               
t_class_2 = [-1*ones(70,1); ones(70,1); -1*ones(70,1)]; 
t_class_3 = [-1*ones(140,1); ones(70,1)];                

% -- Losses class 1

% executes perceptron 20 times
for i = 1:20
    [w_normalized, u(i)] = perceptron(x, t_class_1);
    wi{i,1} = w_normalized;
end
[D,n] = size(x);
one = ones(D,1);
% adds colomn of one to x
x = [one x];
t = t_class_1; 


% binary loss
binary = [zeros(20,1)];
for j = 1:20
    for i = 1:D
        if t(i) * dot(wi{j,1}, x(i,:)) <= 0
            binary_i(i) = 1;
        else
            binary_i(i) = 0;
        end
        binary(j) = binary(j) + binary_i(i);
    end   
    % average
    binary(j) = binary(j)/D;
end

[minimum_binary_1, run_binary_1] = min(binary);
binary_min_1 = [minimum_binary_1 run_binary_1];
w_min_1 = wi{run_binary_1,1};

% plot
figure
x_axis = 1:1:20;
b = plot(x_axis, binary,'linewidth',2);
grid;
hold on

% -- Losses class 2

% executes perceptron 20 times
x = seeds(:,1:end-1);  
for i = 1:20
    [w_normalized, u(i)] = perceptron(x, t_class_2);
    wi{i,1} = w_normalized;
end
[D,n] = size(x);
one = ones(D,1);
% adds colomn of one to x
x = [one x];
t = t_class_2; 

% binary loss
binary = [zeros(20,1)];
for j = 1:20
    for i = 1:D
        if t(i) * dot(wi{j,1}, x(i,:)) <= 0
            binary_i(i) = 1;
        else
            binary_i(i) = 0;
        end
        binary(j) = binary(j) + binary_i(i);
    end   
    % average
    binary(j) = binary(j)/D;
end

[minimum_binary_2, run_binary_2] = min(binary);
binary_min_2 = [minimum_binary_2 run_binary_2];
w_min_2 = wi{run_binary_2,1};

% plot
x_axis = 1:1:20;
b = plot(x_axis, binary,'linewidth',2);
grid;
% -- Losses class 3

% executes perceptron 20 times
x = seeds(:,1:end-1);  
for i = 1:20
    [w_normalized, u(i)] = perceptron(x, t_class_3);
    wi{i,1} = w_normalized;
end
[D,n] = size(x);
one = ones(D,1);
% adds colomn of one to x
x = [one x];
t = t_class_3; 

% binary loss
binary = [zeros(20,1)];
for j = 1:20
    for i = 1:D
        if t(i) * dot(wi{j,1}, x(i,:)) <= 0
            binary_i(i) = 1;
        else
            binary_i(i) = 0;
        end
        binary(j) = binary(j) + binary_i(i);
    end   
    % average
    binary(j) = binary(j)/D;
end

[minimum_binary_3, run_binary_3] = min(binary);
binary_min_3 = [minimum_binary_3 run_binary_3];
w_min_3 = wi{run_binary_3,1};

% plot
x_axis = 1:1:20;
b = plot(x_axis, binary,'linewidth',2);
grid;
ylabel('Binary Loss'); 
xlabel('Execution #'); 
title('Empirical binary loss');
legend( 'Class 1', 'Class 2', 'Class 3')
print -depsc binaryLoss


%--- W with the minimum binary loss:
w_min = {w_min_1; w_min_2; w_min_3};
%--- Minimum binary loss
binary_min = [minimum_binary_1 minimum_binary_2 minimum_binary_3]


% -- Multiclass classifier 

% while(true)
%     % user input
%     prompt = 'Data point format: x = [x1 x2 x3 x4 x5 x6 x7]\nType x1 ';
%     x1 = input(prompt);
%     prompt = 'Type x2 ';
%     x2 = input(prompt);
%     prompt = 'Type x3 ';
%     x3 = input(prompt);
%     prompt = 'Type x4 ';
%     x4 = input(prompt);
%     prompt = 'Type x5 ';
%     x5 = input(prompt);
%     prompt = 'Type x6 ';
%     x6 = input(prompt);
%     prompt = 'Type x7 ';
%     x7 = input(prompt);
% 
%     % matrix x
%     x = [1 x1 x2 x3 x4 x5 x6 x7];
% 
%     % vector v
%     v = [dot(w_min{1,1}, x) dot(w_min{2,1}, x) dot(w_min{3,1}, x)]
% 
%     % argmax
%     [argvalue, argmax] = max(v);
% 
%     % shows the class prediction
%     class = argmax
% end