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

%--- W with the minimum binary loss:
w_min = {w_min_1; w_min_2; w_min_3};
%--- Minimum binary loss
binary_min = [minimum_binary_1 minimum_binary_2 minimum_binary_3];


% -- Multiclass classifier 
% binary loss
binary = 0;
for i=1:D
    % matrix x
    x = seeds(:,1:end-1); 
    x = [one x];
    t = seeds(:,end);  

    % vector v
    v = [dot(w_min{1,1}, x(i,:)) dot(w_min{2,1}, x(i,:)) dot(w_min{3,1}, x(i,:))];

    % argmax
    [argvalue, argmax] = max(v);

    % shows the class prediction
    class = argmax;
    if class ~= t(i)
        binary_i(i) = 1;
    else
        binary_i(i) = 0;
    end
        binary = binary + binary_i(i);
end
% average
binary = binary/D
