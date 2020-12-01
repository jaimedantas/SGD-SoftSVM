% SoftSVM for x = (1,x) and w = (b,w)
function [w_last, binary, hinge] = softSVM(x, t, lambda)
    [D,n] = size(x);
    one = ones(D,1);
    % adds one and vector t
    x = [one x t];
    % shuffle the dataset
    random_x = x(randperm(D), :);
    
    % x and t shuffled
    x_shuffled = random_x(:,1:end-1);
    t_shuffled = random_x(:,end);
    
    % initial theta with b = 0 
    theta{1} = [0 zeros(1,n)];
    % T is 500
    updates = 500;
    for j = 1:updates
        w{j} = 1/(lambda*j) * theta{j};
        % ramdom index i
        i = randi(D);
        if t_shuffled(i) * dot(x_shuffled(i,:), w{j}) < 1 
         % updates theta
         theta{j+1} = theta{j} + t_shuffled(i) * x_shuffled(i,:);
        else
         theta{j+1} = theta{j};
        end
        
        % losses
        binary{j} = 0;
        hinge{j} = 0;
        for i = 1:D
            % binary
            if t_shuffled(i) * dot(w{j}, x_shuffled(i,:)) <= 0
                binary_i(i) = 1;
            else
                binary_i(i) = 0;
            end
            binary{j} = binary{j} + binary_i(i);
            % hinge
            hinge_i(i) = max(0,1 - t_shuffled(i) * dot(w{j}, x_shuffled(i,:)));
            hinge{j} = hinge{j} + hinge_i(i);
        end
        % average
        binary{j} = binary{j}/D;
        hinge{j} = hinge{j}/D;
    end
    w_last = w{updates};
end
