% -- runs the SGD for SoftSVM

t = [-1* ones(762,1); ones(610,1)];
x = [dataset(:,1:4)];
[D,n] = size(x);

%runs softSVM
lambda = [100, 10, 1, 0.1, 0.01, 0.001];

for i=1:6
    [w, binary_cell, hinge_cell] = softSVM(x, t,lambda(i));
    wi{i,1} = w;
    binary_cell_i{i,1} = binary_cell;
    hinge_cell_i{i,1} = hinge_cell;
end

% plot for lambda = 100
[value_l_100, position_l_100] = min(cell2mat(binary_cell_i{1,1}))
figure
plot(cell2mat(binary_cell_i{1,1}),'linewidth',3)
%ylim([0 :])
title('Binary loss for λ = 100');
ylabel('Binary loss'); 
xlabel('T'); 
print -depsc bigB
figure
plot(cell2mat(hinge_cell_i{1,1}),'linewidth',3,'color','red')
%ylim([0 a])
title('Hinge loss for λ = 100');
ylabel('Hinge loss'); 
xlabel('T'); 
print -depsc bigH

% plot for lambda = 1
[value_l_1, position_l_1] = min(cell2mat(binary_cell_i{3,1}))
figure
plot(cell2mat(binary_cell_i{3,1}),'linewidth',3)
%ylim([0 :])
title('Binary loss for λ = 1');
ylabel('Binary loss'); 
xlabel('T'); 
print -depsc midB
figure
plot(cell2mat(hinge_cell_i{3,1}),'linewidth',3,'color','red')
%ylim([0 1])
title('Hinge losses for λ = 1');
ylabel('Hinge loss'); 
xlabel('T'); 
print -depsc midH

% plot for lambda = 0.1
[value_l_01, position_l_01] = min(cell2mat(binary_cell_i{6,1}))
figure
plot(cell2mat(binary_cell_i{4,1}),'linewidth',3)
%ylim([0 1])
title('Binary loss for λ = 0.01');
ylabel('Binary loss'); 
xlabel('T'); 
print -depsc smallB
figure
plot(cell2mat(hinge_cell_i{4,1}),'linewidth',3,'color','red')
%ylim([0 1])
title('Hinge loss for λ = 0.01');
ylabel('Hinge loss'); 
xlabel('T'); 
print -depsc smallH


